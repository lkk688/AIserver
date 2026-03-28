from __future__ import annotations

"""
tool_router.py

Responsibilities
----------------
Central router for non-mutation tools.

This module:
1. Looks up the currently ACTIVE tool from GLOBAL_TOOL_REGISTRY.
2. Resolves its handler_name.
3. Dispatches to the concrete handler implementation.

What this router does NOT do
----------------------------
- It does not apply file mutations (write / replace / diff).
- It does not parse raw LLM text.
- It does not own heavy business logic for OCR, web parsing, or document indexing.
  Those belong to dedicated tool modules.

Dispatch order
--------------
1. Dynamic tools
2. Builtin active tools from GLOBAL_TOOL_REGISTRY
3. Optional domain tools (reserved for later)
"""

import ast
import asyncio
import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from BatchAgent.tools.tool_registry_runtime import GLOBAL_TOOL_REGISTRY
from BatchAgent.tools.document_tools import DocumentToolManager
from BatchAgent.tools.local_tools import (
    search_code,
    find_file,
    read_file_chunk,
    list_directory,
    run_bash_command,
    run_shell,
)
from BatchAgent.tools.web_tools import (
    fetch_and_parse_url,
    perform_domain_aware_search,
)
from BatchAgent.working_memory import (
    WorkingMemory,
    handle_get_memory,
    handle_update_memory,
)

try:
    from rich.console import Console
    console = Console()
except Exception:
    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = _DummyConsole()


@dataclass
class ToolRouterContext:
    """
    Lightweight execution context for ToolRouter.
    """
    config: Any
    dynamic_tools_mapping: Dict[str, str]
    dynamic_tools_registry: Dict[str, Dict[str, Any]]


class ToolRouter:
    """
    Central router for active observation / non-mutation tools.

    Notes
    -----
    - Only ACTIVE tools in GLOBAL_TOOL_REGISTRY should be validated and dispatched.
    - Mutation tools must never execute here.
    - Dynamic tools are checked before builtin tools.
    """

    def __init__(
        self,
        config: Any,
        dynamic_tools_mapping: Optional[Dict[str, str]] = None,
        document_tools: Optional[DocumentToolManager] = None,
    ):
        self.ctx = ToolRouterContext(
            config=config,
            dynamic_tools_mapping=dynamic_tools_mapping or {},
            dynamic_tools_registry=getattr(config, "dynamic_tools_registry", {}) if hasattr(config, "dynamic_tools_registry") else {},
        )

        self.document_tools = document_tools or DocumentToolManager(config)

        # handler_name -> callable
        self.handler_registry: Dict[str, Callable[[Dict[str, Any]], str]] = {
            # local / observation
            "web_search": self._handle_web_search,
            "read_url": self._handle_read_url,
            "search_code": self._handle_search_code,
            "find_file": self._handle_find_file,
            "read_file_chunk": self._handle_read_file_chunk,
            "list_directory": self._handle_list_directory,
            "run_bash_command": self._handle_run_bash_command,
            "json_parse_error": self._handle_json_parse_error,
            "finish_task": self._handle_finish_task,

            # document
            "get_document_overview": self._handle_get_document_overview,
            "read_document_section": self._handle_read_document_section,
            "search_document": self._handle_search_document,

            # memory
            "get_memory": self._handle_get_memory,
            "update_memory": self._handle_update_memory,

            # meta
            "load_domain_tools": self._handle_load_domain_tools,
            "register_custom_tool": self._handle_register_custom_tool,

            # parallel / brainstorm (active only when enable_parallel=True)
            "execute_parallel_branches": self._handle_execute_parallel_branches,
        }

        self._validate_registry_bindings()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_registry_bindings(self):
        """
        Validate that all CURRENTLY ACTIVE non-mutation tools in the runtime registry
        have a corresponding handler implementation in this router.

        Important
        ---------
        We validate ONLY active_tools(), not all_tools(), otherwise we will falsely
        fail on tools that are known statically but intentionally inactive in this run.
        """
        missing_handlers = []
        used_handler_names: Set[str] = set()

        for tool in GLOBAL_TOOL_REGISTRY.active_tools():
            if not tool.handler_name:
                continue

            used_handler_names.add(tool.handler_name)

            if tool.category == "mutation":
                continue

            if tool.handler_name not in self.handler_registry:
                missing_handlers.append((tool.name, tool.handler_name))

        if missing_handlers:
            details = ", ".join(f"{tool}->{handler}" for tool, handler in missing_handlers)
            raise RuntimeError(
                f"ToolRouter missing handler implementations for active tools: {details}"
            )

        _unused_handlers = set(self.handler_registry.keys()) - used_handler_names
        # Kept silent intentionally.

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        """
        Dispatch one tool call.

        Order:
        1. dynamic tools
        2. active builtin tools from runtime registry

        Returns a plain text result string.
        """
        # 1. dynamic tools
        if name in self.ctx.dynamic_tools_mapping:
            return self._run_dynamic_tool(name, args)

        # 2. active builtin tools
        tool = GLOBAL_TOOL_REGISTRY.get(name)
        if tool is None:
            return f"Error: Unknown or inactive tool '{name}'"

        handler_name = tool.handler_name
        if not handler_name:
            return f"Error: No handler configured for tool '{name}'"

        if tool.category == "mutation":
            return f"Error: Tool '{name}' is a mutation tool and should not be dispatched via ToolRouter."

        handler = self.handler_registry.get(handler_name)
        if handler is None:
            return f"Error: Handler '{handler_name}' not found for tool '{name}'"

        try:
            return handler(args)
        except Exception as e:
            return f"Error executing tool '{name}': {e}"

    # ------------------------------------------------------------------
    # Dynamic tool execution
    # ------------------------------------------------------------------

    def _run_dynamic_tool(self, name: str, args: Dict[str, Any]) -> str:
        """
        Execute a dynamically registered custom tool script.

        Convention:
        - script path is relative to workspace_dir
        - JSON args are passed as argv[1]
        """
        script_path = self.ctx.dynamic_tools_mapping[name]
        workspace_dir = Path(getattr(self.ctx.config, "workspace_dir", Path(".")))
        target_script = workspace_dir / script_path

        if not target_script.exists():
            return f"System Error: Cannot find the script '{script_path}'. Did you write the file first?"

        args_json = json.dumps(args).replace("'", "'\\''")
        cmd = f"python3 {script_path} '{args_json}'"

        code, out = run_shell(
            cmd,
            cwd=str(workspace_dir),
            cap=10000,
            sandbox_container=getattr(self.ctx.config, "sandbox_container", None),
        )

        if code == 0:
            return out.strip()
        return f"[Custom Tool Execution Error (Exit {code})]:\n{out}"

    # ------------------------------------------------------------------
    # Builtin handlers: web / local
    # ------------------------------------------------------------------

    def _handle_web_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        category = args.get("category", "general")
        configured_time = (
            getattr(self.ctx.config, "current_time", "")
            or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        cfg = self.ctx.config
        return perform_domain_aware_search(
            query=query,
            category=category,
            serper_api_key=getattr(cfg, "serper_api_key", ""),
            current_time=configured_time,
            enable_youtube=getattr(cfg, "enable_youtube", False),
            tavily_api_key=getattr(cfg, "tavily_api_key", ""),
            document_search_fn=self._document_search_proxy,
            embedding_base_url=getattr(cfg, "embedding_base_url", ""),
            embedding_api_key=getattr(cfg, "embedding_api_key", ""),
            embedding_model=getattr(cfg, "embedding_model", ""),
            reranker_base_url=getattr(cfg, "reranker_base_url", ""),
            reranker_api_key=getattr(cfg, "reranker_api_key", ""),
            reranker_model=getattr(cfg, "reranker_model", ""),
        )

    def _document_search_proxy(self, query: str, top_k: int) -> str:
        try:
            return self.document_tools.search_document(query=query, top_k=top_k)
        except Exception:
            return ""

    def _handle_read_url(self, args: Dict[str, Any]) -> str:
        return fetch_and_parse_url(
            args.get("url", ""),
            offset=args.get("offset", 0),
            limit=args.get("limit", 200),
            name_contains=args.get("name_contains", ""),
        )

    def _handle_search_code(self, args: Dict[str, Any]) -> str:
        return search_code(args.get("query", ""))

    def _handle_find_file(self, args: Dict[str, Any]) -> str:
        return find_file(args.get("pattern", ""))

    def _handle_read_file_chunk(self, args: Dict[str, Any]) -> str:
        return read_file_chunk(
            filepath=args.get("filepath", ""),
            start_line=int(args.get("start_line", 1)),
            end_line=int(args.get("end_line", 1000)),
        )

    def _handle_list_directory(self, args: Dict[str, Any]) -> str:
        return list_directory(args.get("dir_path", "."))

    def _handle_run_bash_command(self, args: Dict[str, Any]) -> str:
        cmd = args.get("command", "")
        if len(cmd) > 300 and ("cat >" in cmd or "echo " in cmd):
            return (
                "System Guardrail: Command too long. "
                "Writing files via bash is strictly forbidden. Use write_file or search_and_replace instead."
            )
        return run_bash_command(cmd)

    def _handle_json_parse_error(self, args: Dict[str, Any]) -> str:
        return args.get("error", "JSON Parse Error")

    def _handle_finish_task(self, args: Dict[str, Any]) -> str:
        summary = str(args.get("summary") or "").strip()
        return f"Task marked finished: {summary}" if summary else "Task marked finished."

    # ------------------------------------------------------------------
    # Document handlers
    # ------------------------------------------------------------------

    def _handle_get_document_overview(self, args: Dict[str, Any]) -> str:
        filepath = str(args.get("filepath") or "").strip()
        return self.document_tools.get_overview(filepath=filepath if filepath else None)

    def _handle_read_document_section(self, args: Dict[str, Any]) -> str:
        section_id = str(args.get("section_id") or "").strip()
        page = args.get("page", None)

        if section_id:
            return self.document_tools.read_section(section_id=section_id)

        if page is not None:
            return self.document_tools.read_page(int(page))

        return "Please provide either section_id or page."

    def _handle_search_document(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        top_k = int(args.get("top_k", 5))
        return self.document_tools.search_document(query=query, top_k=top_k)

    # ------------------------------------------------------------------
    # Memory handlers
    # ------------------------------------------------------------------

    def _get_working_memory(self) -> Optional[WorkingMemory]:
        """
        Fetch the WorkingMemory instance from config if available.
        """
        memory = getattr(self.ctx.config, "working_memory", None)
        if memory is None:
            return None
        if not isinstance(memory, WorkingMemory):
            return memory
        return memory

    def _handle_get_memory(self, args: Dict[str, Any]) -> str:
        """
        Use the actual working_memory.py API.
        """
        memory = self._get_working_memory()
        if memory is None:
            return "Memory unavailable."
        return handle_get_memory(memory, args)

    def _handle_update_memory(self, args: Dict[str, Any]) -> str:
        """
        Use the actual working_memory.py API.
        """
        memory = self._get_working_memory()
        if memory is None:
            return "Memory unavailable."
        return handle_update_memory(memory, args)

    # ------------------------------------------------------------------
    # Meta handlers
    # ------------------------------------------------------------------

    def _handle_load_domain_tools(self, args: Dict[str, Any]) -> str:
        domain = str(args.get("domain") or "").strip()
        if not domain:
            return "Please provide a domain name."
        return (
            f"Domain tool loading requested: {domain}. "
            f"Domain-specific activation is not enabled yet in this build."
        )

    def _handle_register_custom_tool(self, args: Dict[str, Any]) -> str:
        tool_name = args["tool_name"]
        description = args["description"]
        schema_properties = args["schema_properties"]
        required_args = args.get("required_args", [])
        script_path = args["script_path"]

        self.ctx.dynamic_tools_mapping[tool_name] = script_path
        self.ctx.dynamic_tools_registry[tool_name] = {
            "properties": schema_properties,
            "required": required_args,
            "category": "meta",
        }

        return (
            f"Registered custom tool '{tool_name}' with script '{script_path}'. "
            f"Description: {description}. Required args: {required_args}. "
            f"Schema properties: {schema_properties}"
        )

    # ------------------------------------------------------------------
    # Parallel branches
    # ------------------------------------------------------------------

    # If the merged inline results exceed this many characters, the full
    # content is stored in working_memory.knowledge.summaries and a compact
    # overview is returned instead, preventing context overflow.
    _BRANCH_INLINE_LIMIT = 8_000   # ~2 000 tokens
    _BRANCH_PREVIEW_LEN  = 600     # chars shown per branch in the overview
    _BRANCH_STORE_LEN    = 3_000   # chars stored per branch in memory

    def _handle_execute_parallel_branches(self, args: Dict[str, Any]) -> str:
        """
        Dispatch multiple read-heavy sub-tasks to independent LLM branches running concurrently.

        Design
        ------
        Each branch gets its own isolated LLM call with full tool access.  The
        branch LLM reads the content it needs (read_document_section, read_url,
        read_file_chunk, etc.) and returns a compact summary.  All branches run
        concurrently via asyncio.gather, exploiting vLLM's prefix-sharing to
        process the shared system/task prompt once across the batch.

        Why this matters
        ----------------
        Sequential tool calls pile raw content into the main context window.
        For long documents or many URLs, the context fills up before any writing
        can happen.  By offloading reading + summarisation to branches, only the
        distilled results return to the main context — preventing overflow.

        Overflow protection
        -------------------
        If the combined branch summaries still exceed _BRANCH_INLINE_LIMIT chars,
        each branch result is stored in working memory (knowledge.summaries) and
        only a compact overview is returned inline.  The main agent can call
        get_memory('knowledge') to access the stored summaries.
        """
        branches_raw = args.get("branches", [])
        branches = self._parse_branches(branches_raw)
        if not branches:
            return "execute_parallel_branches: no valid branches provided."

        console.print(
            f"[cyan]🌿 Parallel branches: executing {len(branches)} branches concurrently…[/cyan]"
        )

        results: List[Tuple[str, str]] = asyncio.run(
            self._gather_branches(branches)
        )

        total_chars = sum(len(r) for _, r in results)

        if total_chars <= self._BRANCH_INLINE_LIMIT:
            # Small enough — return everything inline
            parts = []
            for branch_id, result in results:
                parts.append(f"### Branch `{branch_id}`\n{result}")
            return "\n\n".join(parts)

        # ── Overflow path: store in memory, return compact overview ────────
        console.print(
            f"[yellow]⚠️  Branch results too large ({total_chars:,} chars). "
            f"Storing in memory, returning overview.[/yellow]"
        )
        self._store_branches_in_memory(results)

        lines = [
            f"⚠️  Branch results too large to return inline "
            f"(~{total_chars // 4:,} tokens total). "
            "Stored in memory. Call `get_memory('knowledge')` to retrieve summaries.\n",
            "Branch overview:",
        ]
        for branch_id, result in results:
            clean = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            size_tok = max(1, len(result) // 4)
            tok_str = f"~{size_tok/1000:.1f}k tok" if size_tok >= 1000 else f"~{size_tok} tok"
            preview = clean[: self._BRANCH_PREVIEW_LEN].replace("\n", " ")
            if len(clean) > self._BRANCH_PREVIEW_LEN:
                preview += "…"
            lines.append(f"- `{branch_id}` ({tok_str}): {preview}")

        lines.append(
            "\nTo get full content: call `read_document_section` (or other tools) "
            "directly for specific sections, or call `get_memory('knowledge')` for "
            "the stored summaries."
        )
        return "\n".join(lines)

    def _store_branches_in_memory(self, results: List[Tuple[str, str]]) -> None:
        """Persist branch results (truncated) to working_memory.knowledge.summaries."""
        memory = getattr(self.ctx.config, "working_memory", None)
        if memory is None:
            return
        summaries: Dict[str, str] = {}
        for branch_id, result in results:
            clean = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            summaries[f"branch_{branch_id}"] = clean[: self._BRANCH_STORE_LEN]
        try:
            memory.update_memory({"knowledge": {"summaries": summaries}})
        except Exception:
            pass  # Memory unavailable — fail silently

    # ------------------------------------------------------------------
    # Parallel branch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_branches(raw: Any) -> List[Dict[str, str]]:
        """Parse the `branches` argument from JSON string, Python repr, or list."""
        if isinstance(raw, list):
            return raw
        if not isinstance(raw, str):
            return []
        raw = raw.strip()
        # Try JSON first (double-quoted), then Python literal (single-quoted)
        for loader in (json.loads, ast.literal_eval):
            try:
                result = loader(raw)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        return []

    async def _gather_branches(
        self, branches: List[Dict[str, str]]
    ) -> List[Tuple[str, str]]:
        """Run all branches concurrently and return (branch_id, result) pairs."""
        tasks = [
            self._run_branch_async(
                branch_id=str(b.get("branch_id", f"branch_{i}")),
                instruction=str(b.get("instruction", "")),
            )
            for i, b in enumerate(branches)
        ]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for i, (b, res) in enumerate(zip(branches, raw)):
            bid = str(b.get("branch_id", f"branch_{i}"))
            if isinstance(res, Exception):
                results.append((bid, f"Error: {res}"))
            else:
                results.append((bid, str(res)))
        return results

    @staticmethod
    def _make_branch_client(config: Any) -> Any:
        """
        Create a fresh, isolated HTTP client for branch calls.

        We intentionally do NOT reuse config.client here because that client's
        underlying httpx connection pool is bound to the main event loop.
        Running asyncio.run() inside asyncio.to_thread() creates a new event
        loop in the worker thread; sharing the main-loop client across loops
        causes connection-pool interference and transient connection errors on
        the next main-agent LLM call.

        Connection params (base_url, api_key) are read from the existing
        config.client since AgentConfig does not store them directly.
        """
        provider = getattr(config, "provider", "openai")
        existing = config.client  # already-constructed async client

        if provider == "anthropic":
            from anthropic import AsyncAnthropic
            api_key = getattr(existing, "api_key", None) or "EMPTY"
            return AsyncAnthropic(api_key=api_key)

        import httpx
        from openai import AsyncOpenAI

        # AsyncOpenAI exposes base_url as a httpx.URL; convert to str.
        raw_url = getattr(existing, "base_url", None)
        base_url = str(raw_url) if raw_url is not None else None
        api_key = getattr(existing, "api_key", None) or "EMPTY"

        kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "http_client": httpx.AsyncClient(timeout=300.0),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    async def _run_branch_async(self, branch_id: str, instruction: str) -> str:
        """
        One branch execution:
          1. Send the instruction to the LLM (non-streaming, tools enabled).
          2. Execute any returned tool calls via self.dispatch().
          3. Return tool results (or LLM text if no tools were called).

        Uses a fresh HTTP client (see _make_branch_client) to avoid
        polluting the main agent's connection pool.
        """
        config = self.ctx.config
        model = config.model
        provider = getattr(config, "provider", "openai")
        temperature = getattr(config, "temperature", 0.1)
        max_context = getattr(config, "max_context", 32768)
        backend = getattr(config, "backend", "openai")
        enable_thinking = getattr(config, "enable_thinking", False)

        # Fresh client isolated from main agent's connection pool
        client = self._make_branch_client(config)

        # Read-only tools only (no mutation)
        read_tools = [
            t.to_openai_function()
            for t in GLOBAL_TOOL_REGISTRY.active_tools()
            if t.category != "mutation"
        ]

        safe_max = min(4096, max(512, max_context // 8))
        messages = [
            {
                "role": "user",
                "content": (
                    f"[Parallel Branch: {branch_id}]\n\n{instruction}\n\n"
                    "Execute this instruction using the available tools. "
                    "Return a detailed, complete result."
                ),
            }
        ]

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": safe_max,
            "stream": False,
        }
        if backend in ("llama.cpp", "vllm"):
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
            kwargs["stop"] = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        if read_tools:
            kwargs["tools"] = read_tools

        # Anthropic uses a different client API
        if provider == "anthropic":
            anthropic_tools = [
                t.to_anthropic_tool()
                for t in GLOBAL_TOOL_REGISTRY.active_tools()
                if t.category != "mutation"
            ]
            return await self._run_branch_anthropic(
                client, model, instruction, branch_id, anthropic_tools, safe_max, temperature
            )

        try:
            resp = await client.chat.completions.create(**kwargs)
        except Exception as e:
            return f"LLM call failed: {e}"

        choice = resp.choices[0] if resp.choices else None
        if not choice:
            return "No response from LLM."

        text_content: str = choice.message.content or ""
        native_tool_calls = getattr(choice.message, "tool_calls", None) or []

        tool_results = []
        for tc in native_tool_calls:
            try:
                tool_name = tc.function.name
                args_str = tc.function.arguments
                try:
                    tool_args = json.loads(args_str)
                except Exception:
                    tool_args = {}
                result = self.dispatch(tool_name, tool_args)
                tool_results.append(f"**{tool_name}** →\n{result}")
            except Exception as exc:
                tool_results.append(f"Tool error: {exc}")

        parts = []
        if text_content:
            clean = re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()
            if clean:
                parts.append(clean)
        parts.extend(tool_results)
        return "\n\n".join(parts) if parts else "No output."

    async def _run_branch_anthropic(
        self,
        client: Any,
        model: str,
        instruction: str,
        branch_id: str,
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Anthropic-specific single-shot branch call."""
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"[Parallel Branch: {branch_id}]\n\n{instruction}\n\n"
                            "Execute this instruction and return a detailed result."
                        ),
                    }
                ],
                tools=tools if tools else [],
            )
        except Exception as e:
            return f"LLM call failed: {e}"

        parts = []
        for block in getattr(resp, "content", []):
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                try:
                    result = self.dispatch(block.name, block.input or {})
                    parts.append(f"**{block.name}** →\n{result}")
                except Exception as exc:
                    parts.append(f"Tool error: {exc}")
        return "\n\n".join(parts) if parts else "No output."

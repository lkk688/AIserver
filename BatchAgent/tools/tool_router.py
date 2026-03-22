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

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

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

            # optional / placeholders
            "execute_parallel_branches": self._handle_execute_parallel_branches,
            "inspect_branch_details": self._handle_inspect_branch_details,
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

        return perform_domain_aware_search(
            query=query,
            category=category,
            serper_api_key=getattr(self.ctx.config, "serper_api_key", ""),
            current_time=configured_time,
            enable_youtube=getattr(self.ctx.config, "enable_youtube", False),
            tavily_api_key=getattr(self.ctx.config, "tavily_api_key", ""),
            document_search_fn=self._document_search_proxy,
        )

    def _document_search_proxy(self, query: str, top_k: int) -> str:
        try:
            return self.document_tools.search_document(query=query, top_k=top_k)
        except Exception:
            return ""

    def _handle_read_url(self, args: Dict[str, Any]) -> str:
        return fetch_and_parse_url(args.get("url", ""))

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
    # Optional / placeholders
    # ------------------------------------------------------------------

    def _handle_execute_parallel_branches(self, args: Dict[str, Any]) -> str:
        return "Parallel branch execution is not implemented in this router yet."

    def _handle_inspect_branch_details(self, args: Dict[str, Any]) -> str:
        branch_id = args.get("branch_id", "")
        return f"Branch inspection not implemented yet for branch_id='{branch_id}'."
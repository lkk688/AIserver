from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BatchAgent.tools.tools_registry import get_base_tools


class PromptRegistry:
    """
    Prompt registry for system prompt and task prompt generation.

    Responsibilities
    ----------------
    - Generate strategy-aware system prompt
    - Render tool usage instructions from CURRENT ACTIVE TOOLS
    - Keep prompt generation aligned with runtime registry activation
    - Support dynamic/custom tools automatically via get_base_tools(...)
    """

    # ------------------------------------------------------------------
    # Tool rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_to_xml_example(tool: Dict[str, Any]) -> str:
        """
        Render one tool as XML child-tag example.

        Example:
            <tool_call><web_search><query>...</query></web_search></tool_call>
        """
        name = tool["name"]
        props = tool.get("properties", {}) or {}
        inner = "".join(f"<{k}>...</{k}>" for k in props.keys())
        return f"`<tool_call><{name}>{inner}</{name}></tool_call>`"

    @staticmethod
    def _has_tool(base_tools_list: List[Dict[str, Any]], tool_name: str) -> bool:
        return any(t.get("name") == tool_name for t in base_tools_list)

    @staticmethod
    def _split_tools(base_tools_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split active tools into logical buckets for prompt rendering.
        """
        mutation_names = {"write_file", "search_and_replace"}
        finish_names = {"finish_task"}

        observation = []
        mutation = []
        finish = []
        meta = []

        for t in base_tools_list:
            name = t.get("name", "")
            category = t.get("category", "observation")

            if name in mutation_names:
                mutation.append(t)
            elif name in finish_names:
                finish.append(t)
            elif category == "meta":
                meta.append(t)
            else:
                observation.append(t)

        return {
            "observation": observation,
            "mutation": mutation,
            "finish": finish,
            "meta": meta,
        }

    @classmethod
    def _resolve_base_tools(
        cls,
        strategy: str,
        base_tools_list: Optional[List[Dict[str, Any]]] = None,
        enable_parallel: bool = False,
        domain: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        Resolve the current active tool list.

        Priority:
        1. externally supplied base_tools_list
        2. tools_registry.get_base_tools(...) from runtime active registry
        """
        if base_tools_list is not None:
            return base_tools_list
        return get_base_tools(
            strategy=strategy,
            enable_parallel=enable_parallel,
            domain=domain,
        )

    @classmethod
    def _render_text_only_tool_docs(cls, base_tools_list: List[Dict[str, Any]]) -> str:
        """
        Render XML tool docs for text_only mode from CURRENT ACTIVE TOOLS.
        """
        buckets = cls._split_tools(base_tools_list)

        obs_tools = buckets["observation"] + buckets["meta"]
        mut_tools = buckets["mutation"]
        done_tools = buckets["finish"]

        obs_section = "\n".join(
            f"  {cls._tool_to_xml_example(t)}\n    ↳ {t['description']}"
            for t in obs_tools
        ).strip()

        mut_section = "\n".join(
            f"  {cls._tool_to_xml_example(t)}\n    ↳ {t['description']}"
            for t in mut_tools
        ).strip()

        done_section = "\n".join(
            f"  {cls._tool_to_xml_example(t)}\n    ↳ {t['description']}"
            for t in done_tools
        ).strip()

        parts = [
            "## 1. Execution Mode: TEXT_ONLY",
            "You do NOT have native JSON tools. You must output specific XML formats in your text to interact.",
            "",
            "### Using Interactive Tools",
            "To gather info, inspect files, search, read documents, or execute safe runtime actions, output EXACTLY this XML format:",
            "",
            obs_section if obs_section else "  (No observation tools active in this session)",
            "",
        ]

        if mut_section:
            parts.extend([
                "### Writing / Editing Files",
                mut_section,
                "",
            ])

        if done_section:
            parts.extend([
                "### Completing the Task",
                done_section,
                "",
            ])

        parts.extend([
            "⚠️ CRITICAL FORMATTING RULES:",
            "- ❌ WRONG: `web_search(query='...')` (Python syntax — rejected)",
            "- ❌ WRONG: `<web_search query=\"...\">` (XML attributes — rejected)",
            "- ✅ CORRECT: `<tool_call><web_search><query>...</query></web_search></tool_call>`",
            "- ✅ CORRECT: `<tool_call><read_file_chunk><filepath>...</filepath><start_line>1</start_line><end_line>50</end_line></read_file_chunk></tool_call>`",
            "- NEVER invent tool names or parameter tags. Use EXACTLY the tag names shown above.",
            "",
        ])

        return "\n".join(parts)

    @classmethod
    def _render_hybrid_mode_docs(cls, base_tools_list: List[Dict[str, Any]]) -> str:
        """
        Render hybrid-mode docs.

        All tools — including mutation tools — are available as native JSON tool calls.
        The system also accepts XML text mutations as a fallback, but the preferred
        path for all tools is the native JSON API.
        """
        return (
            "## 1. Execution Mode: HYBRID\n"
            "You are operating with full native JSON Function/Tool Calling support.\n"
            "- Use the provided JSON tools for ALL interactions — observations, file writes, "
            "and file edits alike.\n"
            "- Do NOT use markdown code blocks to write code. Use the proper file mutation "
            "tools (write_file, search_and_replace) when available.\n"
            "- For general chat or thinking, reply in plain text.\n\n"
        )

    @classmethod
    def _render_native_mode_docs(cls) -> str:
        """
        Render native_all mode docs.
        """
        return (
            "## 1. Execution Mode: NATIVE_ALL\n"
            "You are operating in an environment that fully supports native JSON Function/Tool Calling.\n"
            "- You MUST use the provided JSON functions (tools) for all tool-based interactions.\n"
            "- Do NOT use markdown code blocks to write code. Use the proper file mutation tools when available.\n"
            "- For general chat or thinking, reply in plain text.\n\n"
        )

    # ------------------------------------------------------------------
    # Main system prompt
    # ------------------------------------------------------------------

    @classmethod
    def get_system_prompt(
        cls,
        strategy: str,
        base_tools_list: Optional[List[Dict[str, Any]]] = None,
        domain: str = "general",
        location: str = "California, United States",
        current_time: str = "",
        enable_parallel: bool = False,
    ) -> str:
        """
        Generate the system prompt based on current runtime-active tools.

        Notes
        -----
        - If base_tools_list is None, this method resolves the active tool list
          from tools_registry.get_base_tools(...), which in turn reads the
          runtime registry.
        - This keeps prompt instructions aligned with actual active tools.
        """
        base_tools_list = cls._resolve_base_tools(
            strategy=strategy,
            base_tools_list=base_tools_list,
            enable_parallel=enable_parallel,
            domain=domain,
        )

        resolved_current_time = (
            current_time.strip()
            if isinstance(current_time, str) and current_time.strip()
            else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        )

        base_prompt = (
            f"You are an elite, general-purpose AI Agent.\n"
            f"**Current System Time**: {resolved_current_time}\n"
            f"**Assumed Location**: {location}\n\n"
            f"⚠️ **ENVIRONMENT RULES (CRITICAL)**:\n"
            f"- You are already inside the target workspace directory.\n"
            f"- ALL file paths you read, write, or execute should be RELATIVE to your current directory when possible (e.g., `./script.py`).\n"
            f"- NEVER use `cd` in bash commands. Execute directly (e.g., `python3 script.py`, not `cd dir && python3 script.py`).\n\n"
            f"- NO HUMAN INTERACTION: You are running in an autonomous loop. Do NOT ask the user for clarification, permission, or follow-up questions.\n"
            f"- Make reasonable assumptions and proceed autonomously using tools until the goal is fully achieved.\n\n"
        )

        if domain != "general":
            base_prompt += (
                f"**Domain Profile**: You are operating as a specialist in the '{domain.upper()}' domain.\n\n"
            )

        if strategy == "native_all":
            mode_prompt = cls._render_native_mode_docs()
        elif strategy == "hybrid":
            mode_prompt = cls._render_hybrid_mode_docs(base_tools_list)
        elif strategy == "text_only":
            mode_prompt = cls._render_text_only_tool_docs(base_tools_list)
        else:
            mode_prompt = ""

        has_finish_task = cls._has_tool(base_tools_list, "finish_task")

        general_rules = (
            "## 2. Information Gathering & Verification Rules (CRITICAL)\n"
            "1. **Never Guess Code**: Check provided file context first. If missing, use tools to read files.\n"
            "2. **Search for Unknowns**: Use `web_search` for unknown libraries, APIs, or recent information when that tool is active.\n"
            "3. **Context-Aware Verification**:\n"
            "   - **For Executable Code**: After modifying scripts, you MUST run a verification command when the appropriate tools are active.\n"
            "   - **For Documents/Reports**: (e.g., `.md`, `.txt`) Do NOT attempt to execute them. Once the writing tool reports success, consider them verified.\n"
        )

        if has_finish_task:
            general_rules += (
                "4. **End of Task**: The moment you have fully achieved the user's goal, you MUST immediately call the `finish_task` tool in your very next response.\n"
            )
        else:
            general_rules += (
                "4. **End of Task**: Once the goal is fully achieved, provide a concise completion response.\n"
            )

        cot_rules = (
            "## Chain-of-Thought Protocol\n"
            "- If you need to think before acting, enclose your thoughts within `<think>...</think>` tags.\n"
            "- CRITICAL: Your actual tool call (JSON or XML) MUST be placed OUTSIDE and AFTER the `<think>...</think>` tags, otherwise the system cannot execute it.\n\n"
        )

        return base_prompt + cot_rules + mode_prompt + general_rules

    # ------------------------------------------------------------------
    # Task prompt
    # ------------------------------------------------------------------

    @staticmethod
    def format_task(
        goal: str,
        allowlist: List[str],
        context_files: List[str],
        workspace_dir: str,
        content_injector,
    ) -> str:
        """
        Format the task prompt with workspace and file context.
        """
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Absolute Directory Path: `{workspace_dir}`\n"
            f"Working Directory: `./` (You are already here. Do not try to cd into it.)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n\n"
        )

        files_md = content_injector(allowlist + context_files)
        if files_md:
            base_md += f"## Provided File Context\n{files_md}"

        return base_md

    # ------------------------------------------------------------------
    # Verification feedback prompts
    # ------------------------------------------------------------------

    @staticmethod
    def get_verification_escalation_prompt(n_fails: int) -> str:
        """
        Return an escalation hint based on consecutive failure count.
        Returns "" for n_fails < 2 (no escalation needed).
        """
        if n_fails >= 3:
            return (
                f"\n🚨 **SAME ERROR {n_fails}x IN A ROW — "
                "YOU MUST CHANGE YOUR APPROACH**\n\n"
                "Incremental `search_and_replace` patches are NOT fixing this bug.\n"
                "**Mandatory action**: Rewrite the entire affected function "
                "from scratch using `write_file`.\n"
                "- Think carefully about the root cause BEFORE writing.\n"
                "- Do NOT copy the old broken logic.\n"
                "- Write a completely new, correct implementation.\n"
            )
        if n_fails == 2:
            return (
                f"\n⚠️ Same error **{n_fails} times in a row**.\n"
                "Your fix is not working. Consider rewriting the affected "
                "function entirely instead of patching it.\n"
            )
        return ""

    @staticmethod
    def get_pattern_hint(out: str, written_src: str = "") -> str:
        """
        Return a pattern-specific coaching hint based on the error output and
        (optionally) the source of the first written Python file.

        Covered patterns
        ----------------
        - Numba NoneType callable  ('NoneType' object is not callable + njit/numba)
        - Multiprocessing hang     (No output for 30s | Hard cap reached)
        - AssertionError           (wrong algorithm output, model should fix logic not tests)
        """
        if "'NoneType' object is not callable" in out:
            if "njit" in written_src or "numba" in out or "numba" in written_src:
                return (
                    "\n## 💡 Hint: Numba Conditional Decorator\n"
                    "The error is caused by `@njit` being applied when numba is unavailable "
                    "(the fallback `njit` stub returns `None`, not a callable).\n"
                    "Replace the decorator with a conditional pattern:\n"
                    "```python\n"
                    "try:\n"
                    "    from numba import njit, prange\n"
                    "    HAS_NUMBA = True\n"
                    "except ImportError:\n"
                    "    HAS_NUMBA = False\n"
                    "    prange = range\n"
                    "    def njit(*args, **kwargs):\n"
                    "        return args[0] if args and callable(args[0]) else (lambda f: f)\n"
                    "\n"
                    "def _my_func(x):  # define WITHOUT decorator\n"
                    "    ...\n"
                    "\n"
                    "_my_func = njit(parallel=True, fastmath=True)(_my_func) if HAS_NUMBA else _my_func\n"
                    "```\n"
                    "Do NOT use `@njit(parallel=True)` as a class-level decorator when numba may be absent.\n"
                )
        if "No output for 30s — killed" in out or "Hard cap reached" in out:
            return (
                "\n## 💡 Hint: Multiprocessing Guard Required\n"
                "The script was killed after producing no output — this is almost always "
                "caused by multiprocessing code running without the "
                "`if __name__ == '__main__':` guard.\n\n"
                "**Fix**: Wrap ALL multiprocessing/Process/Pool creation in:\n"
                "```python\n"
                "if __name__ == '__main__':\n"
                "    # pool = multiprocessing.Pool(...)  ← must be inside this block\n"
                "    # p = Process(target=..., ...)      ← must be inside this block\n"
                "    main()\n"
                "```\n"
                "Without this guard each worker subprocess re-imports the module and "
                "spawns more workers recursively, hanging forever.\n"
            )
        if "AssertionError" in out:
            return (
                "\n## 💡 Hint: Fix the Algorithm, Not the Test\n"
                "An `AssertionError` means your implementation returned the **wrong answer**.\n"
                "The expected value in the test assertion is the ground truth — "
                "do NOT change it.\n"
                "- Read the assertion: `assert your_result == expected_value`\n"
                "- Fix the **algorithm/logic** in your function, not the `assert` line.\n"
                "- Trace through your code manually with the failing input to find the bug.\n"
            )
        return ""

    # ------------------------------------------------------------------
    # Execute-task feedback prompts
    # ------------------------------------------------------------------

    @staticmethod
    def finish_task_rejected(task_goal: str, found_kws: List[str]) -> str:
        return (
            "❌ **finish_task REJECTED**: Your task requires writing an output file "
            f"(goal matches: {found_kws}), "
            "but no output file was found on disk.\n\n"
            "Write the file first using `write_file`, then call `finish_task`.\n"
            "Use information already gathered — do NOT read more sections."
        )

    @staticmethod
    def json_parse_strategy_switch() -> str:
        return (
            "System switched strategy native_all → hybrid after repeated "
            "write_file JSON parse failures. Use native JSON for observations "
            "and XML write_file / search_and_replace for file mutations."
        )

    @staticmethod
    def no_action_warning(strategy: str) -> str:
        if strategy == "native_all":
            return (
                "⚠️ System Warning: No valid tool calls detected.\n"
                "Use native JSON function calling (e.g. `web_search`, `write_file`). "
                "Do NOT put JSON in plain-text code blocks."
            )
        if strategy == "text_only":
            return (
                "⚠️ System Warning: No valid XML tool calls detected.\n"
                "Use `<tool_call><web_search><query>…</query></web_search></tool_call>` "
                "or the XML write_file format."
            )
        return (
            "⚠️ System Warning: No valid tool calls or file modifications detected.\n"
            "Use native JSON tools for searching/reading and `write_file` / "
            "`search_and_replace` for file mutations. "
            "Call `finish_task` if done."
        )

    @staticmethod
    def domain_loaded(domain: str, tool_names: List[str]) -> str:
        return (
            f"✅ Loaded domain plugin '{domain}'.\n"
            f"New tools: {', '.join(tool_names)}. Use them to complete the task."
        )

    @staticmethod
    def domain_unknown(domain: str) -> str:
        return f"❌ Unknown domain plugin: '{domain}'."

    @staticmethod
    def custom_tool_registered(tool_name: str) -> str:
        return (
            f"✅ Custom tool `{tool_name}` registered and available immediately. "
            "Call it like any native tool."
        )

    @staticmethod
    def tool_retry_blocked(tool_name: str, err_summary: str) -> str:
        return (
            f"⚠️ **Tool unavailable**: `{tool_name}` previously failed: "
            f'"{err_summary}". Do NOT retry — proceed with what you have.'
        )

    @staticmethod
    def loop_detected(skipped_names: List[str]) -> str:
        return (
            f"⚠️ **Loop detected**: You already called {skipped_names} with identical "
            "arguments. Results already provided. Synthesize what you know — "
            "call `write_file` or `finish_task`."
        )

    @staticmethod
    def mutation_dedup(verify_failed: bool) -> str:
        if verify_failed:
            return (
                "⚠️ Repeated identical file mutation skipped — "
                "and the previous verification FAILED with the same fix.\n"
                "Your incremental edit is NOT fixing the bug. "
                "Read the error carefully and write a DIFFERENT, correct fix. "
                "Do NOT repeat the same search_and_replace."
            )
        return (
            "⚠️ Repeated identical file mutation skipped. "
            "The same write/edit was already applied. "
            "Call `finish_task` if the goal is completed."
        )

    @staticmethod
    def write_only_guidance(written_file: str) -> str:
        return (
            f"\n✅ **File written**: `{written_file}`\n\n"
            "**What to do next — choose exactly one:**\n"
            "1. **Output complete** → call `finish_task` NOW.\n"
            "2. **Placeholders remain** (`[PENDING:]`, `[TODO]`, etc.) → "
            "read or recall the content, then `search_and_replace` each marker.\n"
            "3. **More source content unread** → `update_memory` what you wrote, "
            "then continue the read→write→fill cycle.\n\n"
            "⚠️ Do NOT read more unless you have a specific placeholder to fill."
        )

    @staticmethod
    def pending_markers_report(markers: List[str]) -> str:
        """Report [PENDING:...] markers found in a written file."""
        marker_list = "\n".join(f"  - {m}" for m in markers[:10])
        return (
            f"\n📌 **Your draft has {len(markers)} pending marker(s)** — fill them next:\n"
            f"{marker_list}\n\n"
            "For each marker:\n"
            "1. Use `get_memory('knowledge')`, `read_document_section`, or `web_search` "
            "to get the content.\n"
            "2. Call `search_and_replace` to replace the marker with the real content.\n"
            "3. Repeat until all markers are filled, then call `finish_task`."
        )

    @staticmethod
    def parallel_branches_complete() -> str:
        """Injected immediately after execute_parallel_branches returns results."""
        return (
            "\n✅ **Parallel reading complete** — all branch results are now in memory.\n\n"
            "**NEXT STEP: Write the output NOW** using `write_file`.\n"
            "- Use `[PENDING: <topic>]` markers for any parts you haven't read yet.\n"
            "- Do NOT read more sections sequentially — everything you need is "
            "available via `get_memory('knowledge')` or specific section reads.\n"
            "- After writing the skeleton, fill each `[PENDING:]` marker with "
            "`search_and_replace` and the relevant content from memory.\n"
            "⚡ **Act now**: call `write_file` to create the output."
        )

    @staticmethod
    def read_only_guidance(output_file: Optional[str]) -> str:
        if output_file:
            return (
                f"\nResults above from your read. "
                f"Output file exists: `{output_file}`. "
                "If complete, call `finish_task` now. "
                "Only continue reading if you have a specific placeholder to fill."
            )
        return "\nAnalyze the results and continue."

    @staticmethod
    def empty_turn_fallback() -> str:
        return (
            "No new actions were executed. "
            "If the output file is already correct, call `finish_task` now."
        )

    @staticmethod
    def write_pressure(ctx_pct: int, output_file: Optional[str]) -> str:
        if not output_file:
            return (
                f"\n\n📋 **LONG-TASK STRATEGY ({ctx_pct}% context used — switch now)**\n\n"
                "You have read enough. **Stop reading. Create the output file NOW.**\n\n"
                "**Cycle: Read batch → Write/update → Track → Repeat**\n"
                "1. `write_file` with what you know; use `[PENDING: <section>]` markers.\n"
                "2. Read next batch → `search_and_replace` the marker.\n"
                "3. `update_memory` progress after each fill.\n"
                "4. Repeat until no markers remain.\n"
                "5. `finish_task` — do not keep reading once complete.\n\n"
                "⚡ **Act now**: call `write_file` to create the initial output."
            )
        return (
            f"\n\n📋 **CONTINUE LONG-TASK CYCLE ({ctx_pct}% context used)**\n"
            f"Output file: `{output_file}`.\n"
            "- Markers remain → read next batch, `search_and_replace` each one.\n"
            "- Output complete → `finish_task` NOW.\n"
            "Check `progress.completed_steps` in memory for what is already done."
        )

    @staticmethod
    def json_parse_soft_warning() -> str:
        return (
            "Tool JSON parsing failed for at least one call. "
            "If writing files, use the JSON write_file/search_and_replace tool."
        )
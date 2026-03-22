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

"""

"""
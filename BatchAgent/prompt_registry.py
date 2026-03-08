from typing import List
from pathlib import Path
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from BatchAgent.mini_batch_agent_libs import (
    now_stamp, ensure_dirs, run_shell, _determine_verify_cmd, 
    _handle_missing_modules, top_level_tree, read_file, estimate_tokens, truncate_to_tokens, robust_json_loads
)

# ==========================================
# 2. Prompt Registry
# ==========================================
class PromptRegistry_old:
    """
    Centralized manager for all LLM prompts.
    Defines strict output formats, verification rules, and tool usages.
    """
    @classmethod
    def get_system_prompt(cls, tools_list: list) -> str:
        """Dynamically generates the system prompt based on available tools."""
        base_prompt = (
            "You are an elite, general-purpose AI Agent. You can write code, author documentation, and answer complex questions.\n"
            "You operate in a structured generation environment.\n\n"
            "## Output Format (STRICT)\n"
            "You MUST output in ONE of these three formats per response. Never mix them.\n\n"
            "### Format A: Unified Diff (For small code edits)\n"
            "1. Start with `## Action` followed by a SINGLE fenced diff block (`diff --git ...`).\n\n"
            "### Format B: WRITE_FILE (For new files, full rewrites, or writing Documents)\n"
            "WRITE_FILE: path/to/file.ext\n"
            "<<<CONTENT\n...\nCONTENT>>>\n\n"
            "### Format C: Direct Response (For Q&A, Summaries, and Tutorials)\n"
            "Output directly in Markdown.\n\n"
            "## Information Gathering & Anti-Hallucination Rules (CRITICAL)\n"
            "1. **Never Guess Code**: Check the `Provided File Context` section first.\n"
            "2. **Read Before Act**: If a requested file is NOT in the context, you MUST use `read_file_chunk` or `find_file` before modifying it.\n"
            "3. **Search for Unknowns**: Use `web_search` for unknown libraries, APIs, or events.\n\n"
            "## Verification & Environment Rules (CRITICAL)\n"
            "1. **Mandatory Verification**: If you write executable code, include a verification command on a standalone line:\n"
            "   `Verification: <command>` (e.g., `Verification: python3 script.py`)\n"
            "2. **Environment Limitations**: If the environment lacks dependencies (e.g., requires Python 3.13 but environment has 3.12):\n"
            "   - Write the correct code anyway.\n"
            "   - Use Format C to provide a step-by-step setup tutorial.\n"
            "   - Do NOT output a `Verification:` command that is guaranteed to fail.\n\n"
        )

        tool_section = "## Interactive Tools\nIf you lack context, output an XML tool call to pause and wait for data:\n"
        for t in tools_list:
            name = t.get("name")
            desc = t.get("description", "")
            # Construct dummy XML example
            args_xml = "".join([f"<{k}>...</{k}>" for k in t.get("properties", {}).keys()])
            tool_section += f"`<tool_call><{name}>{args_xml}</{name}></tool_call>`\n  - {desc}\n"

        return base_prompt + tool_section

    @staticmethod
    def format_task(goal: str, allowlist: List[str], context_files: List[str], notes: str, workspace_dir: str) -> str:
        """Builds the main Turn Prompt for the user message."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        
        all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False
        format_hint = ""
        if not allowlist or all_new_files:
            format_hint = "\n> **HINT**: You are creating NEW files or answering a query. Use **Format B (WRITE_FILE)** or **Format C**.\n"
        elif len(allowlist) > 1:
            format_hint = "\n> **HINT**: You are modifying multiple files. Use **Format B (WRITE_FILE)** to avoid diff truncation.\n"

        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Directory: `./` (inside `{workspace_dir}/`)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n{format_hint}\n"
            f"## Notes & Constraints\n{notes if notes else '(none)'}\n\n"
            f"## Output Contract Reminder\n"
            f"1. Choose ONE format: Format A (Diff), Format B (Write), or Format C (Direct Answer).\n"
            f"2. If generating code, remember the `Verification: <cmd>` rule.\n"
        )
        
        files_md = ""
        for f in list(dict.fromkeys(allowlist + context_files)):
            content = read_file(str(f))
            if content and not content.startswith("[MISSING FILE]"):
                if estimate_tokens(content) > 8000:
                    content = truncate_to_tokens(content, 8000)
                files_md += f"### File: {f}\n```text\n{content}\n```\n"
                
        if files_md:
            base_md += f"\n## Provided File Context\n{files_md}"

        return base_md
    
class PromptRegistry:
    @classmethod
    def get_system_prompt(cls, strategy: str, tools_list: list) -> str:
        """
        Dynamically generates the system prompt based on the chosen Tool Strategy.
        """
        base_prompt = (
            "You are an elite, general-purpose AI Agent.\n"
            "## Information Gathering Rules (CRITICAL)\n"
            "1. **Never Guess Code**: Check `Provided File Context` first.\n"
            "2. **Read Before Act**: Use read/find tools to inspect files before modifying them.\n"
            "3. **Search for Unknowns**: Use `web_search` for unknown libraries or APIs.\n\n"
        )

        if strategy == "native_all":
            mode_prompt = (
                "## Execution Mode: NATIVE_ALL\n"
                "You must use your provided Native JSON Tool Calling capabilities for ALL actions.\n"
                "- To read or search, use the appropriate observation tools.\n"
                "- To modify files, you MUST use the `search_and_replace` or `write_file` JSON tools.\n"
                "- For general chat or explanations, just reply directly in Markdown.\n\n"
            )
        
        elif strategy == "hybrid":
            mode_prompt = (
                "## Execution Mode: HYBRID (JSON + Markdown)\n"
                "You have access to native JSON tools for observation, but MUST use Markdown for writing code.\n"
                "### Modifying Files (Use Markdown ONLY)\n"
                "Do NOT use JSON tools for writing files. Use one of these formats in your text:\n\n"
                "**Format A: Unified Diff** (For modifying existing files)\n"
                "```diff\n--- a/filename.py\n+++ b/filename.py\n@@ -1,2 +1,3 @@\n def main():\n-  pass\n+  print('hi')\n```\n\n"
                "**Format B: WRITE_FILE** (For new files or massive rewrites)\n"
                "WRITE_FILE: path/to/file.ext\n<<<CONTENT\n... code ...\nCONTENT>>>\n\n"
            )

        elif strategy == "text_only":
            mode_prompt = (
                "## Execution Mode: TEXT_ONLY (XML Tags)\n"
                "You do NOT have native JSON tools. You must output specific formats in your Markdown text.\n\n"
                "### Using Tools\n"
                "To use a tool, output exactly this XML format. The system will pause and return results:\n"
                "`<tool_call><web_search><query>python</query></web_search></tool_call>`\n"
                "`<tool_call><run_bash_command><command>ls -la</command></run_bash_command></tool_call>`\n\n"
                "### Modifying Files\n"
                "**Format A: Unified Diff** (Start with `## Action` followed by ```diff block)\n"
                "**Format B: WRITE_FILE** (`WRITE_FILE: path\\n<<<CONTENT\\n...\\nCONTENT>>>`)\n\n"
            )

        # Append Verification Rules
        verify_prompt = (
            "## Verification & Environment Rules\n"
            "1. After modifying code, you MUST run a verification command (e.g., `python3 script.py`).\n"
            "2. Use the `finish_task` tool ONLY when you have verified the solution works.\n"
        )
        
        return base_prompt + mode_prompt + verify_prompt

    @staticmethod
    def format_task(goal: str, allowlist: List[str], context_files: List[str], workspace_dir: str, content_injector) -> str:
        """... (保持原有的 format_task 结构，仅做微调) ..."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        
        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Directory: `./` (inside `{workspace_dir}/`)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n\n"
        )
        
        # 注入上下文
        files_md = content_injector(allowlist + context_files)
        if files_md:
            base_md += f"## Provided File Context\n{files_md}"

        return base_md
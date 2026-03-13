from typing import List
from pathlib import Path
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any
import datetime
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
    def get_system_prompt_v1(cls, strategy: str, tools_list: list) -> str:
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

    @classmethod
    def get_system_prompt(cls, strategy: str, base_tools_list: List[Dict[str, Any]], domain: str = "general", location: str = "California, United States") -> str:
        """
        Dynamically generates the system prompt based on the chosen Tool Strategy.
        Ensures the agent is completely clear on HOW to call tools.
        """
        #base_prompt = "You are an elite, general-purpose AI Agent. You operate in a structured environment.\n\n"
        # [NEW] (Spatio-temporal Awareness)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        
        base_prompt = (
            f"You are an elite, general-purpose AI Agent.\n"
            f"**Current System Time**: {current_time}\n"
            f"**Assumed Location**: {location}\n\n"
            f"⚠️ **ENVIRONMENT RULES (CRITICAL)**:\n"
            f"- You are already inside the target workspace directory.\n"
            f"- ALL file paths you read, write, or execute must be RELATIVE to your current directory (e.g., `./script.py`).\n"
            f"- NEVER use `cd` in bash commands. Execute directly (e.g., `python3 script.py`, not `cd dir && python3 script.py`).\n\n"
            # 👇 新增规则：剥夺交互权限，强制自主推进
            f"- NO HUMAN INTERACTION: You are running in an autonomous loop. Do NOT ask the user for clarification, permission, or follow-up questions.\n"
            f"- Make reasonable assumptions and proceed autonomously using tools until the goal is fully achieved.\n\n"
        )
        
        if domain != "general":
            base_prompt += f"**Domain Profile**: You are operating as a specialist in the '{domain.upper()}' domain. Use your domain-specific tools effectively.\n\n"

        # -------------------------------------------------------------
        # 1. 组装特定模式的指令
        # -------------------------------------------------------------
        mode_prompt = ""
        
        if strategy == "native_all":
            # NATIVE_ALL 模式：绝对不能提任何关于 XML 的内容！
            mode_prompt = (
                "## 1. Execution Mode: NATIVE_ALL\n"
                "You are operating in an environment that fully supports native JSON Function/Tool Calling.\n"
                "- You MUST use the provided JSON functions (tools) for ALL interactions, including reading files, searching, and WRITING files.\n"
                "- Do NOT use markdown code blocks to write code. You MUST use the `write_file` or `search_and_replace` tools.\n"
                "- For general chat or thinking, just reply in plain text.\n\n"
            )
            # 在这种模式下，不需要手动列出工具，底层 API 会通过系统层传递 Schema 给模型
            
        elif strategy == "hybrid":
            # HYBRID: native JSON for observation tools, XML child-tags for file mutations
            mode_prompt = (
                "## 1. Execution Mode: HYBRID (Native JSON + XML File Mutations)\n"
                "You have access to native JSON tools (provided via the API) for gathering information.\n"
                "For writing or editing files, you MUST use the XML formats below instead of JSON tools.\n\n"
                "### Writing / Editing Files (XML format — DO NOT use JSON tools for these)\n"
                "**Create or overwrite a file:**\n"
                "`<tool_call><write_file><path>./path/to/file.py</path><content>full file content here</content></write_file></tool_call>`\n\n"
                "**Replace text in an existing file:**\n"
                "`<tool_call><search_and_replace><path>./file.py</path><old_text>exact old text</old_text><new_text>new text</new_text></search_and_replace></tool_call>`\n\n"
            )

        elif strategy == "text_only":
            # TEXT_ONLY: no native JSON tools at all — everything via XML child-tags.
            # The xml_docs are auto-generated from the base_tools_list schema,
            # using the model's trained convention: <tool_call><name><param>val</param></name></tool_call>
            def _tool_to_xml_example(t: dict) -> str:
                name = t["name"]
                props = t.get("properties", {})
                inner = "".join(f"<{k}>...</{k}>" for k in props)
                return f"`<tool_call><{name}>{inner}</{name}></tool_call>`"

            # Split tools into categories for a cleaner prompt
            obs_tools   = [t for t in base_tools_list if t["name"] not in ("write_file", "search_and_replace", "finish_task")]
            mut_tools   = [t for t in base_tools_list if t["name"] in ("write_file", "search_and_replace")]
            done_tools  = [t for t in base_tools_list if t["name"] == "finish_task"]

            obs_section  = "\n".join(f"  {_tool_to_xml_example(t)}\n    ↳ {t['description']}" for t in obs_tools)
            mut_section  = "\n".join(f"  {_tool_to_xml_example(t)}\n    ↳ {t['description']}" for t in mut_tools)
            done_section = "\n".join(f"  {_tool_to_xml_example(t)}\n    ↳ {t['description']}" for t in done_tools)

            mode_prompt = (
                "## 1. Execution Mode: TEXT_ONLY\n"
                "You do NOT have native JSON tools. You must output specific XML formats in your text to interact.\n\n"
                "### Using Interactive Tools\n"
                "To gather info, execute bash, or finish a task, output EXACTLY this XML format:\n\n"
                f"{obs_section}\n\n"
                "### Writing / Editing Files\n"
                f"{mut_section}\n\n"
                "### Completing the Task\n"
                f"{done_section}\n\n"
                "⚠️ CRITICAL FORMATTING RULES:\n"
                "- ❌ WRONG: `web_search(query='...')` (Python syntax — rejected)\n"
                "- ❌ WRONG: `<web_search query=\"...\">` (XML attributes — rejected)\n"
                "- ❌ WRONG: `WRITE_FILE: path\\n<<<CONTENT` (old format — rejected)\n"
                "- ✅ CORRECT: `<tool_call><web_search><query>...</query></web_search></tool_call>`\n"
                "- ✅ CORRECT: `<tool_call><write_file><path>./foo.py</path><content>...</content></write_file></tool_call>`\n\n"
                # 👇 新增规则：防幻觉
                "- NEVER invent tool names or parameter tags. Use EXACTLY the tag names shown above.\n\n"
            )

        # -------------------------------------------------------------
        # 2. 追加通用规则
        # -------------------------------------------------------------
        # general_rules = (
        #     "## 2. Information Gathering & Verification Rules (CRITICAL)\n"
        #     "1. **Never Guess Code**: Check `Provided File Context` first. If missing, use tools to read files.\n"
        #     "2. **Search for Unknowns**: Use `web_search` for unknown libraries or APIs.\n"
        #     "3. **Verification**: After modifying code, you MUST run a verification command (e.g., `python3 script.py`).\n"
        #     #"4. Use the `finish_task` tool ONLY when you have verified the solution works.\n"
        #     # 👇 修改规则：强制调用 finish_task，防止啰嗦
        #     "4. **End of Task**: The moment you have fully achieved the user's Goal, you MUST immediately call the `finish_task` tool in your very next response. Do NOT provide a plain text summary without calling this tool.\n"
        # )
        general_rules = (
            "## 2. Information Gathering & Verification Rules (CRITICAL)\n"
            "1. **Never Guess Code**: Check `Provided File Context` first. If missing, use tools to read files.\n"
            "2. **Search for Unknowns**: Use `web_search` for unknown libraries or APIs.\n"
            # 👇 核心优化：区分代码验证与文档验证，大幅提升编写文档时的效率
            "3. **Context-Aware Verification**:\n"
            "   - **For Executable Code**: After modifying scripts, you MUST run a verification command (e.g., `python3 script.py` or `pytest`) to ensure it works.\n"
            "   - **For Documents/Reports**: (e.g., `.md`, `.txt`) Do NOT attempt to execute them. Once the writing tool returns a success message, consider it verified.\n"
            "4. **End of Task**: The moment you have fully achieved the user's Goal, you MUST immediately call the `finish_task` tool in your very next response. Do NOT provide a plain text summary without calling this tool.\n"
        )
        cot_rules = (
            "## Chain-of-Thought Protocol\n"
            "- If you need to think before acting, enclose your thoughts within `<think>...</think>` tags.\n"
            "- CRITICAL: Your actual tool call (JSON or XML) MUST be placed OUTSIDE and AFTER the `<think>...</think>` tags, otherwise the system cannot execute it.\n\n"
        )
        return base_prompt + cot_rules + mode_prompt + general_rules

    @staticmethod
    def format_task(goal: str, allowlist: List[str], context_files: List[str], workspace_dir: str, content_injector) -> str:
        """... (保持原有的 format_task 结构，仅做微调) ..."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        
        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Absolute Directory Path: `{workspace_dir}`\n"
            f"Working Directory: `./` (You are ALREADY here. Do not try to cd into it.)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n\n"
        )
        
        # 注入上下文
        files_md = content_injector(allowlist + context_files)
        if files_md:
            base_md += f"## Provided File Context\n{files_md}"

        return base_md
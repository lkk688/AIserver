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
    
class PromptRegistry:

    @classmethod
    def get_system_prompt(
        cls,
        strategy: str,
        base_tools_list: List[Dict[str, Any]],
        domain: str = "general",
        location: str = "California, United States",
        current_time: str = "",
    ) -> str:
        """
        Dynamically generates the system prompt based on the chosen Tool Strategy.
        Ensures the agent is completely clear on HOW to call tools.
        """
        #base_prompt = "You are an elite, general-purpose AI Agent. You operate in a structured environment.\n\n"
        # [NEW] (Spatio-temporal Awareness)
        resolved_current_time = current_time.strip() if isinstance(current_time, str) and current_time.strip() else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        
        base_prompt = (
            f"You are an elite, general-purpose AI Agent.\n"
            f"**Current System Time**: {resolved_current_time}\n"
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
            "3. **Context-Aware Verification**:\n"
            "   - **For Executable Code**: After writing or modifying scripts, you MUST run a verification command "
            "(e.g., `python3 script.py` or `pytest`) and confirm exit code 0 before calling `finish_task`.\n"
            "   - **If verification fails (non-zero exit / error output)**: diagnose and fix the error — "
            "do NOT call `finish_task` until the code runs successfully.\n"
            "   - **For Documents/Reports** (`.md`, `.txt`): Do NOT execute them. "
            "Once write_file succeeds, consider it verified.\n"
            "4. **Output File Naming**:\n"
            "   - When the user asks to **create a new** document or report, always use a **unique, descriptive filename** "
            "(e.g., `qwen3_architecture_report.md`, `analysis_v2.md`). "
            "NEVER silently reuse a previous output filename for unrelated content.\n"
            "   - When the user asks to **update or revise** an existing document, you MAY overwrite the same file.\n"
            "   - If you are unsure, err on the side of a new filename.\n"
            "5. **Multi-Step Mutations — Complete All in One Turn**:\n"
            "   - If you need to insert or replace **multiple items** (e.g., 4 images, 3 sections) into a document, "
            "issue ALL `search_and_replace` or `write_file` calls in the **same response** — do NOT stop after the first one.\n"
            "   - Plan first (inside `<think>`), then emit ALL tool calls back-to-back outside `<think>` in one response.\n"
            "   - NEVER emit only a `<think>` block with no tool call after it — your thinking has no effect unless a tool call follows.\n"
            "6. **End of Task**: The moment you have fully achieved the user's Goal, you MUST immediately call the "
            "`finish_task` tool. Do NOT provide a plain text summary without calling this tool.\n"
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

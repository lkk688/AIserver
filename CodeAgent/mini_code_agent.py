#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mini_claude_code.py (v2 – Robust)
A minimal, non-interactive "Claude Code"-like coding agent with:
- Multi-diff extraction & sanitization
- Write-file fallback when diffs fail
- Robust continuation stitching
- Fault-tolerant JSON planner
- SkillDB injection
- Prompt ledger & session logging

Requirements:
  pip install openai rich tiktoken

python CodeAgent/qwen_coder_evalv4_1.py   --model_source remote_vllm   --remote_vllm_url "https://w0wqtv67-8000.usw3.devtunnels.ms/v1"   --models "Qwen/Qwen3-Coder-Next-FP8"   --run_all --out_dir ./eval_results_remote

Env (overridden by CLI args):
  VLLM_BASE_URL (default https://w0wqtv67-8000.usw3.devtunnels.ms/v1)
  VLLM_API_KEY  (default myhpcvllmqwen)
  VLLM_MODEL    (default Qwen/Qwen3-Coder-Next-FP8)
"""

import os
import re
import json
import time
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from openai import OpenAI
try:
    import tiktoken
except ImportError:
    tiktoken = None

console = Console()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CodeAgent.codeagent_libs import (
    now_stamp,
    is_git_repo,
    estimate_tokens,
    query_model_context_length,
    read_file,
    ensure_dirs,
    top_level_tree,
    _handle_missing_modules,
    run_shell,
    _determine_verify_cmd,
    format_skill_injection,
    save_skill,
    select_relevant_skills,
    extract_skill_insight,
    detect_tech_stack,
    truncate_to_tokens,
    compute_safe_max_tokens,
    extract_json_robust,
    apply_patch_guarded,
    apply_write_files,
    extract_all_diffs,
    extract_write_file_actions,
    sanitize_diff_text,
    apply_fuzzy_patch,
    extract_files_from_diff,
    resolve_path,
)

# ---------------------------
# Config Defaults
# ---------------------------

AGENT_DIR = Path(".agent")
SESSIONS_DIR = AGENT_DIR / "sessions"
SKILL_DIR = AGENT_DIR / "skilldb"
SKILL_SUCCESS = SKILL_DIR / "successes.jsonl"
SKILL_FAIL = SKILL_DIR / "failures.jsonl"
RUNS_LOG = AGENT_DIR / "runs.jsonl"
SKILL_TEACHER = SKILL_DIR / "teacher.jsonl"

# DEFAULT_SYSTEM is now centralized in PromptRegistry.SYSTEM (see below)

# Skill injection limits (keep short to save tokens)
SKILL_INJECT_TOPK = 6
SKILL_INJECT_MAX_LINES = 40  # total lines injected into prompt

@dataclass
class AgentConfig:
    client: OpenAI
    model: str
    session_dir: Path
    max_context: int
    max_output: int
    auto_approve: bool
    agent_dir: Path
    model_max_context: int = 0  # 0 = auto-detected from model, fallback to max_context


# ---------------------------
# Prompt Logic (centralized in PromptRegistry below)
# ---------------------------
# All prompt construction functions have been merged into the PromptRegistry class.
# See PromptRegistry.format_task(), format_bugfix(), format_fix_diff(), format_fix_rewrite().


# ---------------------------
# Core Loop
# ---------------------------

def _try_apply_content(content: str, allowlist: List[str], turn_dir: Path, 
                       config: AgentConfig) -> bool:
    """
    Try all methods to apply model output as file changes.
    Order: 
    1. git apply (Strict Diff)
    2. apply_fuzzy_patch (Loose Diff - handles line/whitespace errors)
    3. WRITE_FILE (Full rewrite) — tried even if diff was found
    4. Diff Extraction (Last resort reconstruction for new files)
    """
    
    # --- Extract Diff once ---
    diff = extract_all_diffs(content)
    changes_applied = False
    apply_method = None
    
    # --- TRY FORMAT A: Unified Diff Strategies ---
    if diff:
        (turn_dir / "patch.diff").write_text(diff, encoding="utf-8")
        
        # Strategy 1: Strict Git Apply
        if is_git_repo():
            changes_applied = apply_patch_guarded(diff, turn_dir, auto_approve=config.auto_approve)
            if changes_applied:
                apply_method = "git_apply"
        else:
            console.print("[red]Not a git repo, skipping strict diff apply.[/red]")
        
        # Strategy 2: Fuzzy Patch
        if not changes_applied:
            console.print("[yellow]Strict apply failed. Attempting fuzzy patch...[/yellow]")
            file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
            fuzzy_successes = 0
            fuzzy_total = 0
            
            fuzzy_logs = ["\n--- Fuzzy Patch Attempt ---"]
            
            for fd in file_diffs:
                if not fd.strip().startswith("diff --git"): continue
                fuzzy_total += 1
                
                # Extract raw path from header
                match = re.search(r'diff --git a/\S+ b/(\S+)', fd)
                if match:
                    raw_path = match.group(1)
                    fuzzy_logs.append(f"Processing diff for: {raw_path}")
                    
                    # Resolve Path
                    target_path = resolve_path(raw_path, allowlist)
                    
                    if target_path:
                        if target_path != Path(raw_path):
                            msg = f"[dim]Redirecting '{raw_path}' -> '{target_path}'[/dim]"
                            console.print(msg)
                            fuzzy_logs.append(msg)
                        
                        if apply_fuzzy_patch(target_path, fd, log_buffer=fuzzy_logs):
                            fuzzy_successes += 1
                            fuzzy_logs.append(">> Success")
                        else:
                            fuzzy_logs.append(">> Failed")
                    else:
                        msg = f"[red]Skipping diff for unresolved path: {raw_path}[/red]"
                        console.print(msg)
                        fuzzy_logs.append(msg)
            
            # Append logs to apply.log
            try:
                with open(turn_dir / "apply.log", "a", encoding="utf-8") as f:
                    f.write("\n".join(fuzzy_logs) + "\n")
            except Exception as e:
                console.print(f"Failed to append to apply.log: {e}")

            # Mark success if at least one file was patched
            if fuzzy_successes > 0:
                changes_applied = True
                apply_method = "fuzzy_patch"
                console.print(f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]")


    # --- TRY FORMAT B: WRITE_FILE ---
    # Try WRITE_FILE regardless of whether a diff was found — some responses
    # contain both a diff AND a WRITE_FILE block. If the diff failed, the
    # WRITE_FILE may still work.
    if not changes_applied:
        write_actions = extract_write_file_actions(content)
        if write_actions:
            valid_actions = []
            for path, text in write_actions:
                # Resolve Path
                target_path = resolve_path(path, allowlist)
                if target_path:
                    valid_actions.append((str(target_path), text))
                else:
                    console.print(f"[red]Skipping WRITE_FILE for unresolved path: {path}[/red]")
            
            if valid_actions:
                changes_applied = apply_write_files(valid_actions, allowlist, turn_dir)
                if changes_applied:
                    apply_method = "write_file"
    
    # --- TRY FORMAT C: Extract NEW files from diff (Last resort) ---
    # SAFETY: extract_files_from_diff ONLY extracts new files (--- /dev/null).
    # For edit diffs, it safely skips to avoid overwriting existing files
    # with tiny fragments (the session 2026-02-16_215657 bug).
    if not changes_applied and diff:
        console.print("[yellow]All patch methods failed. Checking for extractable new files in diff...[/yellow]")
        diff_files = extract_files_from_diff(diff)
        if diff_files:
            changes_applied = apply_write_files(diff_files, allowlist, turn_dir)
            if changes_applied:
                apply_method = "diff_extraction"
                console.print("[green]Wrote new files extracted from diff.[/green]")
        else:
            console.print("[red]No new files to extract. Edit diffs cannot be safely applied as rewrites.[/red]")
    
    # --- Log result ---
    if apply_method:
        console.print(f"[green]Changes applied via: {apply_method}[/green]")
    elif not changes_applied:
        # Check if we missed a WRITE_FILE due to bad formatting
        if "WRITE_FILE:" in content and "CONTENT" in content:
             console.print("[red]Potential malformed WRITE_FILE block detected but extraction failed.[/red]")
        
        if not diff and not extract_write_file_actions(content):
            console.print("[red]No valid diff or WRITE_FILE actions found in response.[/red]")
            
            # --- TRY FORMAT E: Fenced Block Fallback (Session 213156 fix) ---
            # If model wraps code in markdown fences but forgets WRITE_FILE
            if len(allowlist) == 1 and not changes_applied:
                # Look for ```python ... ``` or just ``` ... ```
                # We want EXACTLY ONE code block to be safe
                code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
                
                if len(code_blocks) == 1:
                    target_file = Path(allowlist[0])
                    block_content = code_blocks[0].strip()
                    
                    # Heuristic: does it look like Python code?
                    if "def " in block_content or "import " in block_content:
                        console.print(f"[yellow]Fallback E: Extracting single fenced block for {target_file}[/yellow]")
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_text(block_content + "\n", encoding="utf-8")
                        apply_method = "fenced_fallback"
                        changes_applied = True
            
            # --- TRY FORMAT D: Raw Code Fallback (Session 153128 fix) ---
            # If the model outputs *just* the code without formatting, and we expect 1 file.
            if len(allowlist) == 1 and not changes_applied:
                target_file = Path(allowlist[0])
                # Heuristic: does it look like Python code?
                if "def " in content or "import " in content:
                    console.print(f"[yellow]Fallback D: Treating entire response as content for {target_file}[/yellow]")
                    
                    # Sanitize: Remove markdown fences if they wrap the whole content
                    clean_content = content.strip()
                    if clean_content.startswith("```python"):
                        clean_content = clean_content[len("```python"):].strip()
                    elif clean_content.startswith("```"):
                        clean_content = clean_content[3:].strip()
                    
                    if clean_content.endswith("```"):
                        clean_content = clean_content[:-3].strip()
                        
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(clean_content + "\n", encoding="utf-8")
                    apply_method = "raw_fallback"
                    changes_applied = True
    
    return changes_applied



# ---------------------------
# LLM Interaction
# ---------------------------
def complete_with_continuation(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    model_max_context: int = 16384,
) -> str:
    """
    Calls the LLM. If finish_reason is 'length', appends the partial response
    to messages and asks it to continue, stitching the results.
    
    Robustness Features:
    - Strips conversational filler from continuations ("Here is the rest...").
    - Prevents hallucinated headers/markdown injection inside code blocks.
    - Adaptively caps max_tokens to prevent context overflow.
    """
    full_content = ""
    current_messages = list(messages)
    
    max_loops = 5  # Max continuation loops
    
    for i in range(max_loops):
        console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
        # Adaptive max_tokens: estimate input and cap output accordingly
        input_text = "\n".join(m.get("content", "") for m in current_messages)
        input_est = estimate_tokens(input_text)
        safe_tokens = compute_safe_max_tokens(
            input_tokens=input_est,
            model_max_context=model_max_context,
            desired_max_output=max_output_tokens
        )
        
        if safe_tokens < max_output_tokens:
            console.print(f"[yellow]Adaptive max_tokens: {safe_tokens} "
                          f"(input≈{input_est}, limit={model_max_context})[/yellow]")
        
        # Retry with backoff on API errors
        resp = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=safe_tokens
                )
                break
            except Exception as e:
                err_str = str(e)
                if 'max_tokens' in err_str or 'context length' in err_str:
                    safe_tokens = max(1024, safe_tokens // 2)
                    console.print(f"[red]Context overflow. Retrying with max_tokens={safe_tokens}...[/red]")
                    time.sleep(1)
                    continue
                console.print(f"[red]LLM Call failed: {e}[/red]")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                break
        
        if resp is None:
            console.print(f"[red]All LLM retry attempts failed.[/red]")
            break
            
        
        # Custom Environment Handling: If client returns a string, use it directly
        if isinstance(resp, str):
            full_content += resp
            break

        choice = resp.choices[0]
        console.print(f"[dim]Finish Reason: {choice.finish_reason}[/dim]")
        content = choice.message.content or ""
        
        # --- Robust Stitching Logic ---
        # If this is a continuation (loop > 0), filter out conversational prefixes.
        if i > 0:
            original_len = len(content)
            
            # Check if we were inside a code block in the previous chunk
            # (Odd number of triple-backticks implies we are inside a block)
            prev_chunk_fences = full_content.count("```")
            is_inside_code = (prev_chunk_fences % 2 == 1)
            
            # Check if we are inside a WRITE_FILE block (<<<CONTENT without CONTENT>>>)
            open_tags = len(re.findall(r'<<<CONTENT', full_content))
            close_tags = len(re.findall(r'CONTENT>{2,3}', full_content))
            is_inside_write_file = (open_tags > close_tags)
            
            if is_inside_code or is_inside_write_file:
                # 1. Strip re-opened code fences (e.g. "```python")
                # Models often restart the block when continued
                content = re.sub(r'^\s*```\w*\n', '', content)
                
                # 2. Strip "Here is the rest..." prose if it precedes code
                # If the content starts with prose lines that end in a colon or look like chat
                # (Heuristic: remove lines until we hit what looks like code)
                # Be careful not to remove actual code comments.
                if not content.strip().startswith(('#', 'def ', 'class ', 'print', 'import ')):
                     # Remove first line if it looks like conversation
                     content = re.sub(r'^(Here is the rest.*?|Sure.*?|Continuing.*?)\n', '', content, flags=re.IGNORECASE)

            # 3. Strip hallucinated headers immediately (e.g. "## Reasoning")
            # If we are inside code, a markdown header is almost always a hallucination
            if is_inside_code and content.lstrip().startswith("## "):
                # Stop processing here? Or strip the header? 
                # Usually implies model switched context. We treat it as end of code.
                console.print("[red]Detected hallucinated header in code block. Truncating.[/red]")
                content = content.split("## ")[0]

            if len(content) < original_len:
                console.print(f"[dim]Stitched continuation (stripped {original_len - len(content)} chars)[/dim]")

        full_content += content
        
        if choice.finish_reason == "length":
            console.print("[yellow]Output truncated (limit reached). Continuing...[/yellow]")
            
            # Append partial content to history
            current_messages.append({"role": "assistant", "content": content})
            
            # Strict Continuation Prompt
            cont_prompt = (
                "You were cut off. "
                "IMMEDIATELY continue the code/text exactly where you left off. "
                "DO NOT repeat the last line. "
                "DO NOT output conversational text (e.g. 'Here is the rest'). "
                "DO NOT output markdown headers or code fences. "
                "Just output the missing characters."
            )
            current_messages.append({"role": "user", "content": cont_prompt})
        else:
            break
            
    return full_content



# ---------------------------
# Task Planning
# ---------------------------

def plan_tasks(config: AgentConfig, goal: str, notes: str, allowlist: List[str]) -> List[str]:
    """
    Analyze complexity. 
    Optimized: Skips LLM call if task is constrained to 1 file or allowlist is empty (assuming new file).
    """
    
    # --- Optimization 1: Explicit Single File Constraint ---
    # If the user provided --allowlist task.py, we know we can't edit anything else.
    # Plan = [goal]. No LLM needed.
    if allowlist and len(allowlist) == 1:
        console.print(f"[green]Single file target ({allowlist[0]}) detected. Skipping planner.[/green]")
        return [goal]

    # --- Optimization 2: Implicit Single File Goal ---
    # If allowlist is empty (meaning "create whatever you need"), but the goal 
    # explicitly mentions creating a specific file, assume single task.
    # Regex looks for "Create task.py", "Write script.py", etc.
    if not allowlist:
        # Check for explicit file creation intent in goal
        # m = re.search(r"(?:create|write|implement)\s+(\S+\.py)", goal, re.IGNORECASE)
        
        # NEW: allows words in between (e.g. "Write a new test.py")
        # We use dotall to match across newlines and \b to ensure clean filename start
        m = re.search(r"(?:create|write|implement).*?\b([a-zA-Z0-9_]+\.py)", goal, re.IGNORECASE | re.DOTALL)
        if m:
            filename = m.group(1)
            console.print(f"[green]Goal targets single file ({filename}). Skipping planner.[/green]")
            # Side effect: We can hint to the main loop to verify this file later
            return [goal]

    system_prompt = """You are a technical lead. Plan the execution steps.

**CRITICAL GUIDELINES**:
1. **Prefer Single Step**: Modern LLMs can write 500+ lines at once. Do NOT split a task just because it has multiple functions.
2. **One File = One Step**: Never split the creation of a single file into multiple steps.
3. **Split Only for Isolation**: Only split if the task touches completely different parts of the system (e.g., "Step 1: Update SQL Schema", "Step 2: Update React Frontend").

Output JSON: {"steps": ["step1", ...]}
"""
    
    files_context = f"Target Files: {', '.join(str(p) for p in allowlist)}" if allowlist else "Target Files: (Open)"
    user_prompt = f"Goal: {goal}\nNotes: {notes}\n{files_context}\n\nJSON:"
    
    console.print("[cyan]Analyzing task complexity...[/cyan]")
    try:
        # Calculate adaptive tokens
        planner_input = system_prompt + user_prompt
        planner_input_est = estimate_tokens(planner_input)
        ctx_limit = config.model_max_context or config.max_context
        planner_max_tokens = compute_safe_max_tokens(
            input_tokens=planner_input_est,
            model_max_context=ctx_limit,
            desired_max_output=1024,
            min_output=256
        )

        resp = config.client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=planner_max_tokens
        )
        content = resp.choices[0].message.content or "{}"
        
        # Log planning
        (config.session_dir / "planning_response.md").write_text(content, encoding="utf-8")
        
        data = extract_json_robust(content)
        if not data or "steps" not in data:
            return [goal]
        
        steps = data["steps"]
        
        # --- Heuristic 3: Collapse micro-plans ---
        # If the model outputs many small steps for a small file list, collapse them.
        if len(steps) > 3 and (allowlist and len(allowlist) <= 2):
            console.print("[yellow]Plan too fragmented for small file count. Collapsing to single task.[/yellow]")
            return [goal]

        if len(steps) > 1:
            console.print(Panel(
                "\n".join([f"{i+1}. {s}" for i,s in enumerate(steps)]), 
                title="Task Plan", style="magenta"
            ))
            if config.auto_approve:
                # Still check: if steps look like "Step 1: Imports", collapse them
                return steps
            
            if Confirm.ask("Execute as separate sub-tasks? (No = run as one big task)"):
                return steps
            
        return [goal]

    except Exception as e:
        console.print(f"[red]Planning failed ({e}). Defaulting to single task.[/red]")
        return [goal]


class PromptRegistry:
    """
    Centralized manager for all LLM prompts.
    Optimized to reduce token waste by removing redundant git context.
    """

    SYSTEM = (
        "You are an advanced AI coding agent. Your ONLY job is to produce file changes.\n"
        "\n"
        "## Output Format (STRICT)\n"
        "You MUST output in ONE of these two formats per response. Never mix them.\n"
        "\n"
        "### Format A: Unified Diff (For small edits)\n"
        "1. Start with a brief `## Reasoning` section.\n"
        "2. Then output `## Action` followed by a SINGLE fenced diff code block.\n"
        "3. Each file diff starts with `diff --git a/<path> b/<path>`.\n"
        "4. For NEW files use `--- /dev/null` and `+++ b/<path>`.\n"
        "5. Make sure hunk line-counts are correct (@@ -X,Y +A,B @@).\n"
        "6. Do NOT put prose between diffs inside the block.\n"
        "\n"
        "### Format B: WRITE_FILE (For new files or full rewrites)\n"
        "Use when creating new files or when diffs are too complex.\n"
        "\n"
        "WRITE_FILE: path/to/file.py\n"
        "<<<CONTENT\n"
        "... file content here ...\n"
        "CONTENT>>>\n"
        "\n"
        "## Rules\n"
        "- NEVER embed triple-backtick fences inside a diff block.\n"
        "- NEVER mix Format A and Format B in the same response.\n"
        "- If output will be very long, prefer Format B (WRITE_FILE) to avoid truncation.\n"
        "- Always include `Verification: <command>` on its own line if you know how to verify.\n"
        "\n"
        "## Teacher Guidelines (CRITICAL)\n"
        "If provided, you MUST follow the language-specific guidelines in the User Prompt.\n"
    )

    @staticmethod
    def format_task(
        goal: str,
        allowlist: List[str],
        context_files: List[str],
        notes: str,
        skills: str,
        max_context: int,
        max_output: int = 4096,
    ) -> str:
        """
        Builds the main Turn Prompt.
        Optimized: Removed 'Repo Snapshot' (git status/diff) to save tokens.
        
        Prioritizes context usage:
          1. Essential Instructions & Goal (Base)
          2. File Contents (Critical Context)
          3. Directory Tree (Navigation Context - if space permits)
        """
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"

        # Detect if ALL files are new
        all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False

        # Suggest WRITE_FILE for new or multi-file tasks
        format_hint = ""
        if (allowlist and len(allowlist) > 1) or all_new_files:
            format_hint = (
                "\n> **IMPORTANT**: Use **Format B (WRITE_FILE)** to create all files. "
                "This avoids diff truncation issues and is more reliable for new files.\n"
            )
        
        # Get current relative context
        cwd = Path.cwd().name
        
        # Explicit Workspace Instruction
        workspace_block = (
            f"## Workspace Context\n"
            f"You are working in the directory: `./` (inside `{cwd}/`)\n"
            f"Use ONLY relative paths (e.g. `task.py` or `src/utils.py`).\n"
            f"DO NOT use absolute paths (e.g. `/home/user/...`).\n"
        )

        base_md = (
            f"# Turn Prompt\n\n"
            f"## Goal\n{goal}\n\n"
            f"{workspace_block}\n"  # <--- Added here
            f"## Target Files (Allowlist)\n{allow_txt}\n"
            f"{format_hint}\n"
            f"{skills if skills else ''}\n"
            f"## Constraints / Teacher Guidelines\n"
            f"{notes.strip() if notes.strip() else '(none)'}\n\n"
            f"## Output Contract\n"
            f"1. Return changes using EITHER Format A (Diff) OR Format B (WRITE_FILE).\n"
            f"2. ALL files in the Target Files list must be addressed.\n"
            f"3. (Optional) Include: \"Verification: <command>\" before the changes.\n"
        )

        # --- Token Budgeting ---
        safety_margin = 1000
        usable_context = max_context - max_output - safety_margin
        used_tokens = estimate_tokens(base_md) + estimate_tokens(PromptRegistry.SYSTEM)
        remaining = usable_context - used_tokens

        if remaining < 500:
            console.print("[red]Critical Warning: Goal + Constraints exceed context limit![/red]")
            base_md += "\n> **CRITICAL**: Input too long. Context truncated.\n"
            return base_md

        if remaining < 2000:
            base_md += "\n> **OUTPUT HINT**: Context budget is tight. Use WRITE_FILE format and keep code concise.\n"

        context_sections = []

        # --- Priority 1: File Contents (The most important context) ---
        # Ensure allowlist files come first
        priority_files = list(dict.fromkeys(list(allowlist) + list(context_files)))
        files_md = ""
        
        for f in priority_files:
            content = read_file(str(f))
            if not content or content.startswith("[MISSING FILE]"):
                continue
            
            # Smart truncation: prioritize seeing start/end of large files if needed
            # But for now, simple truncation
            if estimate_tokens(content) > 8000:
                content = truncate_to_tokens(content, 8000)
                
            file_block = f"## File: {f}\n```python\n{content}\n```\n"
            block_cost = estimate_tokens(file_block)
            
            if block_cost < remaining:
                files_md += file_block
                remaining -= block_cost
            else:
                files_md += f"## File: {f}\n[Content Omitted - Context Limit Reached]\n"
        
        if files_md:
            context_sections.append(files_md)

        # --- Priority 2: Directory Tree (Navigation context) ---
        # Only include if we have a healthy buffer (e.g. >500 tokens)
        if not all_new_files and remaining > 500:
            tree = top_level_tree()
            if estimate_tokens(tree) < remaining:
                context_sections.append(f"### File Tree\n{tree}\n")

        if context_sections:
            base_md += "\n## Context\n" + "\n".join(context_sections)

        return base_md

    @staticmethod
    def format_bugfix(file_path: str, error_output: str, original_goal: str = "") -> str:
        """
        Focused bug-fix prompt. Forces WRITE_FILE output.
        """
        content = read_file(str(file_path))
        if not content:
            content = "[FILE NOT FOUND]"

        return (
            f"# Bug Fix Required\n\n"
            f"## Original Goal\n{original_goal if original_goal else '(see previous context)'}\n\n"
            f"## Current File: {file_path}\n```python\n{content}\n```\n\n"
            f"## Error Output\n```\n{error_output[-3000:]}\n```\n\n"
            f"## STRICT Instructions\n"
            f"1. Analyze the Traceback to find the failing function.\n"
            f"2. Fix the specific error shown.\n"
            f"3. **CRITICAL: Scan the rest of that function for similar issues.**\n"
            f"   (e.g., if you change a variable from Tensor to Numpy, ensure ALL subsequent usages handle Numpy).\n"
            f"4. Output the COMPLETE corrected file using WRITE_FILE format.\n"
            f"5. Do NOT use diffs.\n"
            f"6. Output EXACTLY one WRITE_FILE block, nothing else after it.\n\n"
            f"WRITE_FILE: {file_path}\n"
            f"<<<CONTENT\n"
            f"... your complete corrected file here ...\n"
            f"CONTENT>>>\n"
        )

    @staticmethod
    def format_fix_diff(file_path: str, code_content: str, error_log: str, teacher_guidelines: str = "") -> str:
        """
        Prompt for Strategy 1: Quick Fix via Diff.
        """
        return (
            f"# Bug Fix Required (Diff Strategy)\n\n"
            f"The previous code for `{file_path}` failed verification.\n\n"
            f"## Error Output\n```\n{error_log[-3000:]}\n```\n\n"
            f"## Instructions\n"
            f"1. **Analyze**: Look at the error and the code below.\n"
            f"2. **Scope**: Fix ONLY the specific error.\n"
            f"3. **Consistency**: Check the *entire function* for related issues.\n"
            f"4. **Output**: Use **Format A (Unified Diff)**.\n"
            f"{teacher_guidelines}\n\n"
            f"## Current Code: {file_path}\n```python\n{code_content}\n```\n"
        )

    @staticmethod
    def format_fix_rewrite(file_path: str, current_code: str, error_history: str, teacher_guidelines: str = "") -> str:
        """
        Prompt for Strategy 2: Full Rewrite.
        Ensures the model sees the broken code so it can recover logic.
        """
        return (
            f"# Rewrite Required (Fresh Start)\n\n"
            f"Diff-based fixes have failed. We need a clean rewrite of `{file_path}`.\n\n"
            f"## Context: Current File Content (Broken)\n"
            f"```python\n{current_code}\n```\n\n"
            f"## Failure History\n```\n{error_history[-4000:]}\n```\n\n"
            f"## Instructions\n"
            f"1. **Recover**: Use the logic from the 'Current File' above, but fix the errors.\n"
            f"2. **Format**: Output the **COMPLETE** file using **Format B (WRITE_FILE)**.\n"
            f"3. **Constraint**: Do NOT use diffs. Do NOT use placeholders.\n"
            f"4. **Completeness**: You must output every single line of code.\n"
            f"{teacher_guidelines}\n\n"
            f"WRITE_FILE: {file_path}\n"
            f"<<<CONTENT\n"
            f"... complete fixed code ...\n"
            f"CONTENT>>>\n"
        )

def run_subtask_loop(
    config: AgentConfig,
    subtask: str,
    subtask_idx: int,
    allowlist: List[str],
    context_files: List[str],
    global_notes: str,
) -> bool:
    """
    Modular execution loop: Generate -> Verify -> Fix(Diff) -> Fix(Rewrite) -> Exit
    """
    skill_dir = config.agent_dir / "skilldb"
    turn_base = subtask_idx * 10
    console.rule(f"Executing Sub-task {subtask_idx+1}: {subtask}")

    def get_turn_dir(offset: int) -> Path:
        d = config.session_dir / f"{turn_base + offset:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # =========================================================================
    # PHASE 1: GENERATION
    # =========================================================================
    console.print("[bold cyan]Phase 1: Generating Code[/bold cyan]")
    turn_dir = get_turn_dir(0)
    
    # 1. Prepare Prompt
    # Enhanced Selection: Include global_notes (tech stack) in the query
    skill_query = f"{subtask}\n{global_notes}"
    inject = format_skill_injection(select_relevant_skills(skill_query, skill_dir))
    
    combined_guidelines = f"{global_notes}\n\n{inject}".strip()
    
    prompt_md = PromptRegistry.format_task(
        subtask, allowlist, context_files, global_notes, inject, 
        config.max_context, config.max_output
    )
    (turn_dir / "prompt.md").write_text(prompt_md, encoding="utf-8")

    # 2. Call Model
    console.print("[cyan]Generating solution...[/cyan]")
    content = complete_with_continuation(
        config.client, config.model,
        [{"role": "system", "content": PromptRegistry.SYSTEM}, 
         {"role": "user", "content": prompt_md}],
        max_output_tokens=config.max_output,
        model_max_context=config.model_max_context
    )
    (turn_dir / "response.md").write_text(content, encoding="utf-8")

    # 3. Detect Modified Files (Critical for Verification)
    # We parse the output to see what files are being touched
    modified_files = []
    
    # Scan for WRITE_FILE targets
    w_actions = extract_write_file_actions(content)
    for p, _ in w_actions: 
        modified_files.append(p)
    
    # Scan for Diff targets
    diff_text = extract_all_diffs(content)
    if diff_text:
        # Regex to find '+++ b/filename'
        diff_paths = re.findall(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE)
        modified_files.extend(diff_paths)
    
    # Deduplicate
    modified_files = list(set(modified_files))

    # 4. Apply Code
    if not _try_apply_content(content, allowlist, turn_dir, config):
        # Retry logic for malformed WRITE_FILE could go here
        if "WRITE_FILE:" in content and "CONTENT" in content and not w_actions:
             console.print("[yellow]Detected malformed WRITE_FILE. Retrying...[/yellow]")
             # (Optional: Insert retry logic here)
        
        console.print("[red]Failed to apply generated code. Stopping.[/red]")
        return False
    console.print("[green]Code generated and applied.[/green]")

    # =========================================================================
    # PHASE 2: VERIFICATION & FIX
    # =========================================================================
    console.print("[bold cyan]Phase 2: Verification[/bold cyan]")
    
    # Check for explicit verification command in output
    auto_verify_cmd = None
    v_match = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
    if v_match:
        auto_verify_cmd = v_match.group(1).strip()
    
    # Determine the actual command to run
    # We pass 'modified_files' so we can default to 'python3 task.py' 
    # even if allowlist is empty.
    verify_cmd = _determine_verify_cmd(allowlist, modified_files, auto_verify_cmd, config)
    
    if not verify_cmd:
        console.print("[yellow]No verification command selected. Assuming success.[/yellow]")
        return True

    # --- Verification Loop ---
    error_history = []
    
    # Increase from 3 to 4 attempts (0=Initial, 1=Diff, 2=Rewrite, 3=Final Rewrite)
    MAX_RETRIES = 4
    
    for fix_stage in range(MAX_RETRIES): 
        
        console.print(f"[blue]Running verification (Stage {fix_stage})...[/blue]")
        code, out = run_shell(verify_cmd, cap=20000)
        
        # --- Auto-Install Missing Modules ---
        if code != 0:
            install_log = _handle_missing_modules(out)
            if install_log:
                out += install_log
                # Retry verification immediately
                console.print("[blue]Retrying verification after installation...[/blue]")
                code, out_retry = run_shell(verify_cmd, cap=20000)
                out += f"\n[Post-Install Verification]\n{out_retry}\n"
        
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        if code == 0:
            console.print(f"[green]Verification PASSED at Stage {fix_stage}![/green]")
            save_skill(config, subtask, global_notes, True, out)
            return True
        
        console.print(f"[red]Verification Failed (exit={code})[/red]")
        error_history.append(f"Stage {fix_stage} Output:\n{out}\n{'-'*20}")
        
        if fix_stage == MAX_RETRIES - 1:
            console.print("[bold red]All fix attempts failed. Exiting subtask.[/bold red]")
            save_skill(config, subtask, global_notes, False, out)
            return False

        # --- PREPARE FIX ---
        turn_dir = get_turn_dir(fix_stage + 1)
        
        # Pick the most relevant file to fix (heuristic: first python file modified)
        target_file = next((f for f in modified_files if str(f).endswith('.py')), None)
        if not target_file and allowlist:
             target_file = next((f for f in allowlist if str(f).endswith('.py')), allowlist[0])
        
        if not target_file:
            console.print("[red]Cannot identify a target file to fix. Aborting.[/red]")
            return False

        current_code = read_file(str(target_file))

        if fix_stage == 0:
            # STRATEGY 1: DIFF FIX
            console.print("[yellow]Attempting Fix 1: Targeted Diff...[/yellow]")
            fix_prompt = PromptRegistry.format_fix_diff(
                target_file, current_code, out,
                teacher_guidelines=combined_guidelines
            )
        else:
            # STRATEGY 2: FULL REWRITE
            console.print("[yellow]Attempting Fix 2: Full Rewrite (Accumulated Errors)...[/yellow]")
            full_history = "\n".join(error_history)
            # UPDATE: Pass current_code here
            fix_prompt = PromptRegistry.format_fix_rewrite(
                target_file, current_code, full_history,
                teacher_guidelines=combined_guidelines
            )
            #fix_prompt = PromptRegistry.format_fix_rewrite(target_file, full_history)

        (turn_dir / "prompt.md").write_text(fix_prompt, encoding="utf-8")

        # Generate Fix
        console.print("[cyan]Generating fix...[/cyan]")
        fix_content = complete_with_continuation(
            config.client, config.model,
            [{"role": "system", "content": PromptRegistry.SYSTEM}, 
             {"role": "user", "content": fix_prompt}],
            max_output_tokens=config.max_output,
            model_max_context=config.model_max_context
        )
        (turn_dir / "response.md").write_text(fix_content, encoding="utf-8")

        # Apply Fix
        if not _try_apply_content(fix_content, allowlist, turn_dir, config):
            console.print("[red]Failed to apply fix. Moving to next strategy...[/red]")
            # Loop continues to next stage (Rewrite) automatically
    
    return False



# ---------------------------
# Main Orchestrator
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", help="Task goal/description")
    parser.add_argument("--allowlist", help="Comma-separated list of files to allow editing")
    parser.add_argument("--context", help="Comma-separated list of read-only context files")
    parser.add_argument("--notes", help="Extra notes/constraints", default="")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-approve patches and verification")
    
    # Configurable Model/Env
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "https://w0wqtv67-8000.usw3.devtunnels.ms/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "myhpcvllmqwen"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8"))
    
    # Configurable Agent config
    parser.add_argument("--agent-dir", default=".agent", help="Directory for agent artifacts")
    parser.add_argument("--max-context", type=int, default=16000, help="Max context length")
    parser.add_argument("--max-output", type=int, default=4096, help="Max output tokens")
    
    parser.add_argument("--migrate-skills", action="store_true", help="Migrate legacy skill DB to new format")
    parser.add_argument("--artifacts-dir", help="Directory where the agent should save task artifacts (plots, models)")
    
    args = parser.parse_args()

    agent_dir = Path(args.agent_dir)
    ensure_dirs(agent_dir)

    # Initialize Client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    # Migration Mode
    if args.migrate_skills:
        skill_dir = agent_dir / "skilldb"
        console.print("[bold yellow]Starting Skill DB Migration...[/bold yellow]")
        
        # Load legacy skills
        legacy_skills = []
        for kind, filename in [("success", "successes.jsonl"), ("failure", "failures.jsonl")]:
            path = skill_dir / filename
            if path.exists():
                for line in path.read_text(errors="ignore").splitlines():
                    if line.strip(): legacy_skills.append((kind == "success", json.loads(line)))
        
        console.print(f"Found {len(legacy_skills)} legacy records.")
        
        # Process each
        new_db = skill_dir / "skills.jsonl"
        for i, (success, obj) in enumerate(legacy_skills):
            console.print(f"[{i+1}/{len(legacy_skills)}] Extracting insight...")
            goal = obj.get("text", "").split("\n")[0].replace("Goal: ", "")
            evidence = obj.get("evidence", "")
            
            # Use the extraction logic
            skill = extract_skill_insight(client, args.model, goal, success, evidence)
            
            # Save (append to new DB)
            with open(new_db, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(skill)) + "\n")
                
        console.print("[green]Migration Complete![/green]")
        return

    # 1. Auto-detect model context
    detected_ctx = query_model_context_length(client, args.model)
    effective_ctx = detected_ctx if detected_ctx > 0 else args.max_context
    console.print(f"[dim]Effective context limit: {effective_ctx} tokens[/dim]")

    # 2. Setup Session
    session_id = now_stamp()
    session_dir = agent_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    config = AgentConfig(
        client=client,
        model=args.model,
        session_dir=session_dir,
        max_context=args.max_context,
        max_output=args.max_output,
        auto_approve=args.yes,
        agent_dir=agent_dir,
        model_max_context=effective_ctx,
    )

    console.print(Panel(
        f"Session: {session_id}\nbase_url={args.base_url}\nmodel={args.model}\nlogs: {session_dir}",
        title="mini-claude-code (Teacher-Enhanced)",
        style="cyan"
    ))

    # 3. Gather Inputs (Goal & Allowlist)
    goal = args.goal
    if not goal:
        goal = Prompt.ask("Goal").strip()

    allowlist: List[str] = []
    if args.allowlist:
        allowlist = [x.strip() for x in args.allowlist.split(",") if x.strip()]
    elif not args.yes:
        console.print("\n[bold]ALLOWLIST[/bold] (only these files may be modified)")
        while True:
            p = Prompt.ask("Add allowlisted file path (empty to stop)", default="").strip()
            if not p:
                break
            allowlist.append(p)
    
    # Default to allowlist empty -> "create whatever" (Handled in planner)

    # 4. Context Files
    context_files = list(dict.fromkeys(allowlist)) 
    if args.context:
        extra = [x.strip() for x in args.context.split(",") if x.strip()]
        for e in extra:
            if e not in context_files:
                context_files.append(e)
    elif not args.yes:
        console.print("\n[bold]Extra context files[/bold] (read-only context)")
        while True:
            p = Prompt.ask("Add context file path (empty to stop)", default="").strip()
            if not p:
                break
            if p not in context_files:
                context_files.append(p)

    if args.goal:
        console.print(f"\n[bold]Goal:[/bold] {goal}")

    # 5. User Notes + TEACHER INJECTION
    extra_notes = args.notes if args.notes else ""
    if not args.yes and not args.notes:
        extra_notes = Prompt.ask("Constraints / notes (optional)", default="").strip()

    # --- INJECT TEACHER GUIDELINES ---
    console.print("[dim]Scanning task for technical risks...[/dim]")
    teacher_guidelines = detect_tech_stack(goal, allowlist)
    if teacher_guidelines:
        console.print(Panel(teacher_guidelines, title="Teacher Guidelines Injected", style="yellow"))
        # Append to extra_notes so it persists through Planning AND Execution
        extra_notes = f"{extra_notes}\n\n{teacher_guidelines}"

    # --- ARTIFACTS DIR INJECTION ---
    if args.artifacts_dir:
        abs_artifacts = Path(args.artifacts_dir).resolve()
        abs_artifacts.mkdir(parents=True, exist_ok=True)
        artifact_instr = (
            f"\n\n**ARTIFACT MANAGMENT RULE**:\n"
            f"You MUST save ALL generated assets (plots, models, logs, images) to this directory:\n"
            f"`{abs_artifacts}`\n"
            f"Example: `plt.savefig('{abs_artifacts}/plot.png')`\n"
            f"DO NOT save to `./` or `output/` unless explicitly asked."
        )
        extra_notes += artifact_instr
        console.print(f"[cyan]Artifacts directory set: {abs_artifacts}[/cyan]")

    # Print Machine-Readable Log Path for Batch Coder
    print(f"[METADATA] LOG_PATH: {session_dir.resolve()}")

    # 6. Plan (Optimized: Skips LLM for single file tasks)
    # The 'extra_notes' now contains the Teacher Guidelines, so the planner sees them too!
    subtasks = plan_tasks(config, goal, extra_notes, allowlist)
    
    # 7. Execute
    success_count = 0
    for i, subtask in enumerate(subtasks):
        # We pass the same 'extra_notes' (with guidelines) to the subtask loop
        ok = run_subtask_loop(
            config=config,
            subtask=subtask,
            subtask_idx=i,
            allowlist=allowlist,
            context_files=context_files,
            global_notes=extra_notes,
        )
        if ok:
            success_count += 1
        else:
            console.print(f"[red]Sub-task {i+1} failed. Stopping sequence.[/red]")
            break
            
    console.print(Panel(f"Task Complete. Success: {success_count}/{len(subtasks)}", subtitle=str(session_dir)))


if __name__ == "__main__":
    main()

"""

python CodeAgent//mini_claude_codev4.py --goal "Implement Univariate Linear Regression using ONLY PyTorch tensors. Do NOT use torch.nn, torch.optim, or autograd. Write everything in a single task.py file with a complete main() that trains, evaluates, and validates."

python CodeAgent/mini_claude_codev4.py --goal "Implement ML Task: SVM (Score Calibration + ROC/PR). Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

python CodeAgent/mini_claude_codev4.py --api-key "myhpcvllmqwen123" --goal "Implement Multivariate Linear Regression using torch.autograd. Visualize training. Description: Calibrate decision scores; produce ROC/PR curves and AUC. Write a SINGLE self-contained Python file (task.py) with these functions: get_task_metadata, set_seed, get_device, make_dataloaders, build_model, train, evaluate, predict, save_artifacts."

"""
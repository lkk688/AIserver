from __future__ import annotations

"""
verification_manager.py

Owns the entire post-mutation verification pipeline for UniversalAgent:

  1. Shell sandbox execution (run_verification)
  2. Pattern-specific coaching hints (via PromptRegistry)
  3. Truncation detection + code context injection
  4. Local SkillDB lookup on failure
  5. Optional async web-search fallback when SkillDB has no match
  6. Consecutive-failure state tracking
  7. Escalation prompts (via PromptRegistry)
  8. Background async SkillDB saving on success

This keeps UniversalAgent thin. All prompt text lives in PromptRegistry.
All SkillDB I/O reuses helpers from mini_batch_agent_libs.
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from rich.console import Console
from rich.prompt import Confirm

from BatchAgent.mini_batch_agent_libs import (
    _determine_verify_cmd,
    build_debug_prompt,
    format_skill_injection,
    run_shell,
    select_relevant_skills,
    sha1_text,
)
from BatchAgent.mini_batch_agent_base import ActionApplyDiff, ActionWriteFile
from BatchAgent.prompt_registry_v2 import PromptRegistry

console = Console()

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_written_src(mutation_actions: List[Any], workspace_dir: Path) -> str:
    """
    Read the source text of the first written .py file so pattern hints
    can check for numba imports etc.  Returns "" on any error.
    """
    for a in mutation_actions:
        if hasattr(a, "path") and str(a.path).endswith(".py"):
            try:
                p = Path(a.path)
                if not p.is_absolute():
                    p = workspace_dir / p
                return p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
    return ""


def _build_search_query(v_feedback: str, max_len: int = 120) -> str:
    """
    Extract a concise web-search query from the error output.
    Prefers the first concrete Error line; falls back to the first 80 chars.
    """
    m = re.search(r"([A-Z][a-zA-Z]+Error[^\n]{0,80})", v_feedback)
    if m:
        return m.group(1)[:max_len]
    return v_feedback[:max_len].strip()


# ---------------------------------------------------------------------------
# VerificationManager
# ---------------------------------------------------------------------------

class VerificationManager:
    """
    Handles all post-mutation verification logic.

    Usage inside UniversalAgent
    ---------------------------
    Per-agent-instance (in __init__):
        self._vm = VerificationManager(config, skilldb_dir, llm_client=config.client, ...)

    Per-task (at top of execute_task):
        self._vm.reset()

    In the mutation block:
        v_feedback, passed = await self._vm.handle_verification_result(
            content, mutation_actions, allowlist, turn_dir, display_turn
        )
        if v_feedback:
            feedback_blocks.append(v_feedback)

    Dedup-message check:
        if self._vm.last_verify_failed:
            ...
    """

    def __init__(
        self,
        config: Any,
        skilldb_dir: Path,
        llm_client: Any = None,
        model: Optional[str] = None,
        provider: str = "openai",
        web_search_callback: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config:
            AgentConfig — accessed via duck-typing for workspace_dir, verbose,
            require_approval, sandbox_container, max_context, backend.
        skilldb_dir:
            Directory containing *.jsonl skill files (e.g. agent_dir / "skilldb").
        llm_client:
            Optional: the agent's LLM client reused for background skill saving.
            Pass None to skip background skill-save.
        model:
            Model name for the background skill-save LLM call.
        provider:
            "openai" or "anthropic".
        web_search_callback:
            Optional async callable: ``async (query: str) -> str``.
            Called when SkillDB returns no results — used to fetch a live hint.
        """
        self._config = config
        self._skilldb_dir = Path(skilldb_dir)
        self._llm_client = llm_client
        self._model = model or getattr(config, "model", "")
        self._provider = provider
        self._web_search_callback = web_search_callback

        # Per-task mutable state — reset by reset()
        self._consecutive_verify_fails: int = 0
        self._last_verify_error_sig: Optional[str] = None
        self._last_verify_failed: bool = False

    # ------------------------------------------------------------------
    # Public state API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset per-task state.  Call once at the top of every execute_task()."""
        self._consecutive_verify_fails = 0
        self._last_verify_error_sig = None
        self._last_verify_failed = False

    @property
    def last_verify_failed(self) -> bool:
        """True if the most recent verification attempt failed."""
        return self._last_verify_failed

    # ------------------------------------------------------------------
    # Core verification runner (sync)
    # ------------------------------------------------------------------

    def run_verification(
        self,
        content: str,
        mutation_actions: List[Any],
        allowlist: List[str],
        turn_dir: Path,
    ) -> str:
        """
        Execute the sandbox verification command and return a feedback string.

        Includes:
        - Shell execution with timeout / output capture
        - Pattern-specific coaching hint (via PromptRegistry.get_pattern_hint)
        - Truncation detection
        - Relevant code context injection (error-adjacent lines)

        Returns "" if no verification command is required.
        Does NOT modify any instance state — safe to call from sync contexts.
        """
        modified_files: List[str] = []
        for a in mutation_actions:
            if hasattr(a, "path"):
                modified_files.append(a.path)
            elif isinstance(a, ActionApplyDiff):
                paths = re.findall(
                    r"^\+\+\+ b/(.+)$", getattr(a, "diff_text", ""), re.MULTILINE
                )
                modified_files.extend(paths)
        modified_files = list(set(modified_files))

        auto_cmd_match = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
        auto_cmd = auto_cmd_match.group(1).strip() if auto_cmd_match else None
        verify_cmd = _determine_verify_cmd(allowlist, modified_files, auto_cmd, self._config)

        if not verify_cmd:
            console.print("[yellow]No verification command needed.[/yellow]")
            return (
                "✅ File modifications applied to disk.\n"
                "**System Prompt**: No code execution required. "
                "If the task is complete, call `finish_task` immediately. "
                "Do NOT rewrite files unless you found an error."
            )

        if getattr(self._config, "require_approval", False):
            console.print(
                f"\n[bold red]⚠️ Command requires approval:[/bold red] `{verify_cmd}`"
            )
            if not Confirm.ask("Approve execution?"):
                return f"⚠️ User skipped verification for: {verify_cmd}"

        console.print("[cyan]→ Running verification sandbox...[/cyan]")
        code, out = run_shell(
            verify_cmd,
            cap=20000,
            sandbox_container=getattr(self._config, "sandbox_container", None),
        )
        (turn_dir / "verify_stdout.txt").write_text(out, encoding="utf-8")

        if code == 0:
            console.print("[bold green]✅ Verification: Exit 0[/bold green]")
            if getattr(self._config, "verbose", False):
                console.print(f"[dim]{out}[/dim]")
            return (
                f"### ✅ Verification PASSED (exit 0)\n```text\n{out}\n```\n\n"
                "**The script ran successfully.**\n"
                "→ Call `finish_task` immediately.\n"
                "→ Do NOT read, re-run, or rewrite the file — it already works.\n"
                "→ Do NOT use run_bash_command or read_file_chunk after a passing verification.\n"
            )

        # ── Failure path ─────────────────────────────────────────────────────
        console.print(f"[bold red]❌ Verification failed (exit={code})[/bold red]")
        smart_error = build_debug_prompt(out, root_dir=str(self._config.workspace_dir))

        # Pattern hint — reads written source to detect numba usage
        written_src = _extract_written_src(mutation_actions, self._config.workspace_dir)
        pattern_hint = PromptRegistry.get_pattern_hint(out, written_src)

        # Truncation detection + code context
        context_snippet = ""
        truncation_hint = ""
        written_py = [
            a for a in mutation_actions
            if hasattr(a, "path") and str(a.path).endswith(".py")
        ]
        if written_py:
            line_match = re.search(r"line (\d+)", out)
            error_line = int(line_match.group(1)) if line_match else None
            try:
                src_path = Path(written_py[0].path)
                if not src_path.is_absolute():
                    src_path = self._config.workspace_dir / src_path
                file_lines = src_path.read_text(encoding="utf-8").splitlines()
                total_lines = len(file_lines)

                _TRUNC_PATTERNS = (
                    r'["\']$',
                    r'[\(\[{,\\]$',
                    r'\bf["\']',
                )
                last_nonempty = next(
                    (l.rstrip() for l in reversed(file_lines) if l.strip()), ""
                )
                is_truncated = (
                    error_line is not None
                    and total_lines > 50
                    and (total_lines - error_line) <= 15
                    and any(re.search(p, last_nonempty) for p in _TRUNC_PATTERNS)
                )

                if is_truncated:
                    tail_start = max(0, error_line - 5)
                    tail_lines = file_lines[tail_start:]
                    numbered_tail = "\n".join(
                        f"{tail_start + i + 1:4d} | {l}"
                        for i, l in enumerate(tail_lines)
                    )
                    truncation_hint = (
                        f"\n## ⚠️ Truncated File Detected\n"
                        f"The file has {total_lines} lines and the syntax error is at "
                        f"line {error_line} (near the end) — the previous `write_file` "
                        f"was cut off mid-expression by the output-token limit.\n\n"
                        f"**Broken tail (lines {tail_start+1}–{total_lines}):**\n"
                        f"```python\n{numbered_tail}\n```\n\n"
                        "**Fix strategy (choose one):**\n"
                        "- Use `search_and_replace` with `old_text` = the broken last "
                        "few lines and `new_text` = the corrected, complete version.\n"
                        "- OR use `write_file` once more — but this time make sure "
                        "the file ends with a complete `if __name__ == '__main__':` "
                        "block and no open strings.\n"
                        "**Do NOT read the file** — the broken tail is shown above.\n"
                    )
                else:
                    if error_line:
                        lo = max(0, error_line - 20)
                        hi = min(total_lines, error_line + 10)
                    else:
                        lo, hi = 0, min(60, total_lines)
                    numbered = "\n".join(
                        f"{lo + i + 1:4d} | {l}"
                        for i, l in enumerate(file_lines[lo:hi])
                    )
                    context_snippet = (
                        f"\n## Relevant Code (lines {lo+1}–{hi} of {total_lines})\n"
                        f"```python\n{numbered}\n```\n"
                        "Use `search_and_replace` to fix the broken lines above. "
                        "**Do NOT read the file again** — the content is shown here.\n"
                    )
            except Exception:
                pass

        return (
            f"⚠️ [Verification Failed] Exit {code}\n"
            f"Command: {verify_cmd}\n{smart_error}\n"
            f"{pattern_hint}"
            f"{truncation_hint or context_snippet}\n"
            "# Debug Task\n"
            "The file was written but execution failed. **You MUST fix the bug NOW.**\n\n"
            "**Required steps:**\n"
            "1. Read the error and the broken code shown above.\n"
            "2. **Prefer `search_and_replace`** — only replace the broken lines, "
            "not the whole file.\n"
            "3. Only use `write_file` if more than half the file needs to change.\n"
            "4. Do NOT call `read_file_chunk` — the relevant code is already above.\n"
            "5. Do NOT call `finish_task` until verification passes (exit 0).\n"
        )

    # ------------------------------------------------------------------
    # Full async pipeline
    # ------------------------------------------------------------------

    async def handle_verification_result(
        self,
        content: str,
        mutation_actions: List[Any],
        allowlist: List[str],
        turn_dir: Path,
        display_turn: int,
    ) -> Tuple[str, bool]:
        """
        Full pipeline: run_verification → update state → SkillDB → web fallback
        → escalation prompt.

        Returns
        -------
        (augmented_feedback, passed)
            augmented_feedback: the feedback string to append to feedback_blocks
            passed: True if verification passed (or not needed)

        Side effect on success
        ----------------------
        Fires asyncio.create_task(_save_skill_async) in the background —
        does NOT block the current turn.
        """
        v_feedback = self.run_verification(content, mutation_actions, allowlist, turn_dir)

        if not v_feedback:
            return "", True

        passed = (
            "✅ Verification PASSED" in v_feedback
            or "No code execution" in v_feedback
        )
        failed_now = "Verification Failed" in v_feedback

        if passed:
            self._last_verify_failed = False
            self._consecutive_verify_fails = 0
            self._last_verify_error_sig = None
            if self._llm_client:
                asyncio.create_task(
                    self._save_skill_async(content, mutation_actions, v_feedback)
                )
            return v_feedback, True

        if failed_now:
            self._last_verify_failed = True
            cur_sig = sha1_text(v_feedback[:120])
            if cur_sig == self._last_verify_error_sig:
                self._consecutive_verify_fails += 1
            else:
                self._consecutive_verify_fails = 1
                self._last_verify_error_sig = cur_sig

            # ── SkillDB lookup ────────────────────────────────────────────────
            skill_block = ""
            error_query = v_feedback[:400]
            skills = select_relevant_skills(error_query, self._skilldb_dir, topk=4)
            if skills:
                skill_block = format_skill_injection(skills)
            elif self._web_search_callback is not None:
                try:
                    query = _build_search_query(v_feedback)
                    web_result = await self._web_search_callback(query)
                    if web_result:
                        skill_block = (
                            f"## 🔍 Related Solutions (web search)\n{web_result[:800]}\n"
                        )
                except Exception:
                    pass

            # ── Escalation prompt ─────────────────────────────────────────────
            escalation = PromptRegistry.get_verification_escalation_prompt(
                self._consecutive_verify_fails
            )

            parts = [v_feedback]
            if skill_block:
                parts.append(skill_block)
            if escalation:
                parts.append(escalation)
            return "\n\n".join(parts), False

        # Neither PASSED nor FAILED marker — return as-is
        return v_feedback, False

    # ------------------------------------------------------------------
    # Background skill saving (async, fire-and-forget)
    # ------------------------------------------------------------------

    async def _save_skill_async(
        self,
        content: str,
        mutation_actions: List[Any],
        v_feedback: str,
    ) -> None:
        """
        Background coroutine: ask the LLM to summarise the successful fix,
        then append a Skill record to skilldb/<category>.jsonl.

        Never raises — all errors are silently swallowed.
        """
        from BatchAgent.llm_wrapper_v2 import complete_with_continuation_async

        try:
            modified_files = [
                str(a.path) for a in mutation_actions if hasattr(a, "path")
            ]
            prompt = (
                "You just fixed a bug in: " + ", ".join(modified_files) + ".\n"
                "Verification passed after this edit.\n\n"
                "Output a single JSON object (no markdown fences) with keys:\n"
                '  "category": short tech category (e.g. "Python", "NumPy", "Multiprocessing")\n'
                '  "pattern": 2-4 trigger keywords (e.g. "njit numba conditional")\n'
                '  "insight": one-sentence lesson (≤120 chars)\n'
                '  "evidence": ≤80 char snippet from the error or fix\n\n'
                f"Verification output:\n```\n{v_feedback[:600]}\n```\n\n"
                f"Fix description (first 400 chars of assistant response):\n{content[:400]}\n"
            )
            summary_text, _ = await complete_with_continuation_async(
                client=self._llm_client,
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_output_tokens=256,
                model_max_context=getattr(self._config, "max_context", 8192),
                provider=self._provider,
                stream=False,
                verbose=False,
                tool_strategy="text_only",
                allowlist=[],
                backend=getattr(self._config, "backend", "vllm"),
                enable_thinking=False,
            )
            json_match = re.search(r"\{.*?\}", summary_text, re.DOTALL)
            if not json_match:
                return
            obj = json.loads(json_match.group(0))
            category = re.sub(r"[^a-zA-Z0-9_ ]", "", obj.get("category", "General"))
            record = {
                "category": category,
                "pattern": obj.get("pattern", ""),
                "insight": obj.get("insight", ""),
                "evidence": obj.get("evidence", ""),
                "count": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._skilldb_dir.mkdir(parents=True, exist_ok=True)
            skill_file = (
                self._skilldb_dir
                / f"{category.lower().replace(' ', '_')}.jsonl"
            )
            with open(skill_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            console.print(f"[dim]📚 SkillDB: saved → {skill_file.name}[/dim]")
        except Exception:
            pass

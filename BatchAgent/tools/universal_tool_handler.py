from __future__ import annotations

"""
Universal tool handler for the agent runtime.

Design goals
------------
This module is the main execution bridge between parsed LLM actions and the real tool system.

It is responsible for:

1. Receiving parsed AgentAction objects.
2. Splitting them into:
   - observation / read-only tool calls
   - mutation / file-changing actions
3. Dispatching observation tool calls to ToolRouter.
4. Dispatching mutation actions to MutationManager.
5. Returning a single formatted observation string back to the LLM.
6. Tracking written files and artifact metadata for downstream use.

Execution flow
--------------
LLM tool call
  -> UniversalToolHandler.execute()
    -> ToolRouter.dispatch(...)
      -> local_tools / web_tools / document_tools / domain_tools / dynamic tools

LLM mutation action
  -> UniversalToolHandler.execute()
    -> MutationManager.execute_mutations()
      -> write / replace / diff apply
      -> record written files

Notes
-----
- This file should NOT contain business logic for web/document/local tools.
  That belongs in ToolRouter + underlying tool modules.
- This file should NOT parse raw LLM text into actions.
  Parsing belongs elsewhere.
- This file should be robust, observable, and easy to debug.
"""

import re
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from BatchAgent.tools.local_tools import (
    is_git_repo,
    apply_patch_guarded,
    apply_write_files,
    extract_all_diffs,
    apply_fuzzy_patch,
    extract_files_from_diff,
    resolve_path,
)
from BatchAgent.tools.tool_router import ToolRouter

from BatchAgent.mini_batch_agent_base import (
    AgentAction,
    ActionToolCall,
    ActionWriteFile,
    ActionApplyDiff,
    ActionReplaceText,
)

from BatchAgent.tools.tool_registry_runtime import (
    GLOBAL_TOOL_REGISTRY,
    configure_global_tool_registry,
)

"""
In session / agent construction: 
configure_global_tool_registry(
    strategy=config.tool_strategy,
    profile=task_profile,
    domain="general",
    enable_memory=getattr(config, "working_memory", None) is not None,
    enable_web=True,
    enable_document=True,
    enable_code_tools=True,
    enable_mutation=True,
    enable_bash=False,
    enable_parallel=False,
    enable_meta=True,
)
Then create:
UniversalToolHandler
	•	ToolRouter
"""

def _infer_default_tool_profile(self) -> str:
    """
    Best-effort fallback profile inference.

    This is used only when the upper orchestration layer did not explicitly
    configure the runtime tool registry.
    """
    cfg = self.ctx.config

    explicit = getattr(cfg, "task_profile", None)
    if explicit:
        return str(explicit).strip().lower()

    if getattr(cfg, "document_mode", False):
        return "document"
    if getattr(cfg, "research_mode", False):
        return "research"

    return "general"

try:
    from rich.console import Console
    console = Console()
except Exception:
    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = _DummyConsole()


# =============================================================================
# 1. Shared models / context
# =============================================================================

@dataclass
class ToolExecutionContext:
    """
    Immutable-like runtime context shared by the handler and mutation manager.

    Fields
    ------
    config:
        Global runtime/config object from the agent system.

    turn_dir:
        Directory for this agent turn. Used for logs like patch.diff, apply.log, etc.

    allowlist:
        Files/paths the agent is allowed to write to or modify.

    dynamic_tools_mapping:
        Optional mapping of dynamic tool names -> script paths or handlers.
        The current ToolRouter may or may not use this directly, but we keep it here
        so all execution-layer state is centralized.
    """
    config: Any
    turn_dir: Path
    allowlist: List[str]
    dynamic_tools_mapping: Dict[str, str]
    dynamic_tools_registry: Dict[str, Dict[str, Any]]


@dataclass
class ToolExecutionResult:
    """
    Normalized result for a single observation tool call.

    This lets UniversalToolHandler produce a consistent output format for the LLM,
    regardless of which concrete tool was called internally.
    """
    ok: bool
    tool_name: str
    content: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_for_llm(self) -> str:
        """
        Convert this result to a stable markdown block that can be fed back to the LLM.
        """
        if self.ok:
            return f"### Result for {self.tool_name}\n```text\n{self.content}\n```"
        return f"### Error for {self.tool_name}\n```text\n{self.error or self.content}\n```"


# =============================================================================
# 2. Mutation manager
# =============================================================================

class MutationManager:
    """
    Handles all file-changing actions.

    Supported mutation types
    ------------------------
    - ActionWriteFile
    - ActionReplaceText
    - ActionApplyDiff

    Responsibilities
    ----------------
    - Apply writes/replacements/diffs.
    - Keep track of files written or modified.
    - Build file records for downstream artifact upload/UI.
    - Produce human-readable mutation errors for the LLM.

    Important note
    --------------
    The manager is deliberately separate from UniversalToolHandler so that:
    - mutation logic is isolated and easier to test
    - write/diff behavior is easier to debug
    - observation routing stays simple
    """

    def __init__(self, ctx: ToolExecutionContext):
        self.ctx = ctx
        self.written_files: List[str] = []
        self.written_file_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # File record helpers
    # ------------------------------------------------------------------

    def build_written_file_record(self, file_path: Path) -> Dict[str, Any]:
        """
        Build a structured artifact record for a written file.

        The record includes:
        - file name
        - extension
        - size
        - optional uploaded URL
        - local path
        - inline content preview for smaller files

        This is useful for:
        - downstream artifact viewers
        - post-run reporting
        - upload to MinIO / object store
        """
        ext = file_path.suffix.lstrip(".").lower()
        size = file_path.stat().st_size if file_path.exists() else 0
        content = ""

        # For text-like or small files, try to include content inline.
        if ext in {"md", "markdown", "txt"} or size < 200 * 1024:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""

        # Optional artifact upload
        url = ""
        try:
            from BatchAgent.minio_uploader import upload_file

            session_name = getattr(self.ctx.config, "session_dir", Path(".")).name
            object_name = f"agent/{session_name}/{file_path.name}"
            url = upload_file(str(file_path), object_name=object_name) or ""

            if url:
                console.print(f"[blue]MinIO: {url}[/blue]")
        except Exception:
            url = ""

        return {
            "name": file_path.name,
            "ext": ext,
            "size": size,
            "url": url,
            "local_path": str(file_path),
            "content": content,
        }

    def _record_written_file(self, target_path: Path):
        """
        Record a written or modified file into tracking lists.
        """
        self.written_files.append(str(target_path))
        if target_path.exists():
            self.written_file_records.append(self.build_written_file_record(target_path))

    def _record_files_from_diff(self, diff_text: str):
        """
        Extract target files from a diff block and record those that now exist.

        Why needed:
        -----------
        write_file and search_and_replace know their target file explicitly.
        apply_diff may touch multiple files, so we need a helper to record them.
        """
        for m in re.finditer(r"^diff --git a/\S+ b/(\S+)", diff_text, re.MULTILINE):
            raw_path = m.group(1)
            resolved = resolve_path(raw_path, self.ctx.allowlist)
            if resolved and resolved.exists():
                self._record_written_file(resolved)

    def _normalize_write_target(self, requested_path: str) -> Path:
        """
        Resolve a model-provided path into a real target path.

        Behavior
        --------
        1. Try resolving against the allowlist.
        2. If unresolved, fall back to CWD / basename of the requested path.

        The model's path is always honoured — no silent redirects to session_dir.
        Redirecting would cause the verification command (which runs from CWD) to
        fail with "No such file or directory" because the file was moved somewhere
        the model never intended.
        """
        target_path = resolve_path(requested_path, self.ctx.allowlist)

        if target_path is None:
            filename = Path(requested_path).name or "output.md"
            target_path = Path.cwd() / filename
            console.print(
                f"[yellow]Path '{requested_path}' unresolvable — writing to {target_path}[/yellow]"
            )

        return target_path

    # ------------------------------------------------------------------
    # Mutation primitives
    # ------------------------------------------------------------------

    def _apply_write_action(self, action: ActionWriteFile) -> Tuple[bool, List[str]]:
        """
        Apply a full file write.

        Returns:
            (success, error_messages)
        """
        target_path = self._normalize_write_target(action.path)

        # Important:
        # apply_write_files checks allowlist again, so we extend it with the final target path.
        effective_allowlist = list(self.ctx.allowlist) + [str(target_path)]

        ok = apply_write_files(
            [(str(target_path), action.content)],
            effective_allowlist,
            self.ctx.turn_dir,
        )

        if ok:
            self._record_written_file(target_path)
            return True, []

        return False, [f"write_file failed for path: {action.path}"]

    def _apply_replace_action(self, action: ActionReplaceText) -> Tuple[bool, List[str]]:
        """
        Apply a single search-and-replace mutation.

        Returns:
            (success, error_messages)
        """
        target_path = resolve_path(action.path, self.ctx.allowlist)

        if not target_path or not target_path.exists():
            return False, [
                f"search_and_replace FAILED — could not resolve or find file '{action.path}'. "
                f"Use write_file to create it first."
            ]

        try:
            file_text = target_path.read_text(encoding="utf-8")
            if action.old_text not in file_text:
                return False, [
                    f"search_and_replace FAILED — 'old_text' not found in '{action.path}'. "
                    f"The file content may have changed."
                ]

            new_text = file_text.replace(action.old_text, action.new_text, 1)
            target_path.write_text(new_text, encoding="utf-8")
            console.print(f"[green]Replaced text in {target_path}[/green]")

            self._record_written_file(target_path)
            return True, []
        except Exception as e:
            return False, [f"search_and_replace FAILED for '{action.path}': {e}"]

    def _apply_diff_action(self, action: ActionApplyDiff, full_content: str) -> Tuple[bool, List[str]]:
        """
        Apply a unified diff with multiple fallback strategies.

        Strategy order
        --------------
        1. Strict git apply (preferred when in a git repo)
        2. Fuzzy patch by file
        3. New-file extraction fallback from diff text

        Returns:
            (success, error_messages)
        """
        diff = action.diff_text
        errors: List[str] = []
        changes_applied = False

        (self.ctx.turn_dir / "patch.diff").write_text(diff, encoding="utf-8")

        # Strategy 1: strict git apply
        if is_git_repo():
            applied = apply_patch_guarded(
                diff,
                self.ctx.turn_dir,
                auto_approve=getattr(self.ctx.config, "auto_approve", False),
            )
            if applied:
                self._record_files_from_diff(diff)
                console.print("[green]Strict diff applied successfully.[/green]")
                return True, []

        # Strategy 2: fuzzy patch per diff block
        file_diffs = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
        fuzzy_successes, fuzzy_total = 0, 0
        fuzzy_logs = ["\n--- Fuzzy Patch Attempt ---"]

        for fd in file_diffs:
            if not fd.strip().startswith("diff --git"):
                continue

            fuzzy_total += 1
            match = re.search(r"diff --git a/\S+ b/(\S+)", fd)
            if not match:
                continue

            raw_path = match.group(1)
            fuzzy_logs.append(f"Processing diff for: {raw_path}")
            target_path = resolve_path(raw_path, self.ctx.allowlist)

            if target_path:
                if apply_fuzzy_patch(target_path, fd, log_buffer=fuzzy_logs):
                    fuzzy_successes += 1
                    fuzzy_logs.append(">> Success")
                else:
                    fuzzy_logs.append(">> Failed")
            else:
                fuzzy_logs.append(f"Skipping diff for unresolved path: {raw_path}")

        try:
            with open(self.ctx.turn_dir / "apply.log", "a", encoding="utf-8") as f:
                f.write("\n".join(fuzzy_logs) + "\n")
        except Exception as e:
            console.print(f"[yellow]Failed to append to apply.log: {e}[/yellow]")

        if fuzzy_successes > 0:
            changes_applied = True
            self._record_files_from_diff(diff)
            console.print(
                f"[green]Fuzzy patch applied ({fuzzy_successes}/{fuzzy_total} files).[/green]"
            )
            return changes_applied, errors

        # Strategy 3: extract new files from diff
        extracted_diff = extract_all_diffs(full_content) if full_content else None
        if extracted_diff:
            console.print(
                "[yellow]All patch methods failed. Checking for extractable new files in diff...[/yellow]"
            )
            diff_files = extract_files_from_diff(extracted_diff)
            if diff_files and apply_write_files(diff_files, self.ctx.allowlist, self.ctx.turn_dir):
                changes_applied = True
                self.written_files.extend([p for p, _ in diff_files])

                for p, _ in diff_files:
                    resolved = resolve_path(p, self.ctx.allowlist)
                    if resolved and resolved.exists():
                        self.written_file_records.append(self.build_written_file_record(resolved))

                console.print("[green]Wrote new files extracted from diff.[/green]")
                return changes_applied, errors

        errors.append("Patch failed. The diff may be malformed or target paths may be incorrect.")
        return False, errors

    # ------------------------------------------------------------------
    # Public mutation execution
    # ------------------------------------------------------------------

    def execute_mutations(
        self,
        actions: List[AgentAction],
        full_llm_content: str,
    ) -> Tuple[bool, List[str]]:
        """
        Execute a list of mutation actions.

        Returns:
            (has_any_change_been_applied, error_messages)
        """
        has_changes = False
        errors: List[str] = []

        for action in actions:
            if isinstance(action, ActionApplyDiff):
                ok, errs = self._apply_diff_action(action, full_llm_content)
            elif isinstance(action, ActionWriteFile):
                ok, errs = self._apply_write_action(action)
            elif isinstance(action, ActionReplaceText):
                ok, errs = self._apply_replace_action(action)
            else:
                ok, errs = False, [f"Unsupported mutation action: {type(action).__name__}"]

            has_changes = has_changes or ok
            errors.extend(errs)

        return has_changes, errors


# =============================================================================
# 3. Universal handler
# =============================================================================

class UniversalToolHandler:
    """
    Final modular universal tool handler.

    Responsibilities
    ----------------
    1. Execute observation/non-mutation tools via ToolRouter.
    2. Execute mutation tools via MutationManager.
    3. Format tool observations for the LLM.
    4. Expose written file metadata for artifact handling.
    5. Track a semantic 'finish_task' signal.

    This class should remain orchestration-only:
    - no detailed web logic
    - no document parsing logic
    - no low-level patch implementation
    """

    def __init__(
        self,
        config,
        turn_dir: Path,
        allowlist: List[str],
        dynamic_tools_mapping: Optional[Dict[str, str]] = None,
        dynamic_tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_router: Optional["ToolRouter"] = None,
    ):
        self.ctx = ToolExecutionContext(
            config=config,
            turn_dir=turn_dir,
            allowlist=allowlist,
            dynamic_tools_mapping=dynamic_tools_mapping or {},
            dynamic_tools_registry=dynamic_tools_registry or {},
        )

        # New added
        summary = GLOBAL_TOOL_REGISTRY.summary_dict()
        if not summary.get("configured", False):
            inferred_profile = self._infer_default_tool_profile()

            configure_global_tool_registry(
                strategy=getattr(self.ctx.config, "tool_strategy", "hybrid"),
                profile=inferred_profile,
                domain=getattr(self.ctx.config, "tool_domain", "general"),
                enable_memory=getattr(self.ctx.config, "working_memory", None) is not None,
                enable_web=True,
                enable_document=(inferred_profile in {"document", "research"}),
                enable_code_tools=(inferred_profile in {"coding", "general"}),
                enable_mutation=True,
                enable_bash=(inferred_profile == "coding"),
                enable_parallel=False,
                enable_meta=True,
            )

        # Use provided tool_router (shared across turns) or create a fresh one.
        # Sharing the router preserves DocumentToolManager state (loaded documents)
        # across agent turns, preventing "No document loaded" errors.
        if tool_router is not None:
            self.tool_router = tool_router
        else:
            self.tool_router = ToolRouter(
                config=self.ctx.config,
                dynamic_tools_mapping=self.ctx.dynamic_tools_mapping,
            )
        self.mutation_manager = MutationManager(self.ctx)

        # Semantic task-completion flags
        self.task_finished: bool = False
        self.finish_summary: str = ""

    # ------------------------------------------------------------------
    # Exposed state
    # ------------------------------------------------------------------

    @property
    def written_files(self) -> List[str]:
        """
        A flat list of written/modified file paths produced during this execution context.
        """
        return self.mutation_manager.written_files

    @property
    def written_file_records(self) -> List[Dict[str, Any]]:
        """
        Structured metadata records for written artifacts.
        """
        return self.mutation_manager.written_file_records

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_args_for_display(self, args: Dict[str, Any], limit: int = 80) -> Dict[str, Any]:
        """
        Truncate large tool-call arguments for terminal display only.

        This protects CLI/UI logs from huge payloads while keeping the real args untouched.
        """
        safe_args = {}
        for k, v in args.items():
            val_str = str(v)
            safe_args[k] = val_str[:limit] + "... [TRUNCATED]" if len(val_str) > limit else v
        return safe_args

    def _guard_invalid_xml_write_tool(self, name: str) -> Optional[str]:
        """
        Guardrail for invalid pseudo-tool calls such as XML-style WRITE_FILE tool names.

        Actual writes should come in as parsed mutation actions, not as observation tool names.
        """
        if name.upper() in {"WRITE_FILE", "WRITE", "CREATE_FILE"}:
            return (
                "System Guardrail: Do NOT use XML tool calls for writing files. "
                "Use native mutation actions or the expected text mutation format instead."
            )
        return None

    def _split_actions(
        self,
        actions: List[AgentAction],
    ) -> Tuple[List[ActionToolCall], List[AgentAction]]:
        """
        Separate actions into:
        - observation tool calls
        - mutation actions
        """
        tool_actions: List[ActionToolCall] = []
        mutation_actions: List[AgentAction] = []

        for action in actions:
            if isinstance(action, ActionToolCall):
                tool_actions.append(action)
            elif isinstance(action, (ActionWriteFile, ActionApplyDiff, ActionReplaceText)):
                mutation_actions.append(action)

        return tool_actions, mutation_actions

    def _execute_tool_action(self, action: ActionToolCall) -> ToolExecutionResult:
        """
        Execute a single observation tool action.

        Special case:
        -------------
        'finish_task' is treated as a semantic control signal. It does not need to go
        through ToolRouter unless you explicitly implement it there.
        """
        name = action.name
        args = action.args or {}

        # Semantic finish signal
        if name == "finish_task":
            self.task_finished = True
            self.finish_summary = str(args.get("summary", "")).strip()
            return ToolExecutionResult(
                ok=True,
                tool_name=name,
                content="Task marked as finished.",
            )

        # Guard invalid pseudo write tools
        guardrail = self._guard_invalid_xml_write_tool(name)
        if guardrail:
            return ToolExecutionResult(ok=False, tool_name=name, error=guardrail)

        safe_args = self._safe_args_for_display(args)
        console.print(f"[bold magenta]🛠️ Tool Execution:[/bold magenta] {name}({safe_args})")

        try:
            content = self.tool_router.dispatch(name, args)
            return ToolExecutionResult(ok=True, tool_name=name, content=content)
        except Exception as e:
            err = f"Error executing tool '{name}': {e}\n{traceback.format_exc()}"
            console.print(f"[red]{err}[/red]")
            return ToolExecutionResult(ok=False, tool_name=name, error=err)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def parse_and_execute(self, full_llm_content: str) -> Tuple[bool, str]:
        """
        Convenience method to parse raw text and execute resulting actions.
        """
        from BatchAgent.tools.text_action_parser import parse_text_actions
        actions = parse_text_actions(
            content=full_llm_content,
            allowlist=self.ctx.allowlist,
            dynamic_tools_registry=getattr(self.ctx, "dynamic_tools_registry", None),
        )
        return self.execute(actions, full_llm_content)

    def execute(self, actions: List[AgentAction], full_llm_content: str) -> Tuple[bool, str]:
        """
        Execute all parsed actions for one assistant turn.

        Parameters
        ----------
        actions:
            Parsed AgentAction objects from the LLM output.

        full_llm_content:
            Original raw LLM response text. This is useful for diff fallback recovery.

        Returns
        -------
        (has_mutation_occurred, observation_string)

        Where:
        - has_mutation_occurred is True iff at least one mutation was successfully applied.
        - observation_string is the combined markdown/text block that should be fed back to the LLM.
        """
        tool_results: List[str] = []
        has_mutation = False

        tool_actions, mutation_actions = self._split_actions(actions)

        # 1. Execute observation tools
        for action in tool_actions:
            result = self._execute_tool_action(action)
            tool_results.append(result.format_for_llm())

        # 2. Execute mutation tools
        if mutation_actions:
            console.print("[cyan]📝 Applying Code Mutations to Disk...[/cyan]")

            success, mut_errors = self.mutation_manager.execute_mutations(
                mutation_actions,
                full_llm_content,
            )
            has_mutation = success

            if success:
                tool_results.append(
                    "### System Action\n"
                    "✅ File modifications successfully applied to disk. Please proceed to verify or conclude."
                )

            if mut_errors:
                for err in mut_errors:
                    tool_results.append(f"### File Mutation Error\n```text\n{err}\n```")
                    console.print(f"[red]{err}[/red]")

            if not success and not mut_errors:
                tool_results.append(
                    "### System Error\n"
                    "Failed to apply file modifications. The diff may be malformed, "
                    "the file path may be incorrect, or no writeable changes were produced."
                )

        return has_mutation, "\n\n".join(tool_results)
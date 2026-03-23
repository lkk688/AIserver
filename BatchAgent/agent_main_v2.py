"""
agent_main_v2.py — Universal Autonomous Agent v2

Upgraded internals while keeping the same external interface as agent_main.py:
  - AgentConfig dataclass  (same fields + ocr_* + working_memory)
  - UniversalAgent class   (same __init__ / execute_task signatures)
  - check_api_and_context  (unchanged)
  - main_async             (updated CLI with v2 defaults)

v2 changes vs v1
----------------
* llm_wrapper_v2         — native JSON + hybrid + text_only; no separate complete_with_async
* prompt_registry_v2     — strategy-aware system prompts
* tools/tools_registry   — modular tool registry with runtime configuration
* tools/universal_tool_handler v2 — unified ToolRouter + MutationManager
* configure_global_tool_registry — must be called before tool use and on every strategy/domain change
* Memory tools routed through ToolRouter (config.working_memory wires them up automatically)
* LoopDetector class     — extracted loop-break logic for clarity
* _rebuild_tools_and_prompt — single helper for all strategy/domain hot-swaps

Switch between v1 and v2
------------------------
  agent_services_fastapi.py    → from BatchAgent.agent_main    import AgentConfig, UniversalAgent
  agent_services_fastapi_v2.py → from BatchAgent.agent_main_v2 import AgentConfig, UniversalAgent
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import argparse
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── v2 modules ─────────────────────────────────────────────────────────────────
from BatchAgent.llm_wrapper_v2 import complete_with_continuation_async
from BatchAgent.prompt_registry_v2 import PromptRegistry
from BatchAgent.tools.tools_registry import compile_tools_for_provider, get_base_tools
from BatchAgent.tools.tool_registry_runtime import configure_global_tool_registry
from BatchAgent.tools.universal_tool_handler import UniversalToolHandler
from BatchAgent.tools.tool_router import ToolRouter

# ── shared libs (unchanged between v1 and v2) ──────────────────────────────────
from BatchAgent.mini_batch_agent_libs import (
    ensure_dirs,
    estimate_tokens,
    now_stamp,
    read_file,
    sha1_text,
    truncate_to_tokens,
)
from BatchAgent.mini_batch_agent_base import (
    ActionApplyDiff,
    ActionReplaceText,
    ActionToolCall,
    ActionWriteFile,
)
from BatchAgent.tools.domain_tools import DOMAIN_REGISTRY
from BatchAgent.working_memory import WorkingMemory
from BatchAgent.verification_manager import VerificationManager

console = Console()

# ---------------------------------------------------------------------------
# Pending-marker regex (used by _detect_pending_markers and _handle_mutations)
# ---------------------------------------------------------------------------
_PENDING_RE = re.compile(r'\[(?:PENDING|TODO|PLACEHOLDER):[^\]]*\]', re.IGNORECASE)


def _detect_pending_markers(file_path: str) -> List[str]:
    """Scan a written file for [PENDING:...] / [TODO:...] / [PLACEHOLDER:...] markers."""
    try:
        return _PENDING_RE.findall(Path(file_path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class AgentConfig:
    # ── LLM ────────────────────────────────────────────────────────────────────
    client: Any
    model: str
    provider: str = "openai"
    temperature: float = 0.1
    backend: str = "vllm"
    enable_thinking: bool = True

    # ── Context / output ───────────────────────────────────────────────────────
    # Sensible fallback; overridden at runtime by detected_ctx / CLI args.
    max_context: int = 32768
    max_output: int = 4096

    # ── Directories ────────────────────────────────────────────────────────────
    session_dir: Path = field(default_factory=lambda: Path("."))
    workspace_dir: Path = field(default_factory=lambda: Path("."))
    agent_dir: Path = field(default_factory=lambda: Path("."))

    # ── Tool strategy ──────────────────────────────────────────────────────────
    tool_strategy: str = "hybrid"          # native_all | hybrid | text_only
    parallel_thinking: bool = False
    domain: str = "general"

    # ── Document / OCR tools ───────────────────────────────────────────────────
    ocr_server: str = "http://localhost:8002/v1"
    ocr_model: str = "allenai/olmOCR-2-7B-1025-FP8"
    ocr_workspace: str = "./output_old/tmp_ocr"

    # ── Working memory (set by UniversalAgent at task start) ──────────────────
    # ToolRouter reads this via getattr(config, "working_memory", None).
    working_memory: Optional[Any] = field(default=None)

    # ── Behaviour / safety ────────────────────────────────────────────────────
    require_approval: bool = False
    sandbox_container: Optional[str] = None
    verbose: bool = False
    max_retries: int = 4
    enable_turn_limits: bool = False
    max_turns: int = 15
    memory_strategy: str = "sliding_window"   # sliding_window | summarize

    # ── Search keys ───────────────────────────────────────────────────────────
    serper_api_key: str = ""
    tavily_api_key: str = ""
    enable_youtube: bool = False

    # ── Location / time ───────────────────────────────────────────────────────
    location: str = "California, United States"
    current_time: str = ""

    # ── Embedding (document RAG) ───────────────────────────────────────────────
    embedding_base_url: str = "http://localhost:8081/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model: str = "text-embeddings-inference"

    # ── Streaming / SSE ───────────────────────────────────────────────────────
    stream_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = field(default=None)


# =============================================================================
# 2. Tool-error helpers
# =============================================================================

_TOOL_ERROR_PATTERNS = (
    "[Web Search Unavailable]",
    "[Web Search Error]",
    "Network Error during search",
    "Search processing failed",
    "System Error:",
    "Exception:",
    "[Tavily Unavailable]",
    "No document is currently loaded",
    "No document loaded",
    "FileNotFoundError",
    "File not found",
    "Error for ",
)


def _is_tool_error_result(result_str: str) -> bool:
    return any(p in result_str for p in _TOOL_ERROR_PATTERNS)


def _extract_tool_error_summary(result_str: str, max_len: int = 150) -> str:
    m = re.search(r"```text\n(.*?)```", result_str, re.DOTALL)
    text = m.group(1).strip() if m else result_str
    return text[:max_len]


_MEM_SNAPSHOT_RE = re.compile(
    r'^\[Working Memory Snapshot\]\n.*?(?=\n\n### |\Z)',
    re.DOTALL,
)


def _strip_snapshot(content: str) -> str:
    """Remove a stale [Working Memory Snapshot] block from a user message."""
    stripped = _MEM_SNAPSHOT_RE.sub("", content, count=1).lstrip("\n")
    return stripped if stripped.strip() else content


def _fmt_args(args: dict) -> str:
    parts = []
    for k, v in (args or {}).items():
        sv = str(v)
        parts.append(f"{k}={sv[:35]!r}" if len(sv) <= 35 else f"{k}={sv[:32]!r}…")
    return ", ".join(parts)


# Tools whose successful results are worth auto-caching in working memory knowledge.
_AUTO_CACHE_TOOLS = {
    "get_document_overview",
    "read_document_section",
    "search_document",
    "web_search",
    "read_url",
    "read_file_chunk",
}
_AUTO_CACHE_MAX_CHARS = 1200  # truncate long results before storing


def _auto_inject_result_to_memory(
    working_memory: Any,
    action: Any,
    result_text: str,
    turn: int,
) -> None:
    """
    Persist key tool results into working_memory.knowledge.summaries so that
    when the model calls get_memory() it can see what was retrieved — not just
    the action names.
    """
    tool_name = getattr(action, "name", "")
    if tool_name not in _AUTO_CACHE_TOOLS:
        return

    # Extract the plain content between the markdown code-fence.
    m = re.search(r"```text\n(.*?)```", result_text, re.DOTALL)
    content = m.group(1).strip() if m else result_text.strip()
    if not content or content.startswith("No ") or content.startswith("Error"):
        return  # skip errors / empty results

    if len(content) > _AUTO_CACHE_MAX_CHARS:
        content = content[:_AUTO_CACHE_MAX_CHARS] + "…[truncated]"

    args = getattr(action, "args", {}) or {}
    label = tool_name
    if tool_name == "get_document_overview":
        label = f"doc_overview:{args.get('filepath', 'doc')}"
    elif tool_name == "read_document_section":
        label = f"doc_sec:{args.get('section_id') or args.get('page', 'unknown')}"
    elif tool_name == "search_document":
        q = str(args.get("query", ""))[:30]
        label = f"doc_search:{q}"
    elif tool_name in ("web_search", "read_url"):
        q = str(args.get("query") or args.get("url", ""))[:30]
        label = f"web:{q}"
    elif tool_name == "read_file_chunk":
        label = f"file:{args.get('filepath', '')}@{args.get('start_line', '')}"

    try:
        working_memory.update_memory({
            "knowledge": {
                "summaries": {label: f"[T{turn}] {content}"},
            }
        })
    except Exception:
        pass  # never crash the agent loop due to memory injection


# =============================================================================
# 3. Loop Detector
# =============================================================================

class LoopDetector:
    """
    Tracks tool-call and mutation fingerprints across turns to detect and break
    infinite loops without modifying the main agent loop logic.
    """

    def __init__(self, max_tool_history: int = 12, max_mutation_history: int = 8):
        self._tool_fps: list = []
        self._mutation_fps: list = []
        self._failed_fps: dict = {}
        self._max_tool = max_tool_history
        self._max_mut = max_mutation_history

    # ------------------------------------------------------------------
    # Fingerprint builders
    # ------------------------------------------------------------------

    @staticmethod
    def tool_fingerprint(action: ActionToolCall) -> tuple:
        name = str(getattr(action, "name", "") or "").strip()
        args = getattr(action, "args", {}) or {}
        if name == "read_file_chunk":
            raw = str(args.get("filepath", "")).strip().replace("\\", "/")
            norm = re.sub(r"^\./+", "", raw).lower()
            span = 25
            try:
                s = (int(args.get("start_line", 1)) // span) * span
            except Exception:
                s = 0
            try:
                e = (int(args.get("end_line", s + 200)) // span) * span
            except Exception:
                e = s
            return (name, norm, s, e)
        try:
            return (name, json.dumps(args, sort_keys=True))
        except Exception:
            return (name, str(args))

    @staticmethod
    def mutation_fingerprint(action: Any) -> tuple:
        if isinstance(action, ActionWriteFile):
            return ("write_file", str(getattr(action, "path", "")).strip().lower(),
                    sha1_text(getattr(action, "content", "")))
        if isinstance(action, ActionReplaceText):
            return ("search_and_replace", str(getattr(action, "path", "")).strip().lower(),
                    sha1_text(f"{getattr(action, 'old_text', '')}::{getattr(action, 'new_text', '')}"))
        if isinstance(action, ActionApplyDiff):
            return ("apply_diff", sha1_text(getattr(action, "diff_text", "")))
        return ("mutation", sha1_text(str(action)))

    # ------------------------------------------------------------------
    # Tool deduplication
    # ------------------------------------------------------------------

    def check_tools(
        self,
        actions: List[ActionToolCall],
    ) -> Tuple[List[ActionToolCall], List[str], List[Tuple[str, str]]]:
        """
        Returns (deduped_actions, skipped_ok_names, skipped_err_pairs).
        skipped_err_pairs = list of (tool_name, error_summary).

        Policy: allow one retry on any call (threshold = 2).  A call that
        previously errored is blocked sooner (threshold = 1) so the model
        gets a concrete error message rather than a generic loop warning.
        """
        deduped, skipped_ok, skipped_err = [], [], []
        for a in actions:
            fp = self.tool_fingerprint(a)
            count = self._tool_fps.count(fp)
            if fp in self._failed_fps:
                # Previously errored: block after first retry attempt
                if count >= 1:
                    skipped_err.append((a.name, self._failed_fps[fp]))
                else:
                    deduped.append(a)
            elif count >= 2:
                # Succeeded before but called 3+ times: loop
                skipped_ok.append(a.name)
            else:
                deduped.append(a)
        return deduped, skipped_ok, skipped_err

    def record_tools(
        self,
        actions: List[ActionToolCall],
        per_action_errors: Dict[int, str],
    ) -> None:
        for idx, a in enumerate(actions):
            fp = self.tool_fingerprint(a)
            self._tool_fps.append(fp)
            if idx in per_action_errors:
                self._failed_fps[fp] = per_action_errors[idx]
        if len(self._tool_fps) > self._max_tool:
            self._tool_fps[:] = self._tool_fps[-self._max_tool:]

    # ------------------------------------------------------------------
    # Mutation deduplication
    # ------------------------------------------------------------------

    def check_mutations(
        self,
        actions: List[Any],
    ) -> Tuple[List[Any], List[str]]:
        """Returns (deduped_actions, skipped_type_names)."""
        deduped, skipped = [], []
        for a in actions:
            fp = self.mutation_fingerprint(a)
            if self._mutation_fps.count(fp) >= 1:
                skipped.append(fp[0])
            else:
                deduped.append(a)
        return deduped, skipped

    def record_mutations(self, actions: List[Any]) -> List[tuple]:
        fps = [self.mutation_fingerprint(a) for a in actions]
        self._mutation_fps.extend(fps)
        if len(self._mutation_fps) > self._max_mut:
            self._mutation_fps[:] = self._mutation_fps[-self._max_mut:]
        return fps


# =============================================================================
# 3b. Per-task mutable state
# =============================================================================

@dataclass
class _TaskState:
    """Per-task mutable state carried through the execute_task loop."""
    loop_detector: LoopDetector = field(default_factory=LoopDetector)
    consecutive_json_parse_turns: int = 0
    last_write_only_sig: Optional[tuple] = None
    write_only_repeat_count: int = 0
    write_pressure_injected: bool = False
    parallel_complete_pending: bool = False  # True = inject write directive next turn


# =============================================================================
# 4. Universal Agent
# =============================================================================

class UniversalAgent:
    """
    Modular, RL-friendly Autonomous ReAct Agent (v2).

    Drop-in replacement for agent_main.UniversalAgent.
    Uses the v2 tool system: modular ToolRouter, UniversalToolHandler, and
    configure_global_tool_registry for runtime tool activation.
    """

    def __init__(
        self,
        config: AgentConfig,
        system_message: str,
        tools: List[Dict[str, Any]],
    ):
        self.config = config
        self.system_message = system_message
        self.tools = tools

        self.messages: List[Dict[str, str]] = []
        self.rl_trajectory: List[Dict[str, str]] = []

        # Dynamic custom tools (registered at runtime via register_custom_tool)
        self.dynamic_tools_schema: List[Dict[str, Any]] = []
        self.dynamic_tools_mapping: Dict[str, str] = {}
        self.dynamic_tools_registry: Dict[str, Dict[str, Any]] = {}

        # Working memory — initialised in execute_task
        self.working_memory: Optional[WorkingMemory] = None

        # Persistent ToolRouter — shared across all turns so DocumentToolManager
        # keeps its loaded-document state between get_document_overview and
        # subsequent read_document_section / search_document calls.
        self._tool_router: Optional[ToolRouter] = None

        # Verification manager — handles shell sandbox, skilldb, escalation.
        # skilldb lives under agent_dir/skilldb (same convention as mini_batch_agent).
        _skilldb_dir = getattr(config, "agent_dir", config.session_dir) / "skilldb"
        self._vm = VerificationManager(
            config=config,
            skilldb_dir=_skilldb_dir,
            llm_client=config.client,
            model=config.model,
            provider=config.provider,
        )

    # ------------------------------------------------------------------
    # RL trajectory helpers
    # ------------------------------------------------------------------

    def _log_rl(self, role: str, content: str) -> None:
        self.rl_trajectory.append({"role": role, "content": content})

    def _save_trajectory_to_disk(self, task_idx: int, reward: float) -> None:
        log_file = self.config.session_dir / "rl_trajectory.jsonl"
        entry = {
            "task_id": f"{self.config.session_dir.name}_task_{task_idx}",
            "reward": reward,
            "messages": self.rl_trajectory,
        }
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            console.print(f"[dim]RL Trajectory saved (Reward: {reward})[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to write RL trajectory: {e}[/red]")

    # ------------------------------------------------------------------
    # Tool system rebuild (strategy/domain changes)
    # ------------------------------------------------------------------

    def _rebuild_tools_and_prompt(self, strategy: str, domain: str) -> None:
        """
        Reconfigure the global tool registry and rebuild tool schemas + system
        prompt.  Call this whenever strategy or domain changes at runtime.
        """
        configure_global_tool_registry(
            strategy=strategy,
            profile="general",
            domain=domain,
            enable_memory=self.working_memory is not None,
            enable_web=True,
            enable_document=True,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=True,
            enable_parallel=getattr(self.config, "parallel_thinking", False),
            enable_meta=True,
        )
        new_base = get_base_tools(
            strategy=strategy,
            enable_parallel=getattr(self.config, "parallel_thinking", False),
            domain=domain,
        )
        if self.dynamic_tools_schema:
            new_base = new_base + list(self.dynamic_tools_schema)

        self.tools = compile_tools_for_provider(new_base, self.config.provider, strategy)
        self.system_message = PromptRegistry.get_system_prompt(
            strategy=strategy,
            base_tools_list=new_base,
            domain=domain,
            location=self.config.location,
            current_time=getattr(self.config, "current_time", ""),
        )
        if self.messages:
            self.messages[0]["content"] = self.system_message
        console.print(f"[dim]🔄 Tools rebuilt: strategy={strategy}, domain={domain}[/dim]")

    # ------------------------------------------------------------------
    # Concurrent tool execution
    # ------------------------------------------------------------------

    async def _execute_tools_concurrently(
        self,
        tool_actions: List[ActionToolCall],
        turn_dir: Path,
        allowlist: List[str],
    ) -> Tuple[str, Dict[int, str]]:
        """
        Execute observation tools concurrently via UniversalToolHandler.
        Returns (combined_result_str, {action_index: error_summary}).
        Memory tools are included — ToolRouter dispatches them using
        config.working_memory which is set before every turn.
        """
        if not tool_actions:
            return "", {}

        console.print(
            f"[cyan]🚀 Executing {len(tool_actions)} tools concurrently...[/cyan]"
        )

        async def _run_one(action: ActionToolCall) -> str:
            # Share the persistent _tool_router so DocumentToolManager state
            # (loaded documents) survives across turns and tool calls.
            handler = UniversalToolHandler(
                config=self.config,
                turn_dir=turn_dir,
                allowlist=allowlist,
                dynamic_tools_mapping=self.dynamic_tools_mapping,
                dynamic_tools_registry=self.dynamic_tools_registry,
                tool_router=self._tool_router,
            )
            _, result_str = await asyncio.to_thread(handler.execute, [action], "")
            return result_str

        raw = await asyncio.gather(
            *[_run_one(a) for a in tool_actions], return_exceptions=True
        )

        combined, errors = [], {}
        for idx, (action, res) in enumerate(zip(tool_actions, raw)):
            if isinstance(res, Exception):
                err = f"### Result for {action.name}\n```text\nException: {res}\n```"
                combined.append(err)
                errors[idx] = str(res)[:150]
                if self.config.verbose:
                    console.print(f"[red]{err}[/red]")
            else:
                combined.append(res)
                if _is_tool_error_result(res):
                    errors[idx] = _extract_tool_error_summary(res)
                if self.config.verbose:
                    console.print(Panel(
                        res[:500] + "..." if len(res) > 500 else res,
                        title=f"Tool Output: {action.name}",
                        border_style="magenta",
                    ))

        return "\n\n".join(combined), errors

    # ------------------------------------------------------------------
    # Context / memory management
    # ------------------------------------------------------------------

    def _clean_history_messages(self) -> None:
        """
        1. Strip stale [Working Memory Snapshot] blocks from all but the last
           user message (re-injected fresh each turn).
        2. Drop near-empty assistant turns (blank after stripping <think> tags).
        """
        if len(self.messages) < 3:
            return

        for msg in self.messages[:-1]:
            if (
                msg.get("role") == "user"
                and msg.get("content", "").startswith("[Working Memory Snapshot]")
            ):
                msg["content"] = _strip_snapshot(msg["content"])

        kept = [self.messages[0], self.messages[1]]
        for msg in self.messages[2:]:
            if msg.get("role") == "assistant":
                visible = re.sub(
                    r"</?think[^>]*>", "", msg.get("content", ""),
                    flags=re.DOTALL | re.IGNORECASE,
                ).strip()
                if not visible:
                    console.print("[dim]🧹 Dropped empty assistant turn.[/dim]")
                    continue
            kept.append(msg)
        self.messages = kept

    _TOKEN_ESTIMATE_CORRECTION = 1.6

    async def _prune_memory(self) -> int:
        """
        Intelligent context-window management.
        Returns estimated tokens remaining (corrected).
        """
        self._clean_history_messages()

        max_ctx = self.config.max_context
        raw = sum(estimate_tokens(m.get("content", "")) for m in self.messages)
        estimated = int(raw * self._TOKEN_ESTIMATE_CORRECTION)
        headroom = max_ctx - estimated

        if headroom < max_ctx * 0.45:
            console.print(
                f"[yellow]⚠ Context budget low (~{estimated}/{max_ctx} tokens, "
                f"{estimated * 100 // max_ctx}% used). Pruning.[/yellow]"
            )
            for window in (6, 4, 2):
                tail = self.messages[2:][-window:] if len(self.messages) > 2 else []
                candidate = [self.messages[0], self.messages[1]] + tail
                raw = sum(estimate_tokens(m.get("content", "")) for m in candidate)
                estimated = int(raw * self._TOKEN_ESTIMATE_CORRECTION)
                headroom = max_ctx - estimated
                if headroom > 0 or window == 2:
                    self.messages = candidate
                    break
            console.print(
                f"[dim]🧹 Pruned to {len(self.messages)} messages "
                f"(est {estimated}/{max_ctx}, {estimated * 100 // max_ctx}%).[/dim]"
            )

        # Scale sliding-window depth with available context.
        # 16384 is the *reference* baseline (not a cap); formula yields:
        #   16k → 10 turns  |  32k → 14  |  65k → 20  |  128k → 28  |  256k+ → 30
        _ctx = self.config.max_context
        _window = max(10, min(30, int(10 * (_ctx / 16384) ** 0.5)))
        # Summary token budget: ~6% of context, between 500 and 4096 tokens.
        _summary_tokens = max(500, min(4096, _ctx // 16))

        if len(self.messages) > max(8, _window - 2):
            if getattr(self.config, "memory_strategy", "sliding_window") == "summarize":
                msgs_to_compress = self.messages[2:-_window]
                text = "\n\n".join(
                    f"Role: {m['role']}\n{m['content']}" for m in msgs_to_compress
                )
                console.print(
                    "\n[magenta]🧠 Summarizing earlier turns to save context...[/magenta]"
                )
                prompt = (
                    "Concisely summarize the following agent actions, tool results, and "
                    "findings. Focus on WHAT tools were used, WHAT was discovered, and "
                    "WHAT files were changed. No filler.\n\n" + text
                )
                try:
                    summary, _ = await complete_with_continuation_async(
                        client=self.config.client,
                        model=self.config.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_output_tokens=_summary_tokens,
                        model_max_context=self.config.max_context,
                        provider=self.config.provider,
                        stream=False,
                        verbose=False,
                        tool_strategy="text_only",
                        allowlist=[],
                        backend=self.config.backend,
                        enable_thinking=False,
                    )
                    compressed = {
                        "role": "user",
                        "content": (
                            "[SYSTEM: Summary of earlier actions to preserve context]\n"
                            + summary
                        ),
                    }
                    self.messages = (
                        [self.messages[0], self.messages[1], compressed]
                        + self.messages[2:][-_window:]
                    )
                    console.print("[dim]🧹 Memory compressed via LLM summarization.[/dim]")
                except Exception as e:
                    console.print(f"[red]Memory summarization failed ({e}), using sliding window.[/red]")
                    self.messages = [self.messages[0], self.messages[1]] + self.messages[2:][-_window:]
            else:
                self.messages = [self.messages[0], self.messages[1]] + self.messages[2:][-_window:]
                console.print(f"[dim]🧹 Sliding-window prune: kept last {_window} turns.[/dim]")

        return headroom

    # ------------------------------------------------------------------
    # Parallel brainstorming
    # ------------------------------------------------------------------

    async def _execute_batch_brainstorm(
        self,
        action: ActionToolCall,
        allowlist: List[str],
    ) -> str:
        """Batch LLM calls for parallel thinking / brainstorming."""
        problem = action.args.get("problem_statement", "Help me solve this task.")
        n_vars = min(4, max(2, int(action.args.get("n_variations", 3))))

        console.print(
            f"\n[bold magenta]🧠 Launching {n_vars} parallel brainstorm calls...[/bold magenta]"
        )

        bs_messages = self.messages + [{
            "role": "user",
            "content": (
                f"Brainstorm a unique strategy to solve: '{problem}'. "
                "Do NOT write final code — provide step-by-step logic only."
            ),
        }]

        async def _fetch_idea(idx: int) -> str:
            temp = min(0.9, getattr(self.config, "temperature", 0.2) + 0.3 + idx * 0.15)
            content, _ = await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=bs_messages,
                temperature=temp,
                max_output_tokens=min(2048, self.config.max_output // 2),
                model_max_context=self.config.max_context,
                provider=self.config.provider,
                stream=False,
                verbose=False,
                tool_strategy="text_only",
                allowlist=allowlist,
                backend=self.config.backend,
                enable_thinking=self.config.enable_thinking,
            )
            return f"### Approach {idx + 1} (Temp {temp:.2f}):\n{content}\n"

        t0 = time.time()
        ideas = await asyncio.gather(
            *(_fetch_idea(i) for i in range(n_vars)), return_exceptions=True
        )
        valid = [r for r in ideas if not isinstance(r, Exception)]
        console.print(f"[dim]Brainstorming done in {time.time() - t0:.1f}s[/dim]")

        return (
            f"### Parallel Brainstorming Results\n{''.join(valid)}\n\n"
            "Review these approaches, select the best (or a combination), and proceed."
        )

    # ------------------------------------------------------------------
    # execute_task helpers
    # ------------------------------------------------------------------

    def _init_task(
        self,
        task_goal: str,
        resume_messages: Optional[List[Dict[str, str]]],
        resume_rl_trajectory: Optional[List[Dict[str, str]]],
    ) -> None:
        """Initialize message history, working memory, tool registry, and tool router."""
        # Message history
        if resume_messages:
            restored = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in resume_messages
                if isinstance(m, dict) and m.get("role") and m.get("content") is not None
            ]
            if restored and restored[0].get("role") == "system":
                restored[0]["content"] = self.system_message
            else:
                restored.insert(0, {"role": "system", "content": self.system_message})
            self.messages = restored
            self.rl_trajectory = (
                [
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in resume_rl_trajectory
                    if isinstance(m, dict) and m.get("role") and m.get("content") is not None
                ]
                if resume_rl_trajectory
                else list(self.messages)
            )
        else:
            self.messages = [{"role": "system", "content": self.system_message}]
            self._log_rl("system", self.system_message)

        # Working memory
        self.working_memory = WorkingMemory(
            session_dir=self.config.session_dir,
            goal=task_goal,
        )
        self.working_memory.update_memory({"control": {"status": "active", "stage": "planning"}})
        self.config.working_memory = self.working_memory
        console.print("[dim]🧠 Working memory initialised.[/dim]")

        # Tool registry
        _parallel = getattr(self.config, "parallel_thinking", False)
        configure_global_tool_registry(
            strategy=self.config.tool_strategy,
            profile="general",
            domain=self.config.domain,
            enable_memory=True,
            enable_web=True,
            enable_document=True,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=True,
            enable_parallel=_parallel,
            enable_meta=True,
        )
        _base = get_base_tools(
            strategy=self.config.tool_strategy,
            enable_parallel=_parallel,
            domain=self.config.domain,
        )
        self.tools = compile_tools_for_provider(_base, self.config.provider, self.config.tool_strategy)
        _tool_names = [t.get("name", t.get("function", {}).get("name", "?")) for t in self.tools]
        console.print(f"[cyan]🔧 Active tools ({len(_tool_names)}): {', '.join(_tool_names)}[/cyan]")
        if _parallel:
            console.print("[cyan]🧠 Parallel thinking: ON (brainstorm_solutions enabled)[/cyan]")

        # Tool router + verification manager
        self._tool_router = ToolRouter(
            config=self.config,
            dynamic_tools_mapping=self.dynamic_tools_mapping,
        )
        self._vm.reset()

    def _save_turn_debug(self, turn_dir: Path, display_turn: int) -> None:
        (turn_dir / "full_prompt_context.json").write_text(
            json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.messages:
            (turn_dir / "latest_instruction.md").write_text(
                self.messages[-1]["content"], encoding="utf-8"
            )
        console.print(f"\n[bold yellow]>>> ReAct Turn {display_turn} / {self.config.max_turns}[/bold yellow]")

    def _save_action_log(self, turn_dir: Path, actions: list) -> None:
        logs = []
        for a in actions:
            if hasattr(a, "name") and hasattr(a, "args"):
                logs.append({"type": "Tool", "name": a.name, "args": a.args})
            else:
                logs.append({"type": a.__class__.__name__, "path": getattr(a, "path", "N/A")})
        (turn_dir / "parsed_actions.json").write_text(
            json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def _run_llm_call(self, display_turn: int, allowlist: List[str]):
        max_ctx = self.config.max_context
        _raw_input = sum(estimate_tokens(m.get("content", "")) for m in self.messages)
        _est_input = int(_raw_input * self._TOKEN_ESTIMATE_CORRECTION)
        _effective_max_output = min(self.config.max_output, max(256, max_ctx - _est_input - 256))

        async def _call():
            return await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=self.messages,
                temperature=self.config.temperature,
                max_output_tokens=_effective_max_output,
                model_max_context=max_ctx,
                provider=self.config.provider,
                stream=True,
                verbose=self.config.verbose,
                session_dir=self.config.session_dir,
                tools=self.tools,
                tool_strategy=self.config.tool_strategy,
                allowlist=allowlist,
                dynamic_tools_registry=self.dynamic_tools_registry,
                on_event=self.config.stream_callback,
                backend=self.config.backend,
                enable_thinking=self.config.enable_thinking,
            )

        if not self.config.verbose:
            with console.status(
                f"[yellow]Thinking… (Turn {display_turn}/{self.config.max_turns})", spinner="dots"
            ):
                return await _call()
        return await _call()

    def _append_feedback(
        self,
        content: str,
        feedback: str,
        display_turn: int,
        turn_dir: Path,
        filename: str = "user_feedback.md",
    ) -> None:
        """Append assistant content + user feedback to message history."""
        fp = feedback
        if self.config.enable_turn_limits:
            fp = f"[Current Turn: {display_turn} / Max Turns: {self.config.max_turns}]\n{fp}"
        self._log_rl("user", feedback)
        self.messages.append({"role": "assistant", "content": content})
        self.messages.append({"role": "user", "content": fp})
        (turn_dir / filename).write_text(fp, encoding="utf-8")

    async def _handle_finish_action(
        self,
        finish_action: ActionToolCall,
        content: str,
        task_goal: str,
        display_turn: int,
        turn_dir: Path,
        task_idx: int,
    ) -> Optional[Tuple[bool, str]]:
        """
        Returns (success, summary) if the task is done, or None to continue.
        Also handles finish_task rejection (appends to messages directly).
        """
        out_files = (
            self.working_memory.get_memory("artifacts").get("output_files", [])
            if self.working_memory else []
        )
        confirmed = [f for f in out_files if Path(f).exists()]
        mem_stage = (
            self.working_memory.get_memory("control").get("stage", "")
            if self.working_memory else ""
        )
        _WRITE_KW = ("write", "document", "report", "summary", "save", "output", "create", "generate")
        found_kws = [kw for kw in _WRITE_KW if kw in task_goal.lower()]
        needs_file = mem_stage == "writing" or bool(found_kws)

        if needs_file and not confirmed:
            console.print("[bold red]🚫 finish_task REJECTED — no output file found.[/bold red]")
            rejection = PromptRegistry.finish_task_rejected(task_goal, found_kws)
            self._append_feedback(content, rejection, display_turn, turn_dir, "finish_task_rejected.md")
            return None  # continue loop

        # Accepted
        console.print("[bold green]🏁 Agent finished the task.[/bold green]")
        for fpath in confirmed:
            console.print(f"[bold blue]📄 Output: {fpath}[/bold blue]")
            try:
                from BatchAgent.minio_uploader import upload_file
                url = upload_file(
                    fpath,
                    object_name=f"agent/{self.config.session_dir.name}/{Path(fpath).name}",
                ) or ""
                if url:
                    console.print(f"[bold cyan]🌐 MinIO: {url}[/bold cyan]")
            except Exception:
                pass
        if self.working_memory:
            self.working_memory.update_memory({"control": {"status": "done", "stage": "done"}})
            self.working_memory.save_memory()
        self._save_trajectory_to_disk(task_idx, reward=1.0)
        return True, finish_action.args.get("summary", "")

    async def _stream_written_files(self, handler: Any) -> None:
        """Stream file_written events to SSE clients."""
        if not self.config.stream_callback:
            return
        file_records = list(getattr(handler, "written_file_records", []) or [])
        if not file_records and handler.written_files:
            for fpath in handler.written_files:
                fname = Path(fpath).name
                url = ""
                inline = None
                try:
                    from BatchAgent.minio_uploader import upload_file
                    url = upload_file(
                        fpath,
                        object_name=f"agent/{self.config.session_dir.name}/{fname}",
                    ) or ""
                except Exception:
                    pass
                try:
                    ext = Path(fpath).suffix.lower()
                    if ext in {".md", ".markdown", ".txt"} or (
                        os.path.exists(fpath) and os.path.getsize(fpath) < 200 * 1024
                    ):
                        with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                            inline = _f.read()
                except Exception:
                    pass
                file_records.append({
                    "name": fname, "url": url, "local_path": fpath,
                    "content": inline,
                    "size": os.path.getsize(fpath) if os.path.exists(fpath) else 0,
                })
        for rec in file_records:
            await self.config.stream_callback({
                "type": "file_written",
                "name": rec.get("name") or "",
                "url": rec.get("url") or "",
                "local_path": rec.get("local_path") or "",
                "content": rec.get("content"),
                "size": int(rec.get("size") or 0),
            })

    async def _handle_observation_tools(
        self,
        standard_actions: List[ActionToolCall],
        turn_dir: Path,
        display_turn: int,
        allowlist: List[str],
        ts: "_TaskState",
    ) -> Tuple[List[str], bool]:
        """
        Execute observation tools, do loop detection, detect parallel-branches completion.
        Returns (feedback_blocks, parallel_just_completed).
        """
        feedback_blocks: List[str] = []
        parallel_just_completed = False

        if not standard_actions:
            return feedback_blocks, parallel_just_completed

        # Loop detection
        deduped, skipped_ok, skipped_err = ts.loop_detector.check_tools(standard_actions)
        if skipped_err:
            console.print(
                f"[yellow]⚠️ Blocking retry of failed tools: {[n for n, _ in skipped_err]}[/yellow]"
            )
            for tool_name, err_summary in skipped_err:
                feedback_blocks.append(PromptRegistry.tool_retry_blocked(tool_name, err_summary))
        if skipped_ok:
            console.print(f"[yellow]⚠️ Loop detected — skipping repeated: {skipped_ok}[/yellow]")
            feedback_blocks.append(PromptRegistry.loop_detected(skipped_ok))
        standard_actions = deduped

        if not standard_actions:
            return feedback_blocks, parallel_just_completed

        # Update memory stage
        if self.working_memory:
            self.working_memory.update_memory({
                "control": {
                    "stage": "gathering",
                    "current_step": f"T{display_turn}: {', '.join(a.name for a in standard_actions[:3])}",
                }
            })

        concurrent_results, per_errors = await self._execute_tools_concurrently(
            standard_actions, turn_dir, allowlist
        )
        ts.loop_detector.record_tools(standard_actions, per_errors)

        # Auto-record in working memory
        if self.working_memory:
            per_results = concurrent_results.split("\n\n")
            for idx, a in enumerate(standard_actions):
                self.working_memory.record_action(
                    f"T{display_turn}: {a.name}({_fmt_args(a.args)})"
                )
                if idx in per_errors:
                    self.working_memory.record_failure(
                        f"T{display_turn}: {a.name} → {per_errors[idx][:80]}"
                    )
                else:
                    result_text = per_results[idx] if idx < len(per_results) else ""
                    _auto_inject_result_to_memory(
                        self.working_memory, a, result_text, display_turn
                    )

        feedback_blocks.append(f"### Tool Execution Results\n{concurrent_results}")

        # Detect execute_parallel_branches completion → trigger write directive
        parallel_names = {a.name for a in standard_actions}
        if "execute_parallel_branches" in parallel_names:
            parallel_just_completed = True

        return feedback_blocks, parallel_just_completed

    async def _handle_mutations(
        self,
        mutation_actions: List[Any],
        standard_actions: List[ActionToolCall],
        content: str,
        allowlist: List[str],
        turn_dir: Path,
        display_turn: int,
        task_idx: int,
        ts: "_TaskState",
    ) -> Tuple[List[str], Optional[Tuple[bool, str]]]:
        """
        Execute mutations, verify, detect PENDING markers, handle write-only repeat.
        Returns (feedback_blocks, Optional[done_tuple]).
        """
        feedback_blocks: List[str] = []

        if not mutation_actions:
            return feedback_blocks, None

        # Mutation dedup
        deduped_mut, skipped_mut = ts.loop_detector.check_mutations(mutation_actions)
        if skipped_mut:
            feedback_blocks.append(PromptRegistry.mutation_dedup(self._vm.last_verify_failed))
        mutation_actions = deduped_mut

        if not mutation_actions:
            return feedback_blocks, None

        # Execute
        handler = UniversalToolHandler(
            config=self.config,
            turn_dir=turn_dir,
            allowlist=allowlist,
            dynamic_tools_mapping=self.dynamic_tools_mapping,
            dynamic_tools_registry=self.dynamic_tools_registry,
        )
        has_mutation, mutation_res = handler.execute(mutation_actions, content)
        feedback_blocks.append(mutation_res)

        # SSE streaming
        await self._stream_written_files(handler)

        if has_mutation:
            written = list(getattr(handler, "written_files", []))
            if self.working_memory and written:
                for fpath in written:
                    self.working_memory.record_artifact(fpath, is_output=True)
                self.working_memory.update_memory({"control": {
                    "stage": "writing",
                    "current_step": f"T{display_turn}: writing {written[0]}",
                }})
            ts.loop_detector.record_mutations(mutation_actions)

            # Verification
            v_feedback, _v_passed = await self._vm.handle_verification_result(
                content, mutation_actions, allowlist, turn_dir, display_turn
            )
            if v_feedback:
                feedback_blocks.append(v_feedback)

            # [PENDING:...] marker detection on written files
            write_file_actions = [a for a in mutation_actions if isinstance(a, ActionWriteFile)]
            for a in write_file_actions:
                markers = _detect_pending_markers(str(a.path))
                if not markers:
                    # also try workspace-relative
                    try:
                        markers = _detect_pending_markers(
                            str(self.config.workspace_dir / a.path)
                        )
                    except Exception:
                        pass
                if markers:
                    feedback_blocks.append(PromptRegistry.pending_markers_report(markers))

            # Write-only guidance + repeat detection
            write_only = [a for a in mutation_actions if isinstance(a, ActionWriteFile)]
            is_write_only = (
                bool(write_only)
                and len(write_only) == len(mutation_actions)
                and not standard_actions
            )
            if is_write_only:
                sig = tuple(sorted(str(a.path).strip().lower() for a in write_only))
                no_verify = v_feedback and "No code execution" in v_feedback
                if no_verify:
                    feedback_blocks.append(PromptRegistry.write_only_guidance(str(write_only[0].path)))
                if not _v_passed:
                    ts.last_write_only_sig = None
                    ts.write_only_repeat_count = 0
                elif len(set(sig)) == 1:
                    if sig == ts.last_write_only_sig:
                        ts.write_only_repeat_count += 1
                    else:
                        ts.last_write_only_sig = sig
                        ts.write_only_repeat_count = 1
                    if ts.write_only_repeat_count >= 2:
                        self._save_trajectory_to_disk(task_idx, reward=1.0)
                        return feedback_blocks, (True, f"Completed task writing output file: {sig[0]}")
                else:
                    ts.last_write_only_sig = None
                    ts.write_only_repeat_count = 0
            else:
                ts.last_write_only_sig = None
                ts.write_only_repeat_count = 0

        return feedback_blocks, None

    # ------------------------------------------------------------------
    # Main ReAct loop
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        task_goal: str,
        task_idx: int,
        allowlist: List[str],
        prompt_md: str,
        resume_messages: Optional[List[Dict[str, str]]] = None,
        start_turn_index: int = 0,
        resume_rl_trajectory: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, str]:
        console.print(f"\n[bold green]=== Starting Agent Task {task_idx + 1} ===[/bold green]")

        # ── Init ────────────────────────────────────────────────────────────
        self._init_task(task_goal, resume_messages, resume_rl_trajectory)

        max_ctx = self.config.max_context
        ts = _TaskState()
        final_result_text = "Task aborted or failed."

        for current_turn in range(self.config.max_turns):
            ctx_headroom = await self._prune_memory()
            absolute_turn = start_turn_index + current_turn
            display_turn = absolute_turn + 1
            turn_dir = self.config.session_dir / f"{task_idx * 100 + absolute_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            self._save_turn_debug(turn_dir, display_turn)

            if self.config.stream_callback:
                await self.config.stream_callback({
                    "type": "turn_start",
                    "turn": display_turn,
                    "max_turns": self.config.max_turns,
                })

            if current_turn == 0:
                first_prompt = prompt_md
                if self.config.enable_turn_limits:
                    first_prompt = (
                        f"[Current Turn: {display_turn} / Max Turns: {self.config.max_turns}]\n"
                        + first_prompt
                    )
                self._log_rl("user", prompt_md)
                self.messages.append({"role": "user", "content": first_prompt})
                (turn_dir / "user_prompt.md").write_text(prompt_md, encoding="utf-8")

            # ── LLM call ────────────────────────────────────────────────────
            content, actions = await self._run_llm_call(display_turn, allowlist)
            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)
            final_result_text = content
            self._save_action_log(turn_dir, actions)

            # ── finish_task ──────────────────────────────────────────────────
            finish_action = next(
                (a for a in actions if getattr(a, "name", "") == "finish_task"), None
            )
            if finish_action:
                result = await self._handle_finish_action(
                    finish_action, content, task_goal, display_turn, turn_dir, task_idx
                )
                if result is not None:
                    return result
                continue

            # ── JSON parse failures → strategy switch ───────────────────────
            json_parse_errors = [
                a for a in actions
                if isinstance(a, ActionToolCall)
                and getattr(a, "name", "") == "json_parse_error"
                and "write_file" in str(getattr(a, "args", {}).get("error", "")).lower()
            ]
            if json_parse_errors:
                ts.consecutive_json_parse_turns += 1
                if self.config.tool_strategy == "native_all" and ts.consecutive_json_parse_turns >= 2:
                    self.config.tool_strategy = "hybrid"
                    self._rebuild_tools_and_prompt("hybrid", self.config.domain)
                    self._append_feedback(
                        content,
                        PromptRegistry.json_parse_strategy_switch(),
                        display_turn, turn_dir, "strategy_switch_feedback.md",
                    )
                    continue
            else:
                ts.consecutive_json_parse_turns = 0

            # ── No actions → format reminder ─────────────────────────────────
            if not actions:
                console.print("[yellow]No actions detected. Injecting format reminder.[/yellow]")
                self._append_feedback(
                    content,
                    PromptRegistry.no_action_warning(self.config.tool_strategy),
                    display_turn, turn_dir,
                )
                continue

            # ── Segregate ────────────────────────────────────────────────────
            info_actions = [a for a in actions if isinstance(a, ActionToolCall)]
            mutation_actions = [
                a for a in actions
                if isinstance(a, (ActionWriteFile, ActionApplyDiff, ActionReplaceText))
            ]
            feedback_blocks: List[str] = []
            if json_parse_errors:
                feedback_blocks.append(PromptRegistry.json_parse_soft_warning())

            # ── Domain hot-swap ──────────────────────────────────────────────
            load_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "load_domain_tools"), None
            )
            if load_action:
                new_domain = load_action.args.get("domain", "")
                console.print(
                    f"\n[bold green]🔌 Hot-swapping domain: [white]{new_domain}[/white][/bold green]"
                )
                if new_domain in DOMAIN_REGISTRY:
                    self.config.domain = new_domain
                    self._rebuild_tools_and_prompt(self.config.tool_strategy, new_domain)
                    fb = PromptRegistry.domain_loaded(
                        new_domain, [t["name"] for t in DOMAIN_REGISTRY[new_domain]]
                    )
                else:
                    fb = PromptRegistry.domain_unknown(new_domain)
                self._append_feedback(content, fb, display_turn, turn_dir, "load_feedback.md")
                continue

            # ── Brainstorm ───────────────────────────────────────────────────
            bs_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "brainstorm_solutions"), None
            )
            if bs_action and getattr(self.config, "parallel_thinking", False):
                bs_fb = await self._execute_batch_brainstorm(bs_action, allowlist)
                self._append_feedback(content, bs_fb, display_turn, turn_dir, "bs_feedback.md")
                continue

            # ── Custom tool registration ─────────────────────────────────────
            reg_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "register_custom_tool"), None
            )
            if reg_action:
                t_name = reg_action.args.get("tool_name", "custom_tool")
                console.print(
                    f"\n[bold green]🛠️ Registering custom tool: [white]{t_name}[/white][/bold green]"
                )
                new_schema = {
                    "name": t_name,
                    "description": reg_action.args.get("description", ""),
                    "properties": reg_action.args.get("schema_properties", {}),
                    "required": reg_action.args.get("required_args", []),
                }
                self.dynamic_tools_schema.append(new_schema)
                self.dynamic_tools_mapping[t_name] = reg_action.args.get("script_path", "")
                self.dynamic_tools_registry[t_name] = new_schema
                self._rebuild_tools_and_prompt(self.config.tool_strategy, self.config.domain)
                self._append_feedback(
                    content,
                    PromptRegistry.custom_tool_registered(t_name),
                    display_turn, turn_dir, "custom_tool_feedback.md",
                )
                continue

            # ── Observation tools ────────────────────────────────────────────
            _SKIP_NAMES = {
                "brainstorm_solutions", "finish_task", "load_domain_tools",
                "register_custom_tool", "json_parse_error",
            }
            standard_actions = [a for a in info_actions if a.name not in _SKIP_NAMES]
            obs_blocks, parallel_done = await self._handle_observation_tools(
                standard_actions, turn_dir, display_turn, allowlist, ts
            )
            feedback_blocks.extend(obs_blocks)
            if parallel_done or ts.parallel_complete_pending:
                feedback_blocks.append(PromptRegistry.parallel_branches_complete())
                ts.parallel_complete_pending = False

            # ── Mutations ────────────────────────────────────────────────────
            mut_blocks, done_tuple = await self._handle_mutations(
                mutation_actions, standard_actions, content, allowlist,
                turn_dir, display_turn, task_idx, ts,
            )
            feedback_blocks.extend(mut_blocks)
            if done_tuple is not None:
                return done_tuple

            # ── Memory compaction ────────────────────────────────────────────
            if self.working_memory and current_turn > 0 and current_turn % 5 == 0:
                self.working_memory.compact_memory()
                console.print("[dim]🧠 Working memory compacted.[/dim]")

            # ── Build combined feedback ──────────────────────────────────────
            combined_feedback = "\n\n".join(feedback_blocks)
            if not mutation_actions and standard_actions:
                out_files = (
                    self.working_memory.get_memory("artifacts").get("output_files", [])
                    if self.working_memory else []
                )
                combined_feedback += PromptRegistry.read_only_guidance(
                    out_files[0] if out_files else None
                )
            if not combined_feedback.strip():
                combined_feedback = PromptRegistry.empty_turn_fallback()

            # ── Write-pressure ───────────────────────────────────────────────
            _ctx_pct = 100 - int(ctx_headroom * 100 / max_ctx)
            if ctx_headroom < max_ctx * 0.40 and not ts.write_pressure_injected:
                ts.write_pressure_injected = True
                out_files = (
                    self.working_memory.get_memory("artifacts").get("output_files", [])
                    if self.working_memory else []
                )
                combined_feedback += PromptRegistry.write_pressure(
                    _ctx_pct, out_files[0] if out_files else None
                )
                console.print(
                    f"[bold yellow]⚠ Write-pressure injected ({_ctx_pct}% context used).[/bold yellow]"
                )

            # ── Memory snapshot + append ─────────────────────────────────────
            if self.working_memory:
                self.working_memory.save_memory()
                (turn_dir / "working_memory_snapshot.json").write_text(
                    json.dumps(self.working_memory.get_memory(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                combined_feedback = (
                    f"[Working Memory Snapshot]\n{self.working_memory.to_context_string()}\n\n"
                    + combined_feedback
                )

            (turn_dir / "tool_execution_results.md").write_text(
                combined_feedback, encoding="utf-8"
            )
            fp = combined_feedback
            if self.config.enable_turn_limits:
                fp = (
                    f"[Current Turn: {display_turn} / Max Turns: {self.config.max_turns}]\n"
                    + fp
                )
            self._log_rl("user", combined_feedback)
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": fp})
            (turn_dir / "user_feedback.md").write_text(fp, encoding="utf-8")

        # ── Max turns exceeded ───────────────────────────────────────────────────
        console.print("[bold red]Max ReAct turns exceeded. Task aborted.[/bold red]")
        if self.working_memory:
            self.working_memory.update_memory({"control": {"status": "failed", "stage": "done"}})
            self.working_memory.save_memory()
        self._save_trajectory_to_disk(task_idx, reward=-1.0)
        return False, final_result_text


# =============================================================================
# 5. API / context helpers
# =============================================================================

async def check_api_and_context(
    client: Any, provider: str, model: str, default_ctx: int
) -> int:
    """Check API connectivity and auto-detect max context length."""
    try:
        if provider == "anthropic":
            return 200000
        if hasattr(client, "models"):
            models = await client.models.list()
            for m in models.data:
                if m.id == model:
                    if hasattr(m, "max_model_len") and m.max_model_len:
                        return int(m.max_model_len)
        await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=5
        )
        return default_ctx
    except Exception as e:
        console.print(f"\n[bold red]CRITICAL: Cannot connect to LLM API: {e}[/bold red]")
        console.print("[yellow]Check your vLLM server or API key.[/yellow]")
        sys.exit(1)


# =============================================================================
# 6. CLI entry point
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Universal Autonomous ReAct Agent v2")
    parser.add_argument("--goal", help="Task goal/description")
    parser.add_argument("--allowlist", default="",
                        help="Comma-separated paths the agent may write to")
    parser.add_argument("--context", default="",
                        help="Comma-separated read-only context files")

    # Provider / model
    parser.add_argument("--tool-strategy", choices=["native_all", "hybrid", "text_only"],
                        default="native_all")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model",
                        default=os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--base-url",
                        default=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"))
    parser.add_argument("--api-key",
                        default=os.environ.get("VLLM_API_KEY", "EMPTY"))

    # Embedding
    parser.add_argument("--embedding-base-url",
                        default=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8081/v1"))
    parser.add_argument("--embedding-api-key",
                        default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model",
                        default=os.environ.get("EMBEDDING_MODEL", "text-embeddings-inference"))

    # OCR / document tools
    parser.add_argument("--ocr-server", default="http://localhost:8002/v1")
    parser.add_argument("--ocr-model", default="allenai/olmOCR-2-7B-1025-FP8")
    parser.add_argument("--ocr-workspace", default="./output_old/tmp_ocr")

    # Search keys
    parser.add_argument("--serper-key",
                        default=os.environ.get("SERPER_API_KEY", ""))
    parser.add_argument("--tavily-key",
                        default=os.environ.get("TAVILY_API_KEY", ""))
    parser.add_argument("--enable-youtube", action="store_true")

    # Features
    parser.add_argument("--parallel-thinking", action="store_true")
    parser.add_argument("--output-dir", default="./agent_workspace")
    parser.add_argument("--sandbox", default=None)
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-context", type=int, default=0,
                        help="Override detected context window (0 = auto-detect)")
    parser.add_argument("--domain", default="general",
                        choices=["general", "medical", "academic", "news",
                                 "software_eng", "math", "science", "language",
                                 "business", "assistant", "sales_support"])
    parser.add_argument("--location", default="California, United States")
    parser.add_argument("--enable-turn-limits", action="store_true")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--memory-strategy", choices=["sliding_window", "summarize"],
                        default="summarize")
    parser.add_argument("--backend", choices=["vllm", "llama.cpp", "openai", "anthropic"],
                        default="vllm")

    args = parser.parse_args()

    # ── Directory setup ────────────────────────────────────────────────────────
    workspace_dir = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = (workspace_dir / ".agent").resolve()
    ensure_dirs(agent_dir)
    session_dir = agent_dir / "sessions" / now_stamp()
    session_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(workspace_dir)
    os.environ["EMBEDDING_BASE_URL"] = args.embedding_base_url
    os.environ["EMBEDDING_API_KEY"] = args.embedding_api_key
    os.environ["EMBEDDING_MODEL"] = args.embedding_model

    # ── Build client ───────────────────────────────────────────────────────────
    if args.provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=args.api_key)
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            http_client=httpx.AsyncClient(timeout=1200.0),
        )

    # ── Pre-flight UI ──────────────────────────────────────────────────────────
    console.print(Text(
        " ╔══════════════════════════════════════════════════════════════╗\n"
        " ║           UNIVERSAL AUTONOMOUS AGENT ENGINE  v2            ║\n"
        " ╚══════════════════════════════════════════════════════════════╝",
        style="bold green",
    ))

    with Status("[bold cyan]Connecting to LLM API…[/bold cyan]", spinner="bouncingBar"):
        detected_ctx = await check_api_and_context(
            client, args.provider, args.model, args.max_context
        )

    # Scale max_output proportionally to the detected context window.
    # 4096 is suitable for 16k contexts; scale linearly, cap at 32768.
    detected_max_output = min(32768, max(4096, (detected_ctx // 4)))

    console.print(f"[green]✔ API Connected[/green]")
    console.print(f"[green]✔ Context: {detected_ctx} tokens  /  max_output: {detected_max_output}[/green]")
    console.print(f"[green]✔ Workspace: {workspace_dir}[/green]")
    console.print(f"[green]✔ Embedding: {args.embedding_base_url} / {args.embedding_model}[/green]")
    if args.sandbox:
        console.print(f"[yellow]⚠ Sandbox: {args.sandbox}[/yellow]")
    if args.require_approval:
        console.print("[yellow]⚠ Human-in-the-loop: approval required for mutations[/yellow]")

    # ── Build config ───────────────────────────────────────────────────────────
    config = AgentConfig(
        client=client,
        model=args.model,
        provider=args.provider,
        temperature=0.1,
        backend=args.backend,
        max_context=detected_ctx,
        max_output=detected_max_output,
        session_dir=session_dir,
        workspace_dir=workspace_dir,
        agent_dir=agent_dir,
        tool_strategy=args.tool_strategy,
        parallel_thinking=args.parallel_thinking,
        domain=args.domain,
        ocr_server=args.ocr_server,
        ocr_model=args.ocr_model,
        ocr_workspace=args.ocr_workspace,
        require_approval=args.require_approval,
        sandbox_container=args.sandbox,
        verbose=args.verbose,
        serper_api_key=args.serper_key,
        tavily_api_key=args.tavily_key,
        enable_youtube=args.enable_youtube,
        location=args.location,
        current_time=os.environ.get("BATCHAGENT_CURRENT_TIME", ""),
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        embedding_model=args.embedding_model,
        enable_turn_limits=args.enable_turn_limits,
        max_turns=args.max_turns,
        memory_strategy=args.memory_strategy,
    )

    # ── Build tools + prompt ───────────────────────────────────────────────────
    configure_global_tool_registry(
        strategy=config.tool_strategy,
        profile="general",
        domain=config.domain,
        enable_memory=True,
        enable_web=True,
        enable_document=True,
        enable_code_tools=True,
        enable_mutation=True,
        enable_bash=True,
        enable_parallel=config.parallel_thinking,
        enable_meta=True,
    )
    base_tools = get_base_tools(
        strategy=config.tool_strategy,
        enable_parallel=config.parallel_thinking,
        domain=config.domain,
    )
    compiled_tools = compile_tools_for_provider(base_tools, config.provider, config.tool_strategy)
    system_prompt = PromptRegistry.get_system_prompt(
        strategy=config.tool_strategy,
        base_tools_list=base_tools,
        domain=config.domain,
        location=config.location,
        current_time=config.current_time,
    )

    # ── Goal collection ────────────────────────────────────────────────────────
    goal = args.goal
    if not goal:
        print()
        goal = Prompt.ask("[bold magenta]🎯 Enter your Goal / Task[/bold magenta]").strip()

    allowlist = [x.strip() for x in args.allowlist.split(",") if x.strip()]
    context_files = [x.strip() for x in args.context.split(",") if x.strip()]

    def _content_injector(files: list) -> str:
        out = ""
        for f in list(dict.fromkeys(files)):
            text = read_file(str(f))
            if text and not text.startswith("[MISSING FILE]"):
                if estimate_tokens(text) > 8000:
                    text = truncate_to_tokens(text, 8000)
                out += f"### File: {f}\n```text\n{text}\n```\n"
        return out

    task_prompt = PromptRegistry.format_task(goal, [], [], workspace_dir.name, _content_injector)

    # ── Run ────────────────────────────────────────────────────────────────────
    agent = UniversalAgent(config=config, system_message=system_prompt, tools=compiled_tools)
    success, _ = await agent.execute_task(
        task_goal=goal,
        task_idx=0,
        allowlist=allowlist,
        prompt_md=task_prompt,
    )

    if success:
        console.print("\n[bold green]🏁 Agent completed successfully![/bold green]")
    else:
        console.print("\n[bold red]💀 Agent terminated / failed.[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")


"""
Example usage:

python BatchAgent/agent_main_v2.py --tool-strategy native_all --verbose

python BatchAgent/agent_main_v2.py --model "qwen-35b" --base-url "http://127.0.0.1:8000/v1" --api-key "myhpcvllmqwen101" --tool-strategy native_all --verbose


python BatchAgent/agent_main_v2.py --provider anthropic \\
  --model claude-3-5-sonnet-20241022 --tool-strategy native_all

VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen3.5-9B \\
    --served-model-name qwen3.5-9b \\
    --enable-auto-tool-choice \\
    --tool-call-parser qwen3_xml
"""

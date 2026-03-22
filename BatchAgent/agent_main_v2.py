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
from rich.prompt import Confirm, Prompt
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
    _determine_verify_cmd,
    build_debug_prompt,
    ensure_dirs,
    estimate_tokens,
    now_stamp,
    read_file,
    run_shell,
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

console = Console()


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
    # Mutation verification
    # ------------------------------------------------------------------

    def _verify_mutations(
        self,
        content: str,
        mutation_actions: List[Any],
        allowlist: List[str],
        turn_dir: Path,
    ) -> str:
        """
        Run a sandbox verification command after mutations.
        Returns a feedback string for the agent.
        """
        modified_files = []
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
        verify_cmd = _determine_verify_cmd(allowlist, modified_files, auto_cmd, self.config)

        if not verify_cmd:
            console.print("[yellow]No verification command needed.[/yellow]")
            return (
                "✅ File modifications applied to disk.\n"
                "**System Prompt**: No code execution required. "
                "If the task is complete, call `finish_task` immediately. "
                "Do NOT rewrite files unless you found an error."
            )

        if getattr(self.config, "require_approval", False):
            console.print(f"\n[bold red]⚠️ Command requires approval:[/bold red] `{verify_cmd}`")
            if not Confirm.ask("Approve execution?"):
                return f"⚠️ User skipped verification for: {verify_cmd}"

        console.print("[cyan]→ Running verification sandbox...[/cyan]")
        code, out = run_shell(
            verify_cmd,
            cap=20000,
            sandbox_container=getattr(self.config, "sandbox_container", None),
        )
        (turn_dir / "verify_stdout.txt").write_text(out, encoding="utf-8")

        if code == 0:
            console.print("[bold green]✅ Verification: Exit 0[/bold green]")
            if self.config.verbose:
                console.print(f"[dim]{out}[/dim]")
            return (
                f"### ✅ Verification PASSED (exit 0)\n```text\n{out}\n```\n\n"
                "**The script ran successfully.**\n"
                "→ Call `finish_task` immediately.\n"
                "→ Do NOT read, re-run, or rewrite the file — it already works.\n"
                "→ Do NOT use run_bash_command or read_file_chunk after a passing verification.\n"
            )
        else:
            console.print(f"[bold red]❌ Verification failed (exit={code})[/bold red]")
            smart_error = build_debug_prompt(out, root_dir=str(self.config.workspace_dir))

            # Extract the error line number and inject that code region so the
            # model can patch immediately — without needing to re-read the file.
            context_snippet = ""
            written_py = [
                a for a in mutation_actions
                if hasattr(a, "path") and str(a.path).endswith(".py")
            ]
            if written_py:
                # Find the first line-number reference in the traceback
                line_match = re.search(r"line (\d+)", out)
                error_line = int(line_match.group(1)) if line_match else None
                try:
                    src_path = Path(written_py[0].path)
                    if not src_path.is_absolute():
                        src_path = self.config.workspace_dir / src_path
                    lines = src_path.read_text(encoding="utf-8").splitlines()
                    if error_line:
                        lo = max(0, error_line - 20)
                        hi = min(len(lines), error_line + 10)
                    else:
                        lo, hi = 0, min(60, len(lines))
                    numbered = "\n".join(
                        f"{lo + i + 1:4d} | {l}" for i, l in enumerate(lines[lo:hi])
                    )
                    context_snippet = (
                        f"\n## Relevant Code (lines {lo+1}–{hi})\n"
                        f"```python\n{numbered}\n```\n"
                        "Use `search_and_replace` or `write_file` to fix the lines above. "
                        "**Do NOT read the file again** — the content is shown here.\n"
                    )
                except Exception:
                    pass

            return (
                f"⚠️ [Verification Failed] Exit {code}\n"
                f"Command: {verify_cmd}\n{smart_error}\n"
                f"{context_snippet}\n"
                "# Debug Task\n"
                "The file was written but execution failed. **You MUST fix the bug NOW.**\n\n"
                "**Required steps:**\n"
                "1. Read the error traceback above carefully — the broken code is shown.\n"
                "2. Syntax error → use `search_and_replace` to fix just the bad lines, "
                "or `write_file` to rewrite the entire file.\n"
                "3. Runtime error → trace the failing line and patch it with "
                "`search_and_replace`.\n"
                "4. Do NOT call `read_file_chunk` — the relevant code is already above.\n"
                "5. Do NOT call `finish_task` until verification passes (exit 0).\n"
            )

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
        console.print(
            f"\n[bold green]=== Starting Agent Task {task_idx + 1} ===[/bold green]"
        )

        # ── Restore or initialise message history ──────────────────────────────
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

        # ── Initialise working memory ──────────────────────────────────────────
        self.working_memory = WorkingMemory(
            session_dir=self.config.session_dir,
            goal=task_goal,
        )
        self.working_memory.update_memory({"control": {"status": "active", "stage": "planning"}})
        # Wire working memory into config so ToolRouter's memory handlers find it.
        self.config.working_memory = self.working_memory
        console.print("[dim]🧠 Working memory initialised.[/dim]")

        # ── Configure the global tool registry for this task ──────────────────
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
            enable_parallel=getattr(self.config, "parallel_thinking", False),
            enable_meta=True,
        )

        # ── Create persistent ToolRouter (shared across all turns) ────────────
        # This preserves DocumentToolManager state so a document loaded by
        # get_document_overview stays available for read_document_section calls
        # in later turns.
        self._tool_router = ToolRouter(
            config=self.config,
            dynamic_tools_mapping=self.dynamic_tools_mapping,
        )

        MAX_REACT_TURNS = self.config.max_turns
        final_result_text = "Task aborted or failed."

        # ── Per-task state ─────────────────────────────────────────────────────
        loop_detector = LoopDetector()
        consecutive_json_parse_turns = 0
        last_write_only_signature: Optional[tuple] = None
        write_only_repeat_count = 0
        _write_pressure_injected = False
        max_ctx = self.config.max_context

        for current_turn in range(MAX_REACT_TURNS):
            ctx_headroom = await self._prune_memory()

            absolute_turn = start_turn_index + current_turn
            display_turn = absolute_turn + 1
            turn_dir = self.config.session_dir / f"{task_idx * 100 + absolute_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)

            # Debug: save full prompt context for every turn
            (turn_dir / "full_prompt_context.json").write_text(
                json.dumps(self.messages, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if self.messages:
                (turn_dir / "latest_instruction.md").write_text(
                    self.messages[-1]["content"], encoding="utf-8"
                )

            console.print(
                f"\n[bold yellow]>>> ReAct Turn {display_turn} / {MAX_REACT_TURNS}[/bold yellow]"
            )

            # Notify SSE clients
            if self.config.stream_callback:
                await self.config.stream_callback({
                    "type": "turn_start",
                    "turn": display_turn,
                    "max_turns": MAX_REACT_TURNS,
                })

            # First turn: append the task prompt
            current_user_prompt = prompt_md if current_turn == 0 else ""
            if self.config.enable_turn_limits:
                current_user_prompt = (
                    f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n"
                    + current_user_prompt
                )

            if current_turn == 0:
                self._log_rl("user", prompt_md)
                self.messages.append({"role": "user", "content": current_user_prompt})
                (turn_dir / "user_prompt.md").write_text(prompt_md, encoding="utf-8")

            # ── 1. LLM Generation ────────────────────────────────────────────────
            async def _run_llm():
                return await complete_with_continuation_async(
                    client=self.config.client,
                    model=self.config.model,
                    messages=self.messages,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_output,
                    model_max_context=self.config.max_context,
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
                    f"[yellow]Thinking… (Turn {display_turn}/{MAX_REACT_TURNS})",
                    spinner="dots",
                ):
                    content, actions = await _run_llm()
            else:
                content, actions = await _run_llm()

            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)
            final_result_text = content

            # Debug: structured action log
            action_logs = []
            for a in actions:
                if hasattr(a, "name") and hasattr(a, "args"):
                    action_logs.append({"type": "Tool", "name": a.name, "args": a.args})
                else:
                    action_logs.append({"type": a.__class__.__name__, "path": getattr(a, "path", "N/A")})
            (turn_dir / "parsed_actions.json").write_text(
                json.dumps(action_logs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # ── 2. Check for task completion ─────────────────────────────────────
            finish_action = next(
                (a for a in actions if getattr(a, "name", "") == "finish_task"), None
            )
            if finish_action:
                out_files = (
                    self.working_memory.get_memory("artifacts").get("output_files", [])
                    if self.working_memory else []
                )
                confirmed = [f for f in out_files if Path(f).exists()]
                mem_stage = (
                    self.working_memory.get_memory("control").get("stage", "")
                    if self.working_memory else ""
                )
                _WRITE_KW = ("write", "document", "report", "summary", "save",
                             "output", "create", "generate")
                needs_file = (
                    mem_stage == "writing"
                    or any(kw in task_goal.lower() for kw in _WRITE_KW)
                )
                if needs_file and not confirmed:
                    console.print(
                        "[bold red]🚫 finish_task REJECTED — task requires an output file "
                        "but none found on disk.[/bold red]"
                    )
                    rejection = (
                        "❌ **finish_task REJECTED**: Your task requires writing an output file "
                        f"(goal contains: {[kw for kw in _WRITE_KW if kw in task_goal.lower()]}), "
                        "but no output file was found on disk.\n\n"
                        "Write the file first using `write_file`, then call `finish_task`.\n"
                        "Use information already gathered — do NOT read more sections."
                    )
                    self.messages.append({"role": "assistant", "content": content})
                    fp = rejection
                    if self.config.enable_turn_limits:
                        fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                    self._log_rl("user", rejection)
                    self.messages.append({"role": "user", "content": fp})
                    (turn_dir / "finish_task_rejected.md").write_text(fp, encoding="utf-8")
                    continue

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
                return True, finish_action.args.get("summary", final_result_text)

            # ── 3. Detect repeated JSON parse failures → strategy fallback ───────
            json_parse_errors = [
                a for a in actions
                if isinstance(a, ActionToolCall)
                and getattr(a, "name", "") == "json_parse_error"
                and "write_file" in str(getattr(a, "args", {}).get("error", "")).lower()
            ]
            if json_parse_errors:
                consecutive_json_parse_turns += 1
                if (
                    self.config.tool_strategy == "native_all"
                    and consecutive_json_parse_turns >= 2
                ):
                    self.config.tool_strategy = "hybrid"
                    self._rebuild_tools_and_prompt("hybrid", self.config.domain)
                    parse_fb = (
                        "System switched strategy native_all → hybrid after repeated "
                        "write_file JSON parse failures. Use native JSON for observations "
                        "and JSON tools for file mutations."
                    )
                    self.messages.append({"role": "assistant", "content": content})
                    fp = parse_fb
                    if self.config.enable_turn_limits:
                        fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                    self._log_rl("user", parse_fb)
                    self.messages.append({"role": "user", "content": fp})
                    (turn_dir / "strategy_switch_feedback.md").write_text(fp, encoding="utf-8")
                    continue
            else:
                consecutive_json_parse_turns = 0

            # ── 4. No actions — format warning ───────────────────────────────────
            if not actions:
                console.print("[yellow]No actions detected. Injecting format reminder.[/yellow]")
                if self.config.tool_strategy == "native_all":
                    fb = (
                        "⚠️ System Warning: No valid tool calls detected.\n"
                        "Use native JSON function calling (e.g. `web_search`, `write_file`). "
                        "Do NOT put JSON in plain-text code blocks."
                    )
                elif self.config.tool_strategy == "text_only":
                    fb = (
                        "⚠️ System Warning: No valid XML tool calls detected.\n"
                        "Use `<tool_call><web_search><query>…</query></web_search></tool_call>` "
                        "or the XML write_file format."
                    )
                else:
                    fb = (
                        "⚠️ System Warning: No valid tool calls or file modifications detected.\n"
                        "Use native JSON tools for searching/reading and `write_file` / "
                        "`search_and_replace` for file mutations. "
                        "Call `finish_task` if done."
                    )
                self.messages.append({"role": "assistant", "content": content})
                fp = fb
                if self.config.enable_turn_limits:
                    fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                self._log_rl("user", fb)
                self.messages.append({"role": "user", "content": fp})
                (turn_dir / "user_feedback.md").write_text(fp, encoding="utf-8")
                continue

            # ── 5. Segregate actions ─────────────────────────────────────────────
            info_actions = [a for a in actions if isinstance(a, ActionToolCall)]
            mutation_actions = [
                a for a in actions
                if isinstance(a, (ActionWriteFile, ActionApplyDiff, ActionReplaceText))
            ]
            feedback_blocks: List[str] = []
            if json_parse_errors:
                feedback_blocks.append(
                    "Tool JSON parsing failed for at least one call. "
                    "If writing files, use the JSON write_file/search_and_replace tool."
                )

            # ── 6. Domain hot-swap ───────────────────────────────────────────────
            load_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "load_domain_tools"),
                None,
            )
            if load_action:
                new_domain = load_action.args.get("domain", "")
                console.print(
                    f"\n[bold green]🔌 Hot-swapping domain: [white]{new_domain}[/white][/bold green]"
                )
                if new_domain in DOMAIN_REGISTRY:
                    self.config.domain = new_domain
                    self._rebuild_tools_and_prompt(self.config.tool_strategy, new_domain)
                    tool_names = [t["name"] for t in DOMAIN_REGISTRY[new_domain]]
                    load_fb = (
                        f"✅ Loaded domain plugin '{new_domain}'.\n"
                        f"New tools: {', '.join(tool_names)}. Use them to complete the task."
                    )
                else:
                    load_fb = f"❌ Unknown domain plugin: '{new_domain}'."
                self.messages.append({"role": "assistant", "content": content})
                fp = load_fb
                if self.config.enable_turn_limits:
                    fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                self._log_rl("user", load_fb)
                self.messages.append({"role": "user", "content": fp})
                (turn_dir / "load_feedback.md").write_text(fp, encoding="utf-8")
                continue

            # ── 7. Parallel brainstorm ───────────────────────────────────────────
            bs_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "brainstorm_solutions"),
                None,
            )
            if bs_action and getattr(self.config, "parallel_thinking", False):
                bs_fb = await self._execute_batch_brainstorm(bs_action, allowlist)
                self.messages.append({"role": "assistant", "content": content})
                fp = bs_fb
                if self.config.enable_turn_limits:
                    fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                self._log_rl("user", bs_fb)
                self.messages.append({"role": "user", "content": fp})
                (turn_dir / "bs_feedback.md").write_text(fp, encoding="utf-8")
                continue

            # ── 8. Custom tool registration ──────────────────────────────────────
            reg_action = next(
                (a for a in info_actions if getattr(a, "name", "") == "register_custom_tool"),
                None,
            )
            if reg_action:
                args = reg_action.args
                t_name = args.get("tool_name", "custom_tool")
                console.print(
                    f"\n[bold green]🛠️ Registering custom tool: [white]{t_name}[/white][/bold green]"
                )
                new_schema = {
                    "name": t_name,
                    "description": args.get("description", ""),
                    "properties": args.get("schema_properties", {}),
                    "required": args.get("required_args", []),
                }
                self.dynamic_tools_schema.append(new_schema)
                self.dynamic_tools_mapping[t_name] = args.get("script_path", "")
                self.dynamic_tools_registry[t_name] = new_schema
                self._rebuild_tools_and_prompt(self.config.tool_strategy, self.config.domain)
                reg_fb = (
                    f"✅ Custom tool `{t_name}` registered and available immediately. "
                    "Call it like any native tool."
                )
                self.messages.append({"role": "assistant", "content": content})
                fp = reg_fb
                if self.config.enable_turn_limits:
                    fp = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{fp}"
                self._log_rl("user", reg_fb)
                self.messages.append({"role": "user", "content": fp})
                (turn_dir / "custom_tool_feedback.md").write_text(fp, encoding="utf-8")
                continue

            # ── 9. Execute standard observation tools (concurrently) ─────────────
            _SKIP_NAMES = {
                "brainstorm_solutions", "finish_task", "load_domain_tools",
                "register_custom_tool", "json_parse_error",
            }
            standard_actions = [a for a in info_actions if a.name not in _SKIP_NAMES]

            # Loop-detection: deduplicate repeated identical calls
            if standard_actions:
                deduped, skipped_ok, skipped_err = loop_detector.check_tools(standard_actions)
                if skipped_err:
                    console.print(
                        f"[yellow]⚠️ Blocking retry of failed tools: "
                        f"{[n for n, _ in skipped_err]}[/yellow]"
                    )
                    for tool_name, err_summary in skipped_err:
                        feedback_blocks.append(
                            f"⚠️ **Tool unavailable**: `{tool_name}` previously failed: "
                            f'"{err_summary}". Do NOT retry — proceed with what you have.'
                        )
                if skipped_ok:
                    console.print(
                        f"[yellow]⚠️ Loop detected — skipping repeated: {skipped_ok}[/yellow]"
                    )
                    feedback_blocks.append(
                        f"⚠️ **Loop detected**: You already called {skipped_ok} with identical "
                        "arguments. Results already provided. Synthesize what you know — "
                        "call `write_file` or `finish_task`."
                    )
                standard_actions = deduped

            if standard_actions:
                # Auto-update memory stage before executing tools
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
                loop_detector.record_tools(standard_actions, per_errors)

                # Auto-record tool calls + errors in working memory, and
                # auto-inject key results so get_memory() shows useful content.
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
                            # Persist key tool results into knowledge so the model
                            # sees them when it calls get_memory().
                            result_text = per_results[idx] if idx < len(per_results) else ""
                            _auto_inject_result_to_memory(
                                self.working_memory, a, result_text, display_turn
                            )

                feedback_blocks.append(f"### Tool Execution Results\n{concurrent_results}")

            # ── 10. Execute mutations ────────────────────────────────────────────
            if mutation_actions:
                deduped_mut, skipped_mut = loop_detector.check_mutations(mutation_actions)
                if skipped_mut:
                    feedback_blocks.append(
                        "⚠️ Repeated identical file mutation skipped. "
                        "The same write/edit was already applied. "
                        "Call `finish_task` if the goal is completed."
                    )
                mutation_actions = deduped_mut

            if mutation_actions:
                handler = UniversalToolHandler(
                    config=self.config,
                    turn_dir=turn_dir,
                    allowlist=allowlist,
                    dynamic_tools_mapping=self.dynamic_tools_mapping,
                    dynamic_tools_registry=self.dynamic_tools_registry,
                )
                has_mutation, mutation_res = handler.execute(mutation_actions, content)
                feedback_blocks.append(mutation_res)

                # Stream file_written events to SSE clients
                if self.config.stream_callback:
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
                                    os.path.exists(fpath)
                                    and os.path.getsize(fpath) < 200 * 1024
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

                if has_mutation:
                    written = list(getattr(handler, "written_files", []))
                    if self.working_memory and written:
                        for fpath in written:
                            self.working_memory.record_artifact(fpath, is_output=True)
                        self.working_memory.update_memory({
                            "control": {
                                "stage": "writing",
                                "current_step": f"T{display_turn}: writing {written[0]}",
                            }
                        })
                    loop_detector.record_mutations(mutation_actions)

                    v_feedback = self._verify_mutations(
                        content, mutation_actions, allowlist, turn_dir
                    )
                    if v_feedback:
                        feedback_blocks.append(v_feedback)

                    # Write-only turn → guide the agent toward finishing
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
                            written_file = str(write_only[0].path) if write_only else ""
                            feedback_blocks.append(
                                f"\n✅ **File written**: `{written_file}`\n\n"
                                "**What to do next — choose exactly one:**\n"
                                "1. **Output complete** → call `finish_task` NOW.\n"
                                "2. **Placeholders remain** (`[TODO]`, `# PLACEHOLDER`, etc.) → "
                                "read the next source batch, then `search_and_replace` each marker.\n"
                                "3. **More source content unread** → `update_memory` what you wrote, "
                                "then continue the read→write→update cycle.\n\n"
                                "⚠️ Do NOT read more unless you have a specific placeholder to fill."
                            )
                        # Detect verification outcome from the feedback string.
                        # Only count repeated writes when verification is absent
                        # (non-executable files) — never when it explicitly failed.
                        verify_failed = v_feedback and (
                            "Verification Failed" in v_feedback
                            or "Exit 1" in v_feedback
                            or "Exit 2" in v_feedback
                            or "SyntaxError" in v_feedback
                            or "Traceback" in v_feedback
                            or "Error" in v_feedback
                        )

                        if verify_failed:
                            # Verification is actively failing → reset counter so the
                            # agent keeps trying to fix the code, never exits early.
                            last_write_only_signature = None
                            write_only_repeat_count = 0
                        elif len(set(sig)) == 1:
                            if sig == last_write_only_signature:
                                write_only_repeat_count += 1
                            else:
                                last_write_only_signature = sig
                                write_only_repeat_count = 1
                            if write_only_repeat_count >= 2:
                                summary = f"Completed task writing output file: {sig[0]}"
                                self._save_trajectory_to_disk(task_idx, reward=1.0)
                                return True, summary
                        else:
                            last_write_only_signature = None
                            write_only_repeat_count = 0
                    else:
                        last_write_only_signature = None
                        write_only_repeat_count = 0

            # ── 11. Periodic memory compaction ───────────────────────────────────
            if self.working_memory and current_turn > 0 and current_turn % 5 == 0:
                self.working_memory.compact_memory()
                console.print("[dim]🧠 Working memory compacted.[/dim]")

            # ── 12. Build feedback message ────────────────────────────────────────
            combined_feedback = "\n\n".join(feedback_blocks)

            if not mutation_actions and standard_actions:
                cur_out = (
                    self.working_memory.get_memory("artifacts").get("output_files", [])
                    if self.working_memory else []
                )
                if cur_out:
                    combined_feedback += (
                        f"\nResults above from your read. "
                        f"Output file exists: `{cur_out[0]}`. "
                        "If complete, call `finish_task` now. "
                        "Only continue reading if you have a specific placeholder to fill."
                    )
                else:
                    combined_feedback += "\nAnalyze the results and continue."

            if not combined_feedback.strip():
                combined_feedback = (
                    "No new actions were executed. "
                    "If the output file is already correct, call `finish_task` now."
                )

            # ── 13. Write-pressure injection (context nearing 60%) ───────────────
            _ctx_pct = 100 - int(ctx_headroom * 100 / max_ctx)
            if ctx_headroom < max_ctx * 0.40 and not _write_pressure_injected:
                _write_pressure_injected = True
                out_files = (
                    self.working_memory.get_memory("artifacts").get("output_files", [])
                    if self.working_memory else []
                )
                if not out_files:
                    combined_feedback += (
                        f"\n\n📋 **LONG-TASK STRATEGY ({_ctx_pct}% context used — switch now)**\n\n"
                        "You have read enough. **Stop reading. Create the output file NOW.**\n\n"
                        "**Cycle: Read batch → Write/update → Track → Repeat**\n"
                        "1. `write_file` with what you know; use `[PENDING: <section>]` markers.\n"
                        "2. Read next batch → `search_and_replace` the marker.\n"
                        "3. `update_memory` progress after each fill.\n"
                        "4. Repeat until no markers remain.\n"
                        "5. `finish_task` — do not keep reading once complete.\n\n"
                        "⚡ **Act now**: call `write_file` to create the initial output."
                    )
                else:
                    combined_feedback += (
                        f"\n\n📋 **CONTINUE LONG-TASK CYCLE ({_ctx_pct}% context used)**\n"
                        f"Output file: `{out_files[0]}`.\n"
                        "- Markers remain → read next batch, `search_and_replace` each one.\n"
                        "- Output complete → `finish_task` NOW.\n"
                        "Check `progress.completed_steps` in memory for what is already done."
                    )
                console.print(
                    f"[bold yellow]⚠ Write-pressure injected ({_ctx_pct}% context used).[/bold yellow]"
                )

            # ── 14. Prepend memory snapshot ──────────────────────────────────────
            if self.working_memory:
                self.working_memory.save_memory()
                (turn_dir / "working_memory_snapshot.json").write_text(
                    json.dumps(self.working_memory.get_memory(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                mem_snap = self.working_memory.to_context_string()
                combined_feedback = f"[Working Memory Snapshot]\n{mem_snap}\n\n" + combined_feedback

            # ── 15. Append feedback to message history ────────────────────────────
            if combined_feedback:
                (turn_dir / "tool_execution_results.md").write_text(
                    combined_feedback, encoding="utf-8"
                )

            self.messages.append({"role": "assistant", "content": content})
            feedback_prompt = combined_feedback
            if self.config.enable_turn_limits:
                feedback_prompt = (
                    f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n"
                    + feedback_prompt
                )
            self._log_rl("user", combined_feedback)
            self.messages.append({"role": "user", "content": feedback_prompt})
            (turn_dir / "user_feedback.md").write_text(feedback_prompt, encoding="utf-8")

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
                        default="hybrid")
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
                        default="sliding_window")
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

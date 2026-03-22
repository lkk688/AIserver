"""
working_memory.py — Persistent long-task memory manager for autonomous agents.

Provides two-layer storage:
  1. In-memory runtime dict for fast reads/writes
  2. Persistent JSON file (saved to the current session folder) for recovery

Designed to work across task types:
  - Long document summarization
  - Repository analysis
  - Web research
  - Coding / debugging
"""

from __future__ import annotations

import json
import datetime
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ──────────────────────────────────────────────────────────────
# Schema constants
# ──────────────────────────────────────────────────────────────

VALID_SECTIONS = frozenset(
    ["control", "task", "knowledge", "progress", "actions", "artifacts"]
)

MAX_RECENT_ACTIONS = 20
MAX_RECENT_ACTION_CHARS = 120  # truncation per entry


# ──────────────────────────────────────────────────────────────
# Default memory factory
# ──────────────────────────────────────────────────────────────

def _ts() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_default_memory(
    goal: str = "",
    inputs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a freshly initialised memory dict with sensible defaults."""
    now = _ts()
    return {
        # ── Orchestration control ─────────────────────────────
        "control": {
            "mode": "general",        # summarize | research | coding | repo_analysis | general
            "stage": "planning",      # planning | gathering | synthesizing | writing | done
            "status": "active",       # active | paused | done | failed
            "can_replan": True,
            "current_step": "initializing",  # human-readable current focus
        },
        # ── Task definition ───────────────────────────────────
        "task": {
            "goal": goal,
            "inputs": list(inputs or []),
            "constraints": [],
        },
        # ── Compressed knowledge ──────────────────────────────
        "knowledge": {
            "key_facts": [],           # list[str] — important discovered facts
            "summaries": {},           # {section_id | topic: compressed_text}
            "report_slots": {},        # {slot_name: text} for structured outputs
            "notes": [],               # free-form scratch notes
        },
        # ── Task progress ─────────────────────────────────────
        "progress": {
            "completed_steps": [],
            "pending_steps": [],
            "open_loops": [],          # unresolved questions / items needing follow-up
        },
        # ── Action history ────────────────────────────────────
        "actions": {
            "recent_actions": [],      # short descriptions of last N actions
            "failed_attempts": [],     # what was tried and failed
        },
        # ── Produced artifacts ────────────────────────────────
        "artifacts": {
            "output_files": [],
            "scratch_files": [],
            "timestamps": {
                "created_at": now,
                "last_updated": now,
            },
        },
    }


# ──────────────────────────────────────────────────────────────
# Deep-merge helper  (PATCH semantics)
# ──────────────────────────────────────────────────────────────

def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *patch* into a copy of *base*.

    Rules:
      - dict + dict   → recursive merge
      - list + list   → append new items, preserving order, skipping duplicates
      - anything else → patch value overwrites base value
    """
    result = deepcopy(base)
    for key, pval in patch.items():
        bval = result.get(key)
        if isinstance(bval, dict) and isinstance(pval, dict):
            result[key] = _deep_merge(bval, pval)
        elif isinstance(bval, list) and isinstance(pval, list):
            seen: set = set()
            merged: List[Any] = []
            for item in bval + pval:
                # Use JSON-stable key for dedup; works for dicts and primitives
                dedup_key = (
                    json.dumps(item, sort_keys=True)
                    if isinstance(item, (dict, list))
                    else str(item)
                )
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    merged.append(item)
            result[key] = merged
        else:
            result[key] = deepcopy(pval)
    return result


# ──────────────────────────────────────────────────────────────
# Schema validation / coercion
# ──────────────────────────────────────────────────────────────

def _validate_memory(mem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and coerce *mem* to match the expected schema.

    - Missing top-level sections are filled with defaults.
    - Missing sub-keys within each section are filled with defaults.
    - Extra keys are preserved for forward compatibility.
    """
    default = _make_default_memory()
    result = deepcopy(mem) if isinstance(mem, dict) else {}
    for section in VALID_SECTIONS:
        if section not in result or not isinstance(result[section], dict):
            result[section] = deepcopy(default[section])
        else:
            for k, v in default[section].items():
                if k not in result[section]:
                    result[section][k] = deepcopy(v)
    return result


# ──────────────────────────────────────────────────────────────
# WorkingMemory class
# ──────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Long-task memory manager for autonomous agents.

    Two-layer storage:
      1. In-memory ``_state`` dict for fast reads/writes.
      2. Persistent JSON file at ``<session_dir>/working_memory.json``
         (or custom filename) for crash recovery and session continuation.

    Intended usage
    --------------
    The *orchestration layer* auto-updates low-level fields (recent_actions,
    artifact paths, timestamps) so the model only needs to patch compressed,
    knowledge-level information at important checkpoints via ``update_memory``.

    The two model-facing tools are ``get_memory`` and ``update_memory``.
    All other methods are called by the agent runtime.
    """

    def __init__(
        self,
        session_dir: Path,
        goal: str = "",
        inputs: Optional[List[str]] = None,
        filename: str = "working_memory.json",
    ) -> None:
        """
        Initialise the memory manager.

        Loads an existing file from *session_dir* if present; otherwise creates
        a fresh default state.

        Args:
            session_dir: Directory where the JSON file will be saved.
            goal:        High-level task goal (written to ``task.goal``).
            inputs:      List of input file paths / resource identifiers.
            filename:    Name of the persistent JSON file.
        """
        self._session_dir = Path(session_dir)
        self._save_path = self._session_dir / filename
        self._state: Dict[str, Any] = {}

        if self._save_path.exists():
            self._state = self.load_memory(self._save_path)
            # Update goal / inputs if a new one was provided
            if goal and not self._state["task"]["goal"]:
                self._state["task"]["goal"] = goal
            if inputs:
                for inp in inputs:
                    if inp not in self._state["task"]["inputs"]:
                        self._state["task"]["inputs"].append(inp)
        else:
            self._state = _make_default_memory(goal=goal, inputs=inputs)

        # Always ensure schema is complete after load / init
        self._state = _validate_memory(self._state)
        self.save_memory()

    # ──────────────────────────────────────────────────────────
    # Model-facing API (these two are exposed as tools)
    # ──────────────────────────────────────────────────────────

    def get_memory(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve the current memory state (model-facing tool).

        Args:
            section: One of ``control``, ``task``, ``knowledge``, ``progress``,
                     ``actions``, ``artifacts``. If *None*, returns the full dict.

        Returns:
            A deep copy of the requested section (or full memory dict).

        Raises:
            ValueError: If *section* is not a recognised section name.
        """
        if section is None:
            return deepcopy(self._state)
        section = section.strip().lower()
        if section not in VALID_SECTIONS:
            raise ValueError(
                f"Unknown section '{section}'. "
                f"Valid sections: {sorted(VALID_SECTIONS)}"
            )
        return deepcopy(self._state[section])

    def update_memory(self, patch: Dict[str, Any]) -> None:
        """
        Apply a PATCH-style update to memory (model-facing tool).

        Merges *patch* recursively into the current state:
          - dicts are merged depth-first
          - list values are appended (duplicates skipped)
          - scalar values overwrite

        Only top-level keys that match a valid section name are accepted;
        unrecognised keys are silently ignored for safety.

        ``artifacts.timestamps.last_updated`` is always refreshed.

        Args:
            patch: Partial memory dict. Example::

                {
                    "control": {"current_step": "reading section 3"},
                    "progress": {
                        "completed_steps": ["Read intro"],
                        "pending_steps": ["Read section 2", "Write summary"],
                    },
                    "knowledge": {
                        "key_facts": ["Model uses 9B parameters"],
                        "summaries": {"sec_1": "Intro covers background..."},
                    },
                }

        Raises:
            TypeError: If *patch* is not a dict.
        """
        if not isinstance(patch, dict):
            raise TypeError(f"patch must be a dict, got {type(patch).__name__}")
        # Silently drop unknown top-level keys
        safe_patch = {k: v for k, v in patch.items() if k in VALID_SECTIONS}
        self._state = _deep_merge(self._state, safe_patch)
        self._state["artifacts"]["timestamps"]["last_updated"] = _ts()

    # ──────────────────────────────────────────────────────────
    # Orchestration-layer API (called by the agent runtime)
    # ──────────────────────────────────────────────────────────

    def record_action(self, description: str) -> None:
        """
        Append a short action description to ``actions.recent_actions``.

        Called automatically by the orchestration layer after each tool
        execution. The list is capped at ``MAX_RECENT_ACTIONS`` entries.

        Args:
            description: Short human-readable summary, e.g.
                         ``"Turn 3: read_document_section(sec_2_1)"``.
        """
        entry = description[:MAX_RECENT_ACTION_CHARS]
        recent: List[str] = self._state["actions"]["recent_actions"]
        recent.append(entry)
        if len(recent) > MAX_RECENT_ACTIONS:
            self._state["actions"]["recent_actions"] = recent[-MAX_RECENT_ACTIONS:]
        self._state["artifacts"]["timestamps"]["last_updated"] = _ts()

    def record_artifact(self, file_path: str, is_output: bool = True) -> None:
        """
        Register a written file in artifacts.

        Called automatically by the orchestration layer when a file mutation
        (write_file / search_and_replace) succeeds.

        Args:
            file_path: Path of the written file.
            is_output: ``True`` for final output files,
                       ``False`` for scratch / temporary files.
        """
        key = "output_files" if is_output else "scratch_files"
        files: List[str] = self._state["artifacts"][key]
        if file_path not in files:
            files.append(file_path)
        self._state["artifacts"]["timestamps"]["last_updated"] = _ts()

    def record_failure(self, description: str) -> None:
        """
        Append a failed-attempt note to ``actions.failed_attempts``.

        Args:
            description: Short description of what was tried and why it failed.
        """
        entry = description[:MAX_RECENT_ACTION_CHARS]
        self._state["actions"]["failed_attempts"].append(entry)
        self._state["artifacts"]["timestamps"]["last_updated"] = _ts()

    # ──────────────────────────────────────────────────────────
    # Memory compaction
    # ──────────────────────────────────────────────────────────

    def compact_memory(
        self,
        max_facts: int = 30,
        max_actions: int = 10,
        max_notes: int = 20,
        compress_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Compact memory by trimming low-value history.

        What is trimmed:
          - ``knowledge.key_facts`` — keeps the last *max_facts* entries.
          - ``knowledge.notes``     — keeps the last *max_notes* entries.
          - ``actions.recent_actions`` — keeps the last *max_actions* entries.

        What is **always preserved**:
          - ``knowledge.summaries`` and ``knowledge.report_slots``
          - ``progress.open_loops`` (unresolved items)
          - ``progress.completed_steps`` and ``pending_steps``
          - All artifact paths and timestamps

        Args:
            max_facts:    Maximum number of ``key_facts`` to keep.
            max_actions:  Maximum number of ``recent_actions`` to keep.
            max_notes:    Maximum number of ``notes`` to keep.
            compress_fn:  Optional callable ``(text: str) -> str`` that
                          LLM-compresses excess entries into a single summary
                          instead of discarding them.
        """
        k = self._state["knowledge"]
        acts = self._state["actions"]

        def _trim_list(lst: List[str], limit: int, label: str) -> List[str]:
            if len(lst) <= limit:
                return lst
            excess = lst[:-limit]
            trimmed = lst[-limit:]
            if compress_fn and excess:
                compressed = compress_fn(
                    f"{label}:\n" + "\n".join(f"- {e}" for e in excess)
                )
                return [f"[compressed] {compressed}"] + trimmed
            return trimmed

        k["key_facts"] = _trim_list(k["key_facts"], max_facts, "Key facts to compress")
        k["notes"] = _trim_list(k["notes"], max_notes, "Notes to compress")
        acts["recent_actions"] = acts["recent_actions"][-max_actions:]
        self._state["artifacts"]["timestamps"]["last_updated"] = _ts()

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save_memory(self, path: Optional[Path] = None) -> Path:
        """
        Persist the current memory state to a JSON file.

        Uses an atomic write (temp-file + rename) on POSIX systems to
        reduce the risk of corruption on crash.

        Args:
            path: Override save path. Defaults to
                  ``<session_dir>/working_memory.json``.

        Returns:
            The actual path that was written.
        """
        target = Path(path) if path else self._save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2, ensure_ascii=False)
        tmp = target.with_suffix(".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)   # atomic rename on POSIX / near-atomic on Windows
        except Exception:
            target.write_text(payload, encoding="utf-8")
        return target

    def load_memory(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load and validate memory from a JSON file.

        Corruption-resistant: if the file is missing, empty, or invalid JSON,
        a warning is emitted and a fresh default memory dict is returned.

        Args:
            path: File to load. Defaults to
                  ``<session_dir>/working_memory.json``.

        Returns:
            A schema-validated memory dict.
        """
        target = Path(path) if path else self._save_path
        if not target.exists():
            return _make_default_memory()
        try:
            raw = target.read_text(encoding="utf-8").strip()
            if not raw:
                raise ValueError("File is empty")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Root must be a JSON object")
            return _validate_memory(data)
        except Exception as exc:
            warnings.warn(
                f"WorkingMemory: corrupt or unreadable file at '{target}' "
                f"({exc}). Starting with a fresh memory state.",
                stacklevel=2,
            )
            return _make_default_memory()

    # ──────────────────────────────────────────────────────────
    # Context rendering (for LLM injection)
    # ──────────────────────────────────────────────────────────

    def to_context_string(self, sections: Optional[List[str]] = None) -> str:
        """
        Render memory as a compact Markdown snapshot for LLM context injection.

        Includes only the most recent / most relevant entries to keep the
        injected text short.

        Args:
            sections: Subset of section names to include.
                      ``None`` includes all sections.

        Returns:
            Markdown-formatted string summarising the current memory state.
        """
        mem = self._state
        parts: List[str] = []

        def _want(name: str) -> bool:
            return sections is None or name in sections

        if _want("control"):
            c = mem["control"]
            parts.append(
                f"**Stage**: {c.get('stage', '?')}  |  "
                f"**Status**: {c.get('status', '?')}  |  "
                f"**Now**: {c.get('current_step', '?')}"
            )

        if _want("progress"):
            p = mem["progress"]
            done = p.get("completed_steps", [])
            pending = p.get("pending_steps", [])
            loops = p.get("open_loops", [])
            if done:
                parts.append("✅ **Done** (last 5): " + " → ".join(done[-5:]))
            if pending:
                parts.append("⏳ **Pending** (first 5): " + " | ".join(pending[:5]))
            if loops:
                parts.append("🔄 **Open loops**: " + " | ".join(loops[:3]))

        if _want("knowledge"):
            k = mem["knowledge"]
            facts = k.get("key_facts", [])
            summaries = k.get("summaries", {})
            if facts:
                parts.append(
                    "📌 **Key facts** (last 5):\n"
                    + "\n".join(f"  - {f}" for f in facts[-5:])
                )
            if summaries:
                parts.append(f"📄 **Summaries on record**: {len(summaries)} section(s)")

        if _want("actions"):
            recent = mem["actions"].get("recent_actions", [])
            if recent:
                parts.append(
                    "🛠 **Recent actions** (last 3):\n"
                    + "\n".join(f"  - {a}" for a in recent[-3:])
                )

        if _want("artifacts"):
            out = mem["artifacts"].get("output_files", [])
            if out:
                parts.append("📁 **Output files**: " + ", ".join(out))

        return "\n".join(parts) if parts else "(memory is empty)"

    def __repr__(self) -> str:
        c = self._state["control"]
        return (
            f"WorkingMemory("
            f"stage={c.get('stage')!r}, "
            f"status={c.get('status')!r}, "
            f"step={c.get('current_step')!r})"
        )


# ──────────────────────────────────────────────────────────────
# Tool handler functions  (called by agent orchestration)
# ──────────────────────────────────────────────────────────────

def handle_get_memory(memory: WorkingMemory, args: Dict[str, Any]) -> str:
    """
    Execute the ``get_memory`` tool call and return a formatted result string.

    Args:
        memory: The active WorkingMemory instance.
        args:   Tool arguments dict from the LLM, e.g. ``{"section": "progress"}``.

    Returns:
        Markdown-formatted tool result.
    """
    section: Optional[str] = args.get("section") or None
    try:
        data = memory.get_memory(section)
        label = section or "full"
        return (
            f"### Memory ({label})\n"
            f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
        )
    except ValueError as exc:
        return f"### Memory Error\n❌ {exc}"


def handle_update_memory(memory: WorkingMemory, args: Dict[str, Any]) -> str:
    """
    Execute the ``update_memory`` tool call and return a confirmation string.

    Args:
        memory: The active WorkingMemory instance.
        args:   Tool arguments dict from the LLM, e.g.
                ``{"patch": {"control": {"current_step": "..."}}}``

    Returns:
        Markdown-formatted confirmation or error message.
    """
    patch = args.get("patch", {})
    if not isinstance(patch, dict):
        return "### Memory Update\n❌ `patch` must be a JSON object."
    if not patch:
        return "### Memory Update\n⚠️ Empty patch — nothing to update."
    try:
        memory.update_memory(patch)
        keys = ", ".join(patch.keys())
        return f"### Memory Update\n✅ Updated sections: `{keys}`"
    except TypeError as exc:
        return f"### Memory Update\n❌ {exc}"

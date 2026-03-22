#!/usr/bin/env python3
"""
batch_eval_v2.py — Batch evaluation harness for UniversalAgent v2.

Runs ML coding tasks (CodeAgent/ml_tasks.json) through agent_main_v2,
tracks success rate, turns-to-completion, and timing, then generates
publication-quality figures.

Usage:
    cd /Developer/AIserver
    python3 BatchAgent/batch_eval_v2.py                                 # run all tasks
    python3 BatchAgent/batch_eval_v2.py --task-id linreg_lvl1_raw_tensors
    python3 BatchAgent/batch_eval_v2.py --start-from 5 --max-tasks 10
    python3 BatchAgent/batch_eval_v2.py --redo-failed
    python3 BatchAgent/batch_eval_v2.py --plot-only                     # re-plot only
    python3 BatchAgent/batch_eval_v2.py --tool-strategy native_all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BatchAgent.agent_services_fastapi_v2 import AgentService


# =============================================================================
# 1. Task loading and goal construction
# =============================================================================

def load_tasks(tasks_json: Path) -> Tuple[List[dict], dict]:
    data = json.loads(tasks_json.read_text())
    return data["tasks"], data.get("interface_protocols", {})


def build_goal(task: dict, protocol: dict) -> str:
    """Construct a clear, self-contained goal for the coding agent."""
    reqs = task.get("requirements", {})
    req_lines = "\n".join(f"  - {k}: {v}" for k, v in reqs.items())

    eval_rules = "\n".join(
        f"  {i+1}. {r}" for i, r in enumerate(protocol.get("evaluation_rules", []))
    )

    return (
        f"## OUTPUT FILE\n"
        f"Write the implementation to **`task.py`** in the CURRENT working directory.\n"
        f"Do NOT create subdirectories. Do NOT look for existing files first — just write `task.py`.\n\n"
        f"## Algorithm\n{task['algorithm']}\n\n"
        f"## Description\n{task['description']}\n\n"
        f"## Requirements\n{req_lines}\n\n"
        f"## Mandatory Structure\n"
        f"The file must contain these functions: "
        f"get_task_metadata, set_seed, get_device, make_dataloaders, "
        f"build_model, train, evaluate, predict, save_artifacts.\n\n"
        f"## The `if __name__ == '__main__':` block MUST:\n"
        f"  1. Call set_seed, make_dataloaders, build_model, train, evaluate.\n"
        f"  2. Evaluate on BOTH train and validation splits.\n"
        f"  3. Print all metrics clearly (MSE, R2, accuracy, etc.).\n"
        f"  4. Assert quality thresholds — exit non-zero on failure.\n"
        f"  5. Print 'PASS' on success, 'FAIL' on failure.\n\n"
        f"## Evaluation Rules\n{eval_rules}\n\n"
        f"CRITICAL: Only create `task.py` in the current directory (`.`). "
        f"No README, no test_task.py, no subdirectories.\n"
        f"{protocol.get('prompt_instructions', '')}"
    )


# =============================================================================
# 2. Single-task runner
# =============================================================================

def _count_turns(session_dir: Path) -> int:
    """Count agent turns from session subdirectory 0000/0001/..."""
    if not session_dir.exists():
        return 0
    return sum(1 for p in session_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d{4}", p.name))


def _verify_task_file(task_file: Path, timeout: int = 600) -> Tuple[bool, str]:
    """Run task.py and check exit code. Returns (passed, output_snippet)."""
    if not task_file.exists():
        return False, "task.py not found"
    try:
        r = subprocess.run(
            [sys.executable, str(task_file)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(task_file.parent),
        )
        snippet = (r.stdout + r.stderr)[-600:]
        return r.returncode == 0, snippet
    except subprocess.TimeoutExpired:
        return False, "verification timeout"
    except Exception as e:
        return False, str(e)


async def run_single_task(
    task: dict,
    protocol: dict,
    args: argparse.Namespace,
) -> dict:
    """
    Run one ML task through UniversalAgent v2 and return a result dict with:
      - success (agent + verification)
      - n_turns (ReAct rounds used)
      - duration_sec
      - task_file_exists
      - verification_passed
      - error snippet
    """
    task_id = task["id"]
    task_dir = (Path(args.output_dir) / "tasks" / task_id).resolve()
    task_file = task_dir / "task.py"

    # Fresh workspace for each run
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "task_id": task_id,
        "algorithm": task["algorithm"],
        "series": task["series"],
        "level": task["level"],
        "status": "unknown",
        "n_turns": 0,
        "start_time": datetime.now().isoformat(),
        "duration_sec": 0.0,
        "task_file_exists": False,
        "verification_passed": False,
        "agent_success": False,
        "error": None,
        "output_snippet": "",
        "session_dir": None,
    }

    goal = build_goal(task, protocol)
    service = AgentService(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        tool_strategy=args.tool_strategy,
        output_dir=str(task_dir),
        verbose=args.verbose,
        max_turns=args.max_turns,
        enable_turn_limits=True,
        backend=args.backend,
        enable_thinking=True,
        temperature=0.1,
        enable_memory=False,   # pure coding task — no persistent memory needed
        enable_document=False,
        enable_web=True,
        enable_mutation=True,
        enable_code_tools=True,
        enable_bash=True,
        enable_meta=False,
    )

    t0 = time.time()
    try:
        print(f"\n{'='*68}")
        print(f"  Task : {task_id}")
        print(f"  Level: {task['level']}  |  Series: {task['series']}")
        print(f"{'='*68}\n")

        agent_success, session_dir, workspace_dir, final_result = await service.run(
            goal=goal,
            allowlist=[str(task_file)],
            continue_workspace_dir=str(task_dir),
        )

        result["duration_sec"] = round(time.time() - t0, 1)
        result["agent_success"] = agent_success
        result["session_dir"] = str(session_dir)
        result["n_turns"] = _count_turns(session_dir)
        result["task_file_exists"] = task_file.exists()
        result["output_snippet"] = str(final_result)[-500:]

        if task_file.exists():
            print(f"  [verify] Running {task_file.name}...")
            verified, snippet = _verify_task_file(task_file)
            result["verification_passed"] = verified
            result["output_snippet"] = snippet
            if verified:
                result["status"] = "success"
                # Keep only task.py, clean intermediates
                for item in task_dir.iterdir():
                    if item.name not in ("task.py", ".agent"):
                        try:
                            shutil.rmtree(item) if item.is_dir() else item.unlink()
                        except Exception:
                            pass
            else:
                result["status"] = "verify_failed"
                result["error"] = snippet[-300:]
        else:
            result["status"] = "no_file"
            result["error"] = str(final_result)[-300:]

    except Exception as exc:
        result["status"] = "exception"
        result["error"] = str(exc)[:400]
        result["duration_sec"] = round(time.time() - t0, 1)

    icon = "✓" if result["status"] == "success" else "✗"
    print(f"\n  {icon} {task_id}: {result['status']}  "
          f"turns={result['n_turns']}  time={result['duration_sec']}s")
    return result


# =============================================================================
# 3. Results persistence
# =============================================================================

def save_results(results: List[dict], path: Path, model: str) -> None:
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "avg_turns_success": _safe_mean(
            [r["n_turns"] for r in results if r["status"] == "success"]
        ),
        "avg_turns_failed": _safe_mean(
            [r["n_turns"] for r in results if r["status"] != "success"]
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "tasks": results}, indent=2), encoding="utf-8")
    print(f"\n  Results saved → {path}")


def _safe_mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0.0


# =============================================================================
# 4. Figure generation (publication quality)
# =============================================================================

def generate_figures(results_path: Path, out_dir: Path) -> None:
    """
    Generate 4 research-paper-ready figures from a results JSON file.

    Figure layout
    ─────────────
    Fig 1 — Overview Dashboard      (2×2 panel)
    Fig 2 — Success by Category      (horizontal bar)
    Fig 3 — Turn Efficiency by Level (box + swarm)
    Fig 4 — Turns × Time Scatter     (colored by outcome)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("[WARNING] matplotlib/numpy not installed — skipping figure generation.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text())
    tasks = data["tasks"]
    summary = data["summary"]

    if not tasks:
        print("[WARNING] No task results found — skipping figures.")
        return

    # ── Style ────────────────────────────────────────────────────────────────
    PAPER_W = 7.0   # inches (one-column IEEE)
    PAPER_H = 5.5

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    C_SUCCESS = "#2E86AB"   # blue
    C_FAIL    = "#E84855"   # red
    C_NEUTRAL = "#A8DADC"   # light blue
    C_ACCENT  = "#F4A261"   # orange

    levels = sorted(set(r["level"] for r in tasks))
    series_list = sorted(set(r["series"] for r in tasks))

    def _success_rate(subset):
        if not subset:
            return 0.0
        return sum(1 for r in subset if r["status"] == "success") / len(subset) * 100

    # ── Figure 1: Overview Dashboard (2×2) ───────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(PAPER_W * 1.6, PAPER_H * 1.4))
    fig1.suptitle(
        f"UniversalAgent v2 — ML Task Benchmark\n"
        f"Model: {summary.get('model','?')}  |  "
        f"Overall Success: {summary['success']}/{summary['total']} "
        f"({summary['success']/max(summary['total'],1)*100:.1f}%)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # 1a: Success rate by difficulty level
    ax = axes[0, 0]
    lvl_rates = [_success_rate([r for r in tasks if r["level"] == lv]) for lv in levels]
    lvl_counts = [sum(1 for r in tasks if r["level"] == lv) for lv in levels]
    bars = ax.bar([f"Level {l}" for l in levels], lvl_rates,
                  color=[C_SUCCESS if v >= 50 else C_FAIL for v in lvl_rates],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for bar, rate, cnt in zip(bars, lvl_rates, lvl_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%\n(n={cnt})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("(a) Success Rate by Difficulty Level")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8)

    # 1b: Turn count distribution (histogram)
    ax = axes[0, 1]
    all_turns = [r["n_turns"] for r in tasks]
    suc_turns = [r["n_turns"] for r in tasks if r["status"] == "success"]
    fail_turns = [r["n_turns"] for r in tasks if r["status"] != "success"]
    max_t = max(all_turns) if all_turns else 1
    bins = range(0, max_t + 2)
    ax.hist(fail_turns, bins=bins, color=C_FAIL, alpha=0.75, label="Failed", zorder=3)
    ax.hist(suc_turns, bins=bins, color=C_SUCCESS, alpha=0.75, label="Success", zorder=4)
    ax.set_xlabel("Turns Used")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("(b) Turns-to-Completion Distribution")
    ax.legend(loc="upper right")

    # 1c: Outcome breakdown (stacked bar by level)
    ax = axes[1, 0]
    status_map = {
        "success": C_SUCCESS, "verify_failed": C_ACCENT,
        "no_file": C_FAIL, "exception": "#6c757d", "timeout": "#343a40",
    }
    all_statuses = sorted(set(r["status"] for r in tasks))
    lvl_labels = [f"L{l}" for l in levels]
    bottom = np.zeros(len(levels))
    for stat in all_statuses:
        counts = [sum(1 for r in tasks if r["level"] == lv and r["status"] == stat)
                  for lv in levels]
        color = status_map.get(stat, "#aaa")
        ax.bar(lvl_labels, counts, bottom=bottom, color=color,
               label=stat.replace("_", " "), edgecolor="white", linewidth=0.5)
        bottom += np.array(counts, dtype=float)
    ax.set_ylabel("Number of Tasks")
    ax.set_title("(c) Outcome Breakdown by Level")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.8)

    # 1d: Avg turns by level × outcome
    ax = axes[1, 1]
    w = 0.35
    x = np.arange(len(levels))
    avg_suc = [_safe_mean([r["n_turns"] for r in tasks
                           if r["level"] == lv and r["status"] == "success"])
               for lv in levels]
    avg_fail = [_safe_mean([r["n_turns"] for r in tasks
                            if r["level"] == lv and r["status"] != "success"])
                for lv in levels]
    ax.bar(x - w/2, avg_suc, width=w, color=C_SUCCESS, label="Success", zorder=3)
    ax.bar(x + w/2, avg_fail, width=w, color=C_FAIL, label="Failed", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.set_ylabel("Avg Turns")
    ax.set_title("(d) Avg Turns: Success vs. Failed")
    ax.legend()

    fig1.tight_layout(pad=1.2)
    p1 = out_dir / "fig1_overview.pdf"
    fig1.savefig(p1, bbox_inches="tight")
    fig1.savefig(p1.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"  Saved: {p1}")
    plt.close(fig1)

    # ── Figure 2: Success Rate by ML Category (horizontal bar) ───────────────
    fig2, ax = plt.subplots(figsize=(PAPER_W * 1.3, max(4.0, len(series_list) * 0.45 + 1.2)))
    rates = []
    counts = []
    for s in series_list:
        subset = [r for r in tasks if r["series"] == s]
        rates.append(_success_rate(subset))
        counts.append(len(subset))

    # Sort by success rate descending
    order = sorted(range(len(series_list)), key=lambda i: rates[i], reverse=False)
    sorted_series = [series_list[i] for i in order]
    sorted_rates  = [rates[i] for i in order]
    sorted_counts = [counts[i] for i in order]

    colors = [C_SUCCESS if r >= 50 else (C_ACCENT if r >= 25 else C_FAIL) for r in sorted_rates]
    y_pos = np.arange(len(sorted_series))
    bars = ax.barh(y_pos, sorted_rates, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_series, fontsize=8.5)
    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 115)
    ax.axvline(50, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(
        f"Figure 2 — Per-Category Success Rate\n"
        f"(sorted ascending; n=tasks per category)",
        fontsize=10, fontweight="bold",
    )
    for bar, rate, cnt in zip(bars, sorted_rates, sorted_counts):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.0f}%  (n={cnt})", va="center", fontsize=7.5)

    legend_patches = [
        mpatches.Patch(color=C_SUCCESS, label="≥50%"),
        mpatches.Patch(color=C_ACCENT,  label="25–49%"),
        mpatches.Patch(color=C_FAIL,    label="<25%"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    fig2.tight_layout(pad=1.2)
    p2 = out_dir / "fig2_category_success.pdf"
    fig2.savefig(p2, bbox_inches="tight")
    fig2.savefig(p2.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"  Saved: {p2}")
    plt.close(fig2)

    # ── Figure 3: Turn Efficiency by Level (box plot with jitter) ────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(PAPER_W * 1.4, PAPER_H))
    fig3.suptitle("Figure 3 — Turn Efficiency by Difficulty Level",
                  fontsize=11, fontweight="bold")

    np.random.seed(42)
    for col_idx, (outcome, color, label) in enumerate([
        ("success", C_SUCCESS, "Successful Tasks"),
        ("failed",  C_FAIL,    "Failed Tasks"),
    ]):
        ax3 = axes3[col_idx]
        box_data = []
        tick_labels = []
        for lv in levels:
            if outcome == "success":
                subset = [r["n_turns"] for r in tasks
                          if r["level"] == lv and r["status"] == "success"]
            else:
                subset = [r["n_turns"] for r in tasks
                          if r["level"] == lv and r["status"] != "success"]
            box_data.append(subset if subset else [0])
            tick_labels.append(f"L{lv}\n(n={len(subset)})")

        bp = ax3.boxplot(box_data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=1.5),
                         whiskerprops=dict(linewidth=1.0),
                         capprops=dict(linewidth=1.0),
                         flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Jitter overlay
        for i, d in enumerate(box_data, 1):
            jitter = np.random.uniform(-0.15, 0.15, len(d))
            ax3.scatter(np.full(len(d), i) + jitter, d,
                        color=color, alpha=0.55, s=18, zorder=4)

        ax3.set_xticks(range(1, len(levels) + 1))
        ax3.set_xticklabels(tick_labels)
        ax3.set_ylabel("Turns Used")
        ax3.set_title(f"({chr(97+col_idx)}) {label}")

    fig3.tight_layout(pad=1.2)
    p3 = out_dir / "fig3_turn_efficiency.pdf"
    fig3.savefig(p3, bbox_inches="tight")
    fig3.savefig(p3.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"  Saved: {p3}")
    plt.close(fig3)

    # ── Figure 4: Turns × Time Scatter ───────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(PAPER_W * 1.2, PAPER_H))
    for r in tasks:
        color = C_SUCCESS if r["status"] == "success" else C_FAIL
        marker = "o" if r["status"] == "success" else "x"
        ax4.scatter(r["n_turns"], r["duration_sec"] / 60,
                    color=color, marker=marker, s=45, alpha=0.72, zorder=3)

    # Annotate level centroids
    for lv in levels:
        sub = [r for r in tasks if r["level"] == lv]
        if sub:
            cx = _safe_mean([r["n_turns"] for r in sub])
            cy = _safe_mean([r["duration_sec"] / 60 for r in sub])
            ax4.annotate(
                f"L{lv}", (cx, cy),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color="#333", fontweight="bold",
            )

    success_patch = mpatches.Patch(color=C_SUCCESS, label="Success")
    fail_patch    = mpatches.Patch(color=C_FAIL,    label="Failed")
    ax4.legend(handles=[success_patch, fail_patch], loc="upper left")
    ax4.set_xlabel("Turns Used (ReAct rounds)")
    ax4.set_ylabel("Wall-Clock Time (minutes)")
    ax4.set_title(
        "Figure 4 — Task Completion: Turns vs. Time\n"
        "(L1–L4 centroids annotated)",
        fontsize=10, fontweight="bold",
    )
    fig4.tight_layout(pad=1.2)
    p4 = out_dir / "fig4_turns_vs_time.pdf"
    fig4.savefig(p4, bbox_inches="tight")
    fig4.savefig(p4.with_suffix(".png"), bbox_inches="tight", dpi=180)
    print(f"  Saved: {p4}")
    plt.close(fig4)

    print(f"\n  All figures written to: {out_dir}/")


# =============================================================================
# 5. Main batch loop
# =============================================================================

async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Batch evaluation for UniversalAgent v2 on ML tasks"
    )
    parser.add_argument("--tasks-json", default="CodeAgent/ml_tasks.json")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument("--status-file", default="eval_output/eval_results.json")
    parser.add_argument("--figures-dir", default="eval_output/figures")

    parser.add_argument("--task-id", default=None,
                        help="Run a single task by ID")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N tasks (0-indexed)")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--redo-failed", action="store_true",
                        help="Re-run tasks that previously failed")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip running — regenerate figures from existing results")

    # Agent options
    parser.add_argument("--base-url",
                        default=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"))
    parser.add_argument("--api-key",
                        default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model",
                        default=os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--tool-strategy",
                        choices=["native_all", "hybrid", "text_only"],
                        default="hybrid")
    parser.add_argument("--backend",
                        choices=["vllm", "llama.cpp", "openai", "anthropic"],
                        default="vllm")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    status_file = Path(args.status_file)
    figures_dir = Path(args.figures_dir)

    # ── Plot-only mode ────────────────────────────────────────────────────────
    if args.plot_only:
        if not status_file.exists():
            print(f"Error: {status_file} not found. Run evaluation first.")
            sys.exit(1)
        print(f"\nGenerating figures from {status_file} ...")
        generate_figures(status_file, figures_dir)
        return

    # ── Load tasks ────────────────────────────────────────────────────────────
    tasks_json = Path(args.tasks_json)
    if not tasks_json.exists():
        print(f"Error: {tasks_json} not found.")
        sys.exit(1)
    all_tasks, protocols = load_tasks(tasks_json)

    # ── Load existing results (for resume / redo) ─────────────────────────────
    results: List[dict] = []
    if status_file.exists():
        try:
            existing = json.loads(status_file.read_text())
            results = existing.get("tasks", [])
        except Exception:
            pass

    # ── Filter task list ──────────────────────────────────────────────────────
    if args.task_id:
        tasks = [t for t in all_tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Error: task '{args.task_id}' not found in {tasks_json}")
            sys.exit(1)
        # Remove any existing result for this task so we re-run cleanly
        results = [r for r in results if r["task_id"] != args.task_id]

    elif args.redo_failed:
        target_ids = {r["task_id"] for r in results if r["status"] != "success"}
        # Also include tasks that never ran
        existing_ids = {r["task_id"] for r in results}
        for t in all_tasks:
            if t["id"] not in existing_ids:
                target_ids.add(t["id"])
        if not target_ids:
            print("No failed or unrun tasks found. Nothing to redo.")
            sys.exit(0)
        tasks = [t for t in all_tasks if t["id"] in target_ids]
        results = [r for r in results if r["task_id"] not in target_ids]
        print(f"  Redo mode: {len(tasks)} task(s) to re-run")

    else:
        # Normal run / resume (skip already-completed tasks)
        done_ids = {r["task_id"] for r in results}
        tasks = [t for t in all_tasks if t["id"] not in done_ids]
        tasks = tasks[args.start_from:]
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]

    print(f"\n{'#'*68}")
    print(f"  Batch Eval v2 — {len(tasks)} task(s)")
    print(f"  Model     : {args.model}")
    print(f"  Strategy  : {args.tool_strategy}")
    print(f"  Max turns : {args.max_turns}")
    print(f"  Output    : {args.output_dir}")
    print(f"{'#'*68}")

    # ── Run tasks ─────────────────────────────────────────────────────────────
    for i, task in enumerate(tasks):
        proto_id = task.get("interface_protocol", "pytorch_task_v1")
        protocol = protocols.get(proto_id, {})
        print(f"\n[{i+1}/{len(tasks)}] {task['id']}")
        result = await run_single_task(task, protocol, args)
        results.append(result)
        save_results(results, status_file, args.model)   # incremental save

    # ── Final summary ─────────────────────────────────────────────────────────
    success_count = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    avg_turns = _safe_mean([r["n_turns"] for r in results if r["status"] == "success"])

    print(f"\n{'='*68}")
    print(f"  BATCH COMPLETE")
    print(f"  Success rate : {success_count}/{total} ({success_count/max(total,1)*100:.1f}%)")
    print(f"  Avg turns    : {avg_turns:.1f} (successful tasks)")
    print(f"{'='*68}\n")

    print(f"  {'Task ID':<42} {'Status':<18} {'Turns':>5} {'Time':>7}")
    print(f"  {'-'*42} {'-'*18} {'-'*5} {'-'*7}")
    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {icon} {r['task_id']:<40} {r['status']:<18} "
              f"{r['n_turns']:>5} {r['duration_sec']:>6.1f}s")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\nGenerating figures...")
    generate_figures(status_file, figures_dir)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


# =============================================================================
# Example usage
# =============================================================================
"""
# Run all tasks (resume-safe — skips already-done tasks):
python3 BatchAgent/batch_eval_v2.py

# Run a single task with verbose output:
python3 BatchAgent/batch_eval_v2.py --task-id linreg_lvl1_raw_tensors --verbose

# Retry all failed tasks:
python3 BatchAgent/batch_eval_v2.py --redo-failed

# Run first 10 tasks with native tool calling:
python3 BatchAgent/batch_eval_v2.py --max-tasks 10 --tool-strategy native_all

# Compare a different model:
python3 BatchAgent/batch_eval_v2.py \
    --model qwen3.5-32b \
    --status-file eval_output/eval_results_32b.json \
    --figures-dir eval_output/figures_32b

# Re-plot from saved results (no agent calls):
python3 BatchAgent/batch_eval_v2.py --plot-only

# Custom vLLM endpoint:
python3 BatchAgent/batch_eval_v2.py \
    --base-url http://192.168.1.10:8000/v1 \
    --api-key mykey \
    --model qwen3-coder-7b


python3 BatchAgent/batch_eval_v2.py --tool-strategy native_all --verbose

python3 BatchAgent/batch_eval_v2.py --model qwen-35b --base-url http://127.0.0.1:8000/v1 --api-key "myhpcvllmqwen101" --tool-strategy native_all --verbose
"""

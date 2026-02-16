#!/usr/bin/env python3
"""
batch_coder.py — Run mini_claude_code agent on all tasks in ml_tasks.json sequentially.

Generates code for each ML task, tracks pass/fail status, and saves results
to a JSON status file.

Usage:
    cd /Developer/AIserver
    python3 CodeAgent/batch_coder.py                           # run all tasks
    python3 CodeAgent/batch_coder.py --start-from 3            # skip first 3 tasks
    python3 CodeAgent/batch_coder.py --task-id linreg_lvl1_raw_tensors  # run one task
    python3 CodeAgent/batch_coder.py --status-file results.json  # custom output file
"""

import json
import sys
import os
import time
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


# ---------------------
# Configuration
# ---------------------
TASKS_JSON = Path("CodeAgent/ml_tasks.json")
OUTPUT_DIR = Path("output")
DEFAULT_STATUS_FILE = Path("output/batch_status.json")

# Inherit from env or use defaults
BASE_URL = os.environ.get("VLLM_BASE_URL", "https://w0wqtv67-8000.usw3.devtunnels.ms/v1")
API_KEY = os.environ.get("VLLM_API_KEY", "myhpcvllmqwen")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-Coder-Next-FP8")


def load_tasks(tasks_json: Path) -> dict:
    """Load the full tasks configuration."""
    data = json.loads(tasks_json.read_text())
    return data


def build_goal_and_notes(task: dict, protocol: dict) -> tuple:
    """Build goal and notes strings from a task definition."""
    reqs = task.get("requirements", {})
    req_str = "\n".join(f"- {k.title()}: {v}" for k, v in reqs.items())

    eval_rules = ""
    if "evaluation_rules" in protocol:
        eval_rules = "\n".join(f"- {r}" for r in protocol["evaluation_rules"])

    goal = (
        f"Implement ML Task: {task['algorithm']}\n\n"
        f"Description: {task['description']}\n\n"
        f"Write a SINGLE self-contained Python file (task.py) with these functions:\n"
        f"get_task_metadata, set_seed, get_device, make_dataloaders, build_model, "
        f"train, evaluate, predict, save_artifacts.\n\n"
        f"CRITICAL: The if __name__ == '__main__' block must:\n"
        f"1. Train the model\n"
        f"2. Evaluate on BOTH train and validation splits\n"
        f"3. Print standard metrics (MSE, R2, accuracy, etc. as appropriate)\n"
        f"4. Assert quality thresholds so script exits non-zero on failure\n"
        f"5. Print a clear PASS/FAIL summary\n\n"
        f"Do NOT create separate test files or README. The script IS the test."
    )

    notes = (
        f"Requirements:\n{req_str}\n\n"
        f"Evaluation Rules:\n{eval_rules}\n\n"
        f"IMPORTANT: Only create task.py. No test_task.py, no README.md.\n"
        f"Protocol: {protocol.get('prompt_instructions', '')}"
    )

    return goal, notes


def run_single_task(task: dict, protocol: dict, output_dir: Path) -> dict:
    """
    Run the mini_claude_code agent for a single task.
    Returns a status dict with success/failure, timing, and details.
    """
    task_id = task["id"]
    task_dir = output_dir / "tasks" / task_id
    task_file = task_dir / "task.py"

    # Clean previous output for this task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    goal, notes = build_goal_and_notes(task, protocol)

    # Build the command
    cmd = [
        sys.executable, "-m", "CodeAgent.mini_claude_code",
        "--goal", goal,
        "--notes", notes,
        "--allowlist", str(task_file),
        "--yes",
        "--base-url", BASE_URL,
        "--api-key", API_KEY,
        "--model", MODEL,
    ]

    start_time = time.time()
    result = {
        "task_id": task_id,
        "algorithm": task["algorithm"],
        "series": task["series"],
        "level": task["level"],
        "status": "unknown",
        "start_time": datetime.now().isoformat(),
        "duration_sec": 0,
        "task_file_exists": False,
        "verification_passed": False,
        "error": None,
        "output_snippet": "",
    }

    try:
        print(f"\n{'='*70}")
        print(f"  Running: {task_id} — {task['algorithm']}")
        print(f"  Level: {task['level']} | Series: {task['series']}")
        print(f"{'='*70}\n")

        # Run the agent as a subprocess
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per task
            cwd=str(Path.cwd()),
        )

        elapsed = time.time() - start_time
        result["duration_sec"] = round(elapsed, 1)
        result["output_snippet"] = (proc.stdout or "")[-1000:]

        # Check if task.py was created
        result["task_file_exists"] = task_file.exists()

        if proc.returncode == 0 and task_file.exists():
            # Try to verify the generated file
            verify_result = subprocess.run(
                [sys.executable, str(task_file)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path.cwd()),
            )
            result["verification_passed"] = verify_result.returncode == 0
            if verify_result.returncode == 0:
                result["status"] = "success"
                result["output_snippet"] = (verify_result.stdout or "")[-500:]
            else:
                result["status"] = "verify_failed"
                result["error"] = (verify_result.stderr or verify_result.stdout or "")[-500:]
        elif task_file.exists():
            result["status"] = "agent_failed_file_exists"
            result["error"] = (proc.stderr or "")[-500:]
        else:
            result["status"] = "agent_failed_no_file"
            result["error"] = (proc.stderr or "")[-500:]

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Task exceeded 10 minute timeout"
        result["duration_sec"] = round(time.time() - start_time, 1)
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        result["duration_sec"] = round(time.time() - start_time, 1)

    status_icon = "✓" if result["status"] == "success" else "✗"
    print(f"\n  {status_icon} {task_id}: {result['status']} ({result['duration_sec']}s)")

    return result


def save_status(results: list, status_file: Path):
    """Save batch results to JSON."""
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] != "success"),
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
    }

    output = {
        "summary": summary,
        "tasks": results,
    }

    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nStatus saved to: {status_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch run ML tasks through mini_claude_code")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip first N tasks (0-indexed)")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum number of tasks to run")
    parser.add_argument("--task-id", type=str, default=None,
                        help="Run only this specific task ID")
    parser.add_argument("--status-file", type=str, default=str(DEFAULT_STATUS_FILE),
                        help="Path to save status JSON")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Base output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    status_file = Path(args.status_file)

    # Load tasks
    data = load_tasks(TASKS_JSON)
    tasks = data["tasks"]
    protocols = data.get("interface_protocols", {})

    # Filter tasks
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]
        if not tasks:
            print(f"Error: task '{args.task_id}' not found in {TASKS_JSON}")
            sys.exit(1)
    else:
        tasks = tasks[args.start_from:]
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]

    print(f"\n{'#'*70}")
    print(f"  Batch Coder — {len(tasks)} task(s) to process")
    print(f"  Model: {MODEL}")
    print(f"  Output: {output_dir}")
    print(f"  Status: {status_file}")
    print(f"{'#'*70}")

    # Load existing results if resuming
    results = []
    if status_file.exists() and not args.task_id:
        try:
            existing = json.loads(status_file.read_text())
            results = existing.get("tasks", [])
            existing_ids = {r["task_id"] for r in results}
            tasks = [t for t in tasks if t["id"] not in existing_ids]
            print(f"  Resuming: {len(results)} completed, {len(tasks)} remaining")
        except Exception:
            pass

    # Run each task
    for i, task in enumerate(tasks):
        task_id = task["id"]
        proto_id = task.get("interface_protocol", "pytorch_task_v1")
        protocol = protocols.get(proto_id, {})

        print(f"\n[{i+1}/{len(tasks)}] Starting {task_id}...")

        result = run_single_task(task, protocol, output_dir)
        results.append(result)

        # Save after each task (in case of crash)
        save_status(results, status_file)

    # Final summary
    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  BATCH COMPLETE: {success}/{total} tasks succeeded")
    print(f"{'='*70}")

    # Print per-task table
    print(f"\n  {'Task ID':<40} {'Status':<20} {'Time':>8}")
    print(f"  {'-'*40} {'-'*20} {'-':->8}")
    for r in results:
        icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {icon} {r['task_id']:<38} {r['status']:<20} {r['duration_sec']:>6.1f}s")

    save_status(results, status_file)


if __name__ == "__main__":
    main()

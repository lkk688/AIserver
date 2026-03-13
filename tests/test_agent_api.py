#!/usr/bin/env python3
"""
Test script for the BatchAgent FastAPI agent service.

Assumes the agent service is running at http://localhost:8090.
Start it with:
    cd /Users/kaikailiu/Documents/MyRepo/AIserver
    uvicorn BatchAgent.agent_service:app --host 0.0.0.0 --port 8090 --reload

Requires: httpx  (pip install httpx)
"""
import json
import sys
import time

import httpx

BASE_URL = "http://localhost:8090"
TIMEOUT  = 600.0   # agent can take a while

# A very quick, self-contained task so the test finishes fast.
QUICK_GOAL = (
    "Write a Python function called `add(a, b)` that returns a + b. "
    "Then call finish_task to report completion."
)


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────
def health_check() -> bool:
    try:
        with httpx.Client(timeout=5) as c:
            resp = c.get(f"{BASE_URL}/health")
        resp.raise_for_status()
        print(f"  Health: {resp.json()}")
        return True
    except Exception as exc:
        print(f"  ❌ Agent service not reachable: {exc}")
        print(f"  Start it with: uvicorn BatchAgent.agent_service:app --host 0.0.0.0 --port 8090 --reload")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. /agent/run  (blocking)
# ─────────────────────────────────────────────────────────────────────────────
def test_agent_run():
    separator("POST /agent/run  (blocking)")
    payload = {
        "goal": QUICK_GOAL,
        "tool_strategy": "text_only", #"native_all",   # simplest strategy — no native JSON tools needed
        "verbose": False,
        "max_output": 1024,
    }
    try:
        print("  Submitting task (this may take 30-120 seconds)…")
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BASE_URL}/agent/run", json=payload)
            
        if resp.status_code != 200:
            print(f"  ❌ API HTTP Error {resp.status_code}: {resp.text}")
            return None, False

        data = resp.json()
        if not isinstance(data, dict):
            print(f"  ❌ Invalid JSON response: {data}")
            return None, False

        print(f"  task_id     : {data.get('task_id')}")
        print(f"  success     : {data.get('success')}")
        print(f"  status      : {data.get('status')}")
        print(f"  started_at  : {data.get('started_at')}")
        print(f"  finished_at : {data.get('finished_at')}")
        result_val = data.get('result')
        if result_val is None:
            result_val = ""
        print(f"  result      : {result_val[:100]}...")

        if data.get("error"):
            print(f"  error       : {data['error']}")
            
        task_id = data.get("task_id")
        if data.get("status") == "done" and data.get("success") is True and "result" in data:
            print("  ✅ PASS")
            return task_id, True
        else:
            print("  ❌ FAIL")
            return task_id, False
    except Exception as exc:
        print(f"  ❌ Request FAIL: {exc}")
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# 2. /agent/status  (poll)
# ─────────────────────────────────────────────────────────────────────────────
def test_agent_status(task_id: str):
    separator(f"GET /agent/status/{task_id}")
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{BASE_URL}/agent/status/{task_id}")
        resp.raise_for_status()
        data = resp.json()
        print(f"  status  : {data['status']}")
        print(f"  success : {data.get('success')}")
        print("  ✅ PASS")
        return True
    except Exception as exc:
        print(f"  ❌ FAIL: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. /agent/stream  (SSE)
# ─────────────────────────────────────────────────────────────────────────────
def test_agent_stream():
    separator("POST /agent/stream  (SSE streaming)")
    payload = {
        "goal": "Say exactly: Agent streaming works! Then call finish_task.",
        "tool_strategy": "text_only", #"native_all",
        "verbose": False,
        "max_output": 256,
    }
    accumulated = ""
    token_count = 0
    task_id_seen = None
    done_event = None

    print("  Streaming output:")
    print("  " + "-"*50)
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{BASE_URL}/agent/stream", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:].strip())

                    if ev["type"] == "start":
                        task_id_seen = ev.get("task_id")
                        print(f"  [START] task_id={task_id_seen}")
                    elif ev["type"] == "token":
                        token_count += 1
                        accumulated += ev["data"]
                        print(ev["data"], end="", flush=True)
                    elif ev["type"] == "done":
                        done_event = ev
                    elif ev["type"] == "error":
                        print(f"\n[SERVER ERROR]: {ev['detail']}")
                        done_event = ev # Set done event so it exits cleanly

        print()  # newline after streamed tokens
        print("  " + "-"*50)
        print(f"  full content (truncated): {accumulated[:200]!r}")
        print(f"  token chunks: {token_count}")
        print(f"  done event  : {done_event}")
        success = done_event is not None
        print("  ✅ PASS" if success else "  ❌ FAIL: no done event received")
        return success
    except Exception as exc:
        print(f"\n  ❌ FAIL: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n🧪 BatchAgent Service — Agent API Test Suite")
    print(f"   Base URL : {BASE_URL}")

    separator("Health Check")
    if not health_check():
        sys.exit(1)

    results = []

    # Run blocking task first (also gives us a task_id to test status endpoint)
    task_id, run_ok = test_agent_run()
    results.append(run_ok)

    # Poll status with the task_id from the run test
    if task_id:
        status_ok = test_agent_status(task_id)
        results.append(status_ok)
    else:
        print("\n  (Skipping status test — no task_id from run test)")

    # Streaming test
    stream_ok = test_agent_stream()
    results.append(stream_ok)

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed")
    print('='*60)
    sys.exit(0 if passed == total else 1)

def single_test():
    payload = {
        "goal": "Search latest Apple Studio display that is just released. What kind of chip is used and what's the storage size?",
        "tool_strategy": "text_only", #"native_all",
        "verbose": True,
        "max_output": 2048,
        "enable_turn_limits": True,
    }
    accumulated = ""
    token_count = 0
    task_id_seen = None
    done_event = None

    print("  Streaming output:")
    print("  " + "-"*50)
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{BASE_URL}/agent/stream", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:].strip())

                    if ev["type"] == "start":
                        task_id_seen = ev.get("task_id")
                        print(f"  [START] task_id={task_id_seen}")
                    elif ev["type"] in ["token", "message"]:  # Compatibility for old and new wrapper
                        token_count += 1
                        val = ev.get("data", "")
                        accumulated += val
                        print(val, end="", flush=True)
                    elif ev["type"] == "think":
                        token_count += 1
                        val = ev.get("data", "")
                        accumulated += val
                        sys.stdout.write(f"\033[90m{val}\033[0m")
                        sys.stdout.flush()
                    elif ev["type"] == "tool":
                        name = ev.get("name", "unknown")
                        args = ev.get("args_delta", "")
                        if ev.get("status") == "started":
                            print(f"\n\033[35m🛠️ Calling Tool: {name}...\033[0m", end="", flush=True)
                        elif args:
                            print(f"\033[35m{args}\033[0m", end="", flush=True)
                    elif ev["type"] == "done":
                        done_event = ev
                    elif ev["type"] == "error":
                        raise RuntimeError(ev["detail"])

        print()  # newline after streamed tokens
        print("  " + "-"*50)
        print(f"  full content (truncated): {accumulated[:200]!r}")
        print(f"  token chunks: {token_count}")
        print(f"  done event  : {done_event}")
        success = done_event is not None
        
        if success:
            final_ans = done_event.get("result", None)
            print(f"\n  [FINAL RESULT EXTRACTED]:\n{final_ans}")
            
        print("  ✅ PASS" if success else "  ❌ FAIL: no done event received")
        return success
    except Exception as exc:
        print(f"\n  ❌ FAIL: {exc}")
        return False

if __name__ == "__main__":
    #main()
    single_test()

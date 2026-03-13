#!/usr/bin/env python3
"""
Test script for the backend LLM API endpoints.

Assumes the backend FastAPI server is running at http://localhost:8080.
Start it with:
    cd /Users/kaikailiu/Documents/MyRepo/AIserver
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload

Requires: httpx  (pip install httpx)
"""
import asyncio
import json
import sys

import httpx

BASE_URL = "http://localhost:8080/api/v1"
TIMEOUT = 120.0

SIMPLE_MESSAGES = [
    {"role": "user", "content": "Say exactly: Hello from LLM API! (nothing else)"}
]


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Blocking complete
# ─────────────────────────────────────────────────────────────────────────────
def test_complete():
    separator("POST /llm/complete  (blocking, non-streaming)")
    payload = {
        "messages": SIMPLE_MESSAGES,
        "temperature": 0.0,
        "max_output_tokens": 64,
        "stream": False,
        "verbose": False,
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BASE_URL}/llm/complete", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"  content  : {data['content']!r}")
        print(f"  usage    : {data['usage']}")
        print("  ✅ PASS")
        return True
    except Exception as exc:
        print(f"  ❌ FAIL: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Streaming complete via SSE
# ─────────────────────────────────────────────────────────────────────────────
def test_complete_stream():
    separator("POST /llm/complete_stream  (SSE streaming)")
    payload = {
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        "temperature": 0.0,
        "max_output_tokens": 64,
        "verbose": False,
    }
    accumulated = ""
    usage = None
    print("  Tokens received: ", end="", flush=True)
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{BASE_URL}/llm/complete_stream", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    ev = json.loads(line[5:].strip())
                    if ev["type"] == "token":
                        accumulated += ev["data"]
                        print(ev["data"], end="", flush=True)
                    elif ev["type"] == "done":
                        usage = ev.get("usage")
                    elif ev["type"] == "error":
                        raise RuntimeError(ev["detail"])
        print()  # newline after tokens
        print(f"  full content: {accumulated!r}")
        print(f"  usage       : {usage}")
        print("  ✅ PASS")
        return True
    except Exception as exc:
        print(f"\n  ❌ FAIL: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Continuation
# ─────────────────────────────────────────────────────────────────────────────
def test_complete_continuation():
    separator("POST /llm/complete_continuation  (auto-continuation, no tools)")
    payload = {
        "messages": SIMPLE_MESSAGES,
        "temperature": 0.0,
        "max_output_tokens": 64,
        "stream": True,
        "verbose": False,
        "tool_strategy": "text",
        "allowlist": [],
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{BASE_URL}/llm/complete_continuation", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print(f"  content  : {data['content']!r}")
        print(f"  usage    : {data['usage']}")
        print("  ✅ PASS")
        return True
    except Exception as exc:
        print(f"  ❌ FAIL: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n🧪 BatchAgent Backend — LLM API Test Suite")
    print(f"   Base URL : {BASE_URL}")

    # Quick health check first
    try:
        with httpx.Client(timeout=5) as c:
            health = c.get(f"{BASE_URL.rsplit('/api', 1)[0]}/health")
        if health.status_code != 200:
            raise RuntimeError(f"HTTP {health.status_code}")
        print(f"\n   Health   : {health.json()}")
    except Exception as exc:
        print(f"\n❌ Backend not reachable at {BASE_URL}: {exc}")
        print("   Start it with: uvicorn backend.app.main:app --host 0.0.0.0 --port 8080 --reload")
        sys.exit(1)

    results = [
        test_complete(),
        test_complete_stream(),
        test_complete_continuation(),
    ]

    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} tests passed")
    print('='*60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

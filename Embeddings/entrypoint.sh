#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"

exec python3 -m uvicorn app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

"""
Agent FastAPI Service — exposes BatchAgent as an HTTP API.

Uses backend.app.services.agent.AgentService which delegates ALL agent
logic to the tested UniversalAgent class from BatchAgent/agent_main.py.

Start with:
    cd /Users/kaikailiu/Documents/MyRepo/AIserver
    uvicorn BatchAgent.agent_service:app --host 0.0.0.0 --port 8090 --reload

Endpoints:
  POST /agent/run              → blocking ReAct loop, returns JSON
  POST /agent/stream           → streaming ReAct loop via SSE
  GET  /agent/status/{task_id} → poll task status
  GET  /health                 → health check
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── resolve project root ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.app.services.agent import AgentService
from BatchAgent.tools_registry import get_base_tools

# ─────────────────────────────────────────────────────────────────────────────
# Task Registry – in-memory (replace with Redis/DB for production)
# ─────────────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    success: Optional[bool] = None
    result:  Optional[str] = None  # [NEW] final result extracted from agent
    error:   Optional[str] = None
    started_at:    Optional[str] = None
    finished_at:   Optional[str] = None
    workspace_dir: Optional[str] = None  # set after run() starts
    session_dir:   Optional[str] = None  # per-turn debug logs dir
    # Token queue is transient — not persisted
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


# ── Disk-persisted task registry (survives uvicorn --reload) ──────────────────

_TASKS_DIR = Path(".agent") / "tasks"
_TASKS_DIR.mkdir(parents=True, exist_ok=True)

_tasks: Dict[str, TaskRecord] = {}  # in-memory cache (rebuilt from disk on reload)


def _task_path(task_id: str) -> Path:
    return _TASKS_DIR / f"{task_id}.json"


def _persist(record: TaskRecord) -> None:
    """Write serialisable fields to disk so status survives reload."""
    data = {
        "task_id":     record.task_id,
        "goal":        record.goal,
        "status":      record.status,
        "success":     record.success,
        "result":      record.result,
        "error":       record.error,
        "started_at":  record.started_at,
        "finished_at": record.finished_at,
        "workspace_dir": record.workspace_dir,
        "session_dir":   record.session_dir,
    }
    _task_path(record.task_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_task(task_id: str) -> Optional[TaskRecord]:
    """Load a task from disk (used for status queries after a reload)."""
    p = _task_path(task_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rec = TaskRecord(
            task_id=data["task_id"],
            goal=data["goal"],
            status=TaskStatus(data["status"]),
            success=data.get("success"),
            result=data.get("result"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            workspace_dir=data.get("workspace_dir"),
            session_dir=data.get("session_dir"),
        )
        _tasks[task_id] = rec   # warm in-memory cache
        return rec
    except Exception:
        return None


def _get_task(task_id: str) -> Optional[TaskRecord]:
    """Get from cache; fall back to disk read (handles post-reload queries)."""
    return _tasks.get(task_id) or _load_task(task_id)



# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

# ─── MinIO initialisation ────────────────────────────────────────────────────
def _init_minio():
    import os, yaml
    cfg_path = os.getenv("APP_CONFIG_PATH", "backend/config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        mc = cfg.get("minio", {})
        if mc.get("endpoint"):
            from BatchAgent.minio_uploader import init_minio, _client as _mc_after
            init_minio(
                endpoint=mc["endpoint"],
                access_key=mc.get("access_key", ""),
                secret_key=mc.get("secret_key", ""),
                bucket=mc.get("bucket", "aiagent"),
                secure=mc.get("secure", False),
            )
            # Only log success if the client is actually available
            from BatchAgent.minio_uploader import _client as _mc_check
            if _mc_check is not None:
                print(f"[MinIO] Ready → {mc['endpoint']}/{mc.get('bucket','aiagent')}")
    except Exception as exc:
        print(f"[MinIO] Config load failed: {exc}")

_init_minio()

app = FastAPI(
    title="BatchAgent Service",
    description=(
        "FastAPI wrapper around BatchAgent/agent_main.py's UniversalAgent. "
        "All agent logic lives in the original agent_main.py — this service "
        "only handles HTTP routing and SSE streaming."
    ),
    version="2.0.0",
)

# Importable router — lets main.py use include_router instead of app.mount,
# which avoids the extra ASGI sub-app layer that can interfere with streaming.
router = APIRouter(tags=["Agent"])


@router.get("/health", tags=["Meta"])
@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "BatchAgent"}


# ─────────────────────────────── Request / Response schemas ───────────────────

class RunRequest(BaseModel):
    goal: str           = Field(..., description="Natural language task goal")
    tool_strategy: str  = Field("hybrid",  description="'native_all' | 'hybrid' | 'text_only'")
    provider: str       = Field("openai",       description="'openai' or 'anthropic'")
    model: str          = Field(default_factory=lambda: os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    base_url: str       = Field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"))
    api_key: str        = Field(default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY"))
    max_context: int    = 16384
    max_output: int     = 4096
    verbose: bool       = False
    output_dir: str     = "./agent_workspace"
    domain: str         = "general"
    location: str       = "California, United States"
    serper_api_key: str = Field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))
    allowlist: List[str] = []
    temperature: float  = Field(0.1, description="LLM temperature, lower is better for strict format")
    memory_strategy: str = Field("sliding_window", description="'sliding_window' or 'summarize'")
    enable_turn_limits: bool = Field(True, description="Expose current and max turns in prompt")
    max_turns: int      = Field(15, description="Maximum ReAct turns before aborting")
    backend: str        = Field(default_factory=lambda: os.environ.get("VLLM_BACKEND", "vllm"))
    enable_thinking: bool = Field(default_factory=lambda: os.environ.get("ENABLE_THINKING", "true").lower() == "true")


class RunResponse(BaseModel):
    task_id:       str
    success:       bool
    status:        str
    result:        Optional[str] = None
    started_at:    Optional[str] = None
    finished_at:   Optional[str] = None
    error:         Optional[str] = None
    workspace_dir: Optional[str] = None  # where agent output files are written
    session_dir:   Optional[str] = None  # per-turn debug logs (response.md, parsed_actions.json, …)


class StatusResponse(BaseModel):
    task_id:       str
    status:        str
    goal:          str
    success:       Optional[bool] = None
    result:        Optional[str] = None
    started_at:    Optional[str] = None
    finished_at:   Optional[str] = None
    error:         Optional[str] = None
    workspace_dir: Optional[str] = None
    session_dir:   Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_service(req: RunRequest) -> AgentService:
    """Construct an AgentService from the request parameters."""
    return AgentService(
        base_url=req.base_url,
        api_key=req.api_key,
        model=req.model,
        provider=req.provider,
        model_max_context=req.max_context,
        max_output_tokens=req.max_output,
        tool_strategy=req.tool_strategy,
        domain=req.domain,
        location=req.location,
        output_dir=req.output_dir,
        verbose=req.verbose,
        serper_api_key=req.serper_api_key,
        temperature=req.temperature,
        memory_strategy=req.memory_strategy,
        enable_turn_limits=req.enable_turn_limits,
        max_turns=req.max_turns,
        backend=req.backend,
        enable_thinking=req.enable_thinking,
    )


async def _run_blocking(record: TaskRecord, req: RunRequest) -> bool:
    record.status = TaskStatus.RUNNING
    record.started_at = datetime.utcnow().isoformat()
    _persist(record)
    try:
        svc = _make_service(req)
        success, session_dir, workspace_dir, final_result = await svc.run(goal=req.goal, allowlist=req.allowlist)
        record.success = success
        record.result  = final_result
        record.status  = TaskStatus.DONE
        record.session_dir   = str(session_dir)
        record.workspace_dir = str(workspace_dir)
    except Exception as exc:
        record.success = False
        record.error   = str(exc)
        record.status  = TaskStatus.FAILED
    finally:
        record.finished_at = datetime.utcnow().isoformat()
        _persist(record)
    return record.success or False


async def _run_streaming(
    record: TaskRecord,
    req: RunRequest,
    on_event: Callable[[Dict[str, Any]], Awaitable[None]],
) -> bool:
    record.status = TaskStatus.RUNNING
    record.started_at = datetime.utcnow().isoformat()
    _persist(record)
    try:
        svc = _make_service(req)
        success, session_dir, workspace_dir, final_result = await svc.run_stream(
            goal=req.goal,
            on_event=on_event,
            allowlist=req.allowlist,
        )
        record.success = success
        record.result  = final_result
        record.status  = TaskStatus.DONE
        record.session_dir   = str(session_dir)
        record.workspace_dir = str(workspace_dir)
    except Exception as exc:
        record.success = False
        record.error   = str(exc)
        record.status  = TaskStatus.FAILED
        _persist(record)
        raise exc
    finally:
        record.finished_at = datetime.utcnow().isoformat()
        _persist(record)
    return record.success or False


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/agent/run", response_model=RunResponse)
@app.post("/agent/run", response_model=RunResponse, tags=["Agent"])
async def agent_run(req: RunRequest):
    """
    Run a full ReAct agent task and wait for completion (blocking).

    For long-running tasks prefer `/agent/stream` to receive live output.
    """
    task_id = str(uuid.uuid4())
    record  = TaskRecord(task_id=task_id, goal=req.goal)
    _tasks[task_id] = record

    await _run_blocking(record, req)

    return RunResponse(
        task_id=task_id,
        success=record.success or False,
        status=record.status,
        result=record.result,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
        workspace_dir=record.workspace_dir,
        session_dir=record.session_dir,
    )


@router.get("/agent/tools", summary="Get active tools available for the agent")
@app.get("/agent/tools", tags=["Agent"], summary="Get active tools available for the agent")
def get_active_tools(strategy: str = "native_all"):
    """
    Returns the list of active tools available to the agent for the given strategy.
    """
    tools = get_base_tools(strategy=strategy)
    return {"tools": tools}


@router.post("/agent/stream", summary="Stream agent task via SSE")
@app.post("/agent/stream", tags=["Agent"], summary="Stream agent task via SSE")
async def agent_stream(req: RunRequest):
    """
    Run a ReAct agent task and stream LLM tokens via Server-Sent Events.

    SSE event shapes:
      `{"type": "turn_start", "turn": N, "max_turns": M}`
      `{"type": "message",    "data": "<token>"}`
      `{"type": "think",      "data": "<token>"}`
      `{"type": "tool",       "name": "<name>", "status": "started"}`
      `{"type": "done",       "success": bool, "task_id": "...", "result": "..."}`
      `{"type": "error",      "detail": "<msg>"}`
    """
    task_id = str(uuid.uuid4())
    record  = TaskRecord(task_id=task_id, goal=req.goal)
    _tasks[task_id] = record
    queue: asyncio.Queue[Optional[str]] = record.token_queue

    token_stats: Dict[str, Any] = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_elapsed_s": 0.0}

    async def on_event(event: Dict[str, Any]) -> None:
        if event.get("type") == "usage":
            token_stats["total_prompt_tokens"] += event.get("prompt_tokens", 0)
            token_stats["total_completion_tokens"] += event.get("completion_tokens", 0)
            token_stats["total_elapsed_s"] += event.get("elapsed_s", 0.0)
            # Don't forward raw usage events to the client — they're summarized in done
            return
        payload = json.dumps(event)
        await queue.put(f"data: {payload}\n\n")

    async def producer() -> None:
        try:
            intro = json.dumps({"type": "start", "task_id": task_id, "goal": req.goal})
            await queue.put(f"data: {intro}\n\n")

            success = await _run_streaming(record, req, on_event=on_event)

            done = json.dumps({
                "type": "done",
                "success": success,
                "task_id": task_id,
                "result": record.result,
                "stats": token_stats,
            })
            await queue.put(f"data: {done}\n\n")
        except Exception as exc:
            import traceback
            trace_str = traceback.format_exc()
            print(f"!!! CRITICAL PRODUCER EXCEPTION !!!\n{trace_str}")
            err = json.dumps({"type": "error", "detail": f"{str(exc)}\n{trace_str}"})
            await queue.put(f"data: {err}\n\n")
        finally:
            await queue.put(None)  # sentinel

    async def event_generator():
        asyncio.create_task(producer())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/agent/status/{task_id}", response_model=StatusResponse)
@app.get("/agent/status/{task_id}", response_model=StatusResponse, tags=["Agent"])
def agent_status(task_id: str):
    """Poll the status of a submitted agent task."""
    record = _get_task(task_id)   # checks in-memory cache, then falls back to disk
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return StatusResponse(
        task_id=record.task_id,
        status=record.status,
        goal=record.goal,
        success=record.success,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
        workspace_dir=record.workspace_dir,
        session_dir=record.session_dir,
    )


def _parse_response_md(content: str) -> dict:
    """Extract thinking and plain text from a raw LLM response.md file."""
    think_matches = re.findall(r'<think>([\s\S]*?)</think>', content, re.DOTALL)
    thinking = '\n'.join(m.strip() for m in think_matches if m.strip())

    # Strip think blocks
    text = re.sub(r'<think>[\s\S]*?</think>', '', content, flags=re.DOTALL).strip()
    # Strip XML tool call blocks (hybrid strategy)
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text, flags=re.DOTALL).strip()
    # Strip JSON code fences
    text = re.sub(r'```json[\s\S]*?```', '', text, flags=re.DOTALL).strip()
    # Strip bare JSON objects that look like tool calls (start with {, contain "name":)
    # Use a simple line-level filter: drop lines inside a likely JSON block
    clean_lines, brace_depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        if brace_depth == 0:
            if s.startswith('{') and ('"name"' in s or '"function"' in s):
                brace_depth += s.count('{') - s.count('}')
                continue
            clean_lines.append(line)
        else:
            brace_depth += s.count('{') - s.count('}')
            if brace_depth <= 0:
                brace_depth = 0
    text = '\n'.join(clean_lines)
    # Remove memory-pruning separators ("---" alone on a line)
    text = re.sub(r'(?m)^-{3,}\s*$', '', text).strip()
    return {'thinking': thinking, 'text': text}


@router.get("/agent/sessions/{task_id}/turns", summary="Get per-turn logs for a session")
@app.get("/agent/sessions/{task_id}/turns", tags=["Agent"], summary="Get per-turn logs for a session")
def get_session_turns(task_id: str):
    """Return structured per-turn data reconstructed from session log directories."""
    record = _get_task(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    turns = []
    session_dir = Path(record.session_dir) if record.session_dir else None
    if session_dir and session_dir.exists():
        for turn_dir in sorted(session_dir.iterdir()):
            if not (turn_dir.is_dir() and turn_dir.name.isdigit()):
                continue
            turn_num = int(turn_dir.name)
            parsed: dict = {"turn": turn_num}

            resp_f = turn_dir / "response.md"
            if resp_f.exists():
                parsed.update(_parse_response_md(resp_f.read_text(encoding="utf-8")))

            act_f = turn_dir / "parsed_actions.json"
            if act_f.exists():
                try:
                    parsed["actions"] = json.loads(act_f.read_text(encoding="utf-8"))
                except Exception:
                    parsed["actions"] = []

            turns.append(parsed)

    # Attach generated workspace files
    workspace = record.workspace_dir
    files: list = []
    if workspace:
        ws = Path(workspace)
        if ws.exists():
            for f in sorted(ws.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if f.is_file() and not f.name.startswith('.') and f.suffix != '.db':
                    files.append({"name": f.name, "ext": f.suffix.lstrip('.').lower(), "size": f.stat().st_size})

    return {
        "task_id": task_id,
        "goal": record.goal,
        "status": record.status,
        "success": record.success,
        "result": record.result,
        "total_turns": len(turns),
        "turns": turns,
        "files": files[:20],
    }


@router.get("/agent/sessions", summary="List recent agent task sessions")
@app.get("/agent/sessions", tags=["Agent"], summary="List recent agent task sessions")
def list_sessions(limit: int = 50):
    """Return persisted task records sorted by most-recent first, with file listings."""
    records = []
    if _TASKS_DIR.exists():
        paths = sorted(_TASKS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # Attach list of files written to the workspace
                workspace = data.get("workspace_dir")
                files: list = []
                if workspace:
                    ws = Path(workspace)
                    if ws.exists():
                        for f in sorted(ws.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                            if f.is_file() and not f.name.startswith('.'):
                                files.append({
                                    "name": f.name,
                                    "size": f.stat().st_size,
                                    "ext": f.suffix.lstrip('.').lower(),
                                })
                data["files"] = files[:20]
                records.append(data)
            except Exception:
                pass
    return {"sessions": records}


@router.delete("/agent/sessions/{task_id}", summary="Delete a task session")
@app.delete("/agent/sessions/{task_id}", tags=["Agent"], summary="Delete a task session")
def delete_session(task_id: str):
    """Remove a task record from disk (and memory cache)."""
    path = _task_path(task_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    path.unlink()
    _tasks.pop(task_id, None)
    return {"deleted": task_id}


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "BatchAgent.agent_service:app",
        host="0.0.0.0",
        port=int(os.environ.get("AGENT_PORT", 8090)),
        reload=False,
    )

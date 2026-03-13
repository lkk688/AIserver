"""
LLM API Router — FastAPI endpoints that expose the LLMService.

Endpoints:
  POST /llm/complete             → single-shot, blocking, returns JSON
  POST /llm/complete_stream      → streaming SSE (text/event-stream)
  POST /llm/complete_continuation→ multi-turn auto-continuation, blocking JSON
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.services.llm import LLMService, get_llm_service

router = APIRouter(prefix="/llm", tags=["LLM"])


# ─────────────────────────── Request / Response schemas ───────────────────────

class Message(BaseModel):
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str


class CompleteRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.2
    max_output_tokens: int = 4096
    stream: bool = False
    verbose: bool = False


class CompleteStreamRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.2
    max_output_tokens: int = 4096
    verbose: bool = False


class ContinuationRequest(BaseModel):
    messages: List[Message]
    temperature: float = 0.2
    max_output_tokens: int = 4096
    stream: bool = True
    verbose: bool = False
    tool_strategy: str = Field(
        "auto",
        description="Tool parsing strategy: 'native_all' | 'hybrid' | 'text_only' | 'auto' | 'text'",
    )
    allowlist: Optional[List[str]] = None


class CompleteResponse(BaseModel):
    content: str
    usage: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_dicts(messages: List[Message]) -> List[Dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/complete", response_model=CompleteResponse, summary="Single-shot LLM completion")
async def complete(
    req: CompleteRequest,
    svc: LLMService = Depends(get_llm_service),
):
    """
    Single-shot LLM completion (no auto-continuation, no tool parsing).
    Returns the full response content and token usage metrics once the
    generation is complete.
    """
    try:
        content, usage = await svc.complete(
            messages=_to_dicts(req.messages),
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            stream=req.stream,
            verbose=req.verbose,
        )
        return CompleteResponse(content=content, usage=usage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/complete_stream", summary="Streaming LLM completion (SSE)")
async def complete_stream(
    req: CompleteStreamRequest,
    svc: LLMService = Depends(get_llm_service),
):
    """
    Streaming LLM completion via Server-Sent Events.

    The response is a text/event-stream where each event is a JSON object:
      - `{"type": "token",  "data": "<token_text>"}` for each token
      - `{"type": "done",   "usage": {...}}` when generation finishes
      - `{"type": "error",  "detail": "<msg>"}` on failure

    Clients should consume the stream and concatenate `data` fields from
    `token` events to reconstruct the full response.
    """
    messages = _to_dicts(req.messages)
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def on_token(token: str) -> None:
        payload = json.dumps({"type": "token", "data": token})
        await queue.put(f"data: {payload}\n\n")

    async def producer() -> None:
        try:
            _, usage = await svc.complete_stream(
                messages=messages,
                on_token=on_token,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                verbose=req.verbose,
            )
            done_payload = json.dumps({"type": "done", "usage": usage})
            await queue.put(f"data: {done_payload}\n\n")
        except Exception as exc:
            error_payload = json.dumps({"type": "error", "detail": str(exc)})
            await queue.put(f"data: {error_payload}\n\n")
        finally:
            await queue.put(None)  # sentinel

    async def event_generator():
        # Kick off the LLM call in the background
        asyncio.create_task(producer())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@router.post(
    "/complete_continuation",
    response_model=CompleteResponse,
    summary="Multi-turn LLM completion with auto-continuation",
)
async def complete_continuation(
    req: ContinuationRequest,
    svc: LLMService = Depends(get_llm_service),
):
    """
    Multi-turn LLM completion with automatic continuation.

    When the model's output is truncated (finish_reason='length'), this
    endpoint automatically sends a follow-up message and stitches the
    responses together.  It also parses native and text-based tool calls.

    Returns the fully stitched content (tool action details are not serialised
    in the response body; use the Agent API for full ReAct loop support).
    """
    try:
        content, actions = await svc.complete_with_continuation(
            messages=_to_dicts(req.messages),
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
            stream=req.stream,
            verbose=req.verbose,
            tool_strategy=req.tool_strategy,
            allowlist=req.allowlist,
        )
        usage: Dict[str, Any] = {
            "actions_count": len(actions),
            "action_types": list({type(a).__name__ for a in actions}),
        }
        return CompleteResponse(content=content, usage=usage)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

"""
LLM Service — wraps BatchAgent's LLM wrapper functions for use inside FastAPI.

Exposes three async methods:
  - complete()                  → single-shot call, returns (content, usage)
  - complete_stream()           → streaming call with on_token callback
  - complete_with_continuation()→ multi-turn auto-continuation, returns (content, actions)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

import httpx

# Make sure the project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from BatchAgent.llm_wrapper import (
    complete_with_async,
    complete_with_continuation_async,
)
from BatchAgent.mini_batch_agent import AgentAction


class LLMService:
    """
    Thin async service that owns an AsyncOpenAI (or Anthropic) client and
    delegates to the BatchAgent llm_wrapper functions.

    Instantiate once (e.g. as a FastAPI lifespan dependency) and reuse.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "EMPTY",
        model: str = "qwen3.5-9b",
        provider: str = "openai",
        model_max_context: int = 16384,
        timeout: float = 600.0,
    ):
        self.model = model
        self.provider = provider
        self.model_max_context = model_max_context

        if provider == "anthropic":
            from anthropic import AsyncAnthropic  # type: ignore
            self.client: Any = AsyncAnthropic(api_key=api_key)
        else:
            from openai import AsyncOpenAI  # type: ignore
            kwargs: Dict[str, Any] = {
                "api_key": api_key,
                "http_client": httpx.AsyncClient(timeout=timeout),
            }
            if base_url:
                kwargs["base_url"] = base_url
            self.client = AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------
    # 1. Simple single-shot completion (no continuation, no tool parsing)
    # ------------------------------------------------------------------
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        stream: bool = False,
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Single-shot LLM call.  No auto-continuation, no tool parsing.

        Returns:
            (content, usage_info)
        """
        return await complete_with_async(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model_max_context=self.model_max_context,
            provider=self.provider,
            stream=stream,
            verbose=verbose,
            on_token=None,
        )

    # ------------------------------------------------------------------
    # 2. Streaming single-shot completion (tokens forwarded via callback)
    # ------------------------------------------------------------------
    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        on_token: Callable[[str], Awaitable[None]],
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        verbose: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Streaming single-shot LLM call.

        Each token is forwarded to `on_token` as it arrives.  The full
        content and usage info are returned when the stream ends.

        Args:
            on_token: async callable receiving each text token fragment.
        """
        return await complete_with_async(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model_max_context=self.model_max_context,
            provider=self.provider,
            stream=True,
            verbose=verbose,
            on_token=on_token,
        )

    # ------------------------------------------------------------------
    # 3. Auto-continuation (multi-turn, with tool-call parsing)
    # ------------------------------------------------------------------
    async def complete_with_continuation(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        stream: bool = True,
        verbose: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_strategy: str = "auto",
        allowlist: Optional[List[str]] = None,
    ) -> Tuple[str, List[AgentAction]]:
        """
        Multi-turn wrapper with auto-continuation on finish_reason='length'
        and native / text-based tool-call parsing.

        Returns:
            (full_content, actions)
        """
        return await complete_with_continuation_async(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model_max_context=self.model_max_context,
            provider=self.provider,
            stream=stream,
            verbose=verbose,
            tools=tools,
            tool_strategy=tool_strategy,
            allowlist=allowlist,
        )


# ---------------------------------------------------------------------------
# Module-level singleton helper (reads config from environment / defaults)
# ---------------------------------------------------------------------------
_default_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Returns a module-level singleton LLMService built from environment vars.
    Suitable for use as a FastAPI dependency.

      VLLM_BASE_URL  – base URL for vLLM OpenAI-compatible endpoint
      VLLM_API_KEY   – API key (default: 'EMPTY' for local vLLM)
      VLLM_MODEL     – model name (default: 'qwen3.5-9b')
      LLM_PROVIDER   – 'openai' | 'anthropic' (default: 'openai')
      LLM_MAX_CTX    – model max context length (default: 16384)
    """
    global _default_service
    if _default_service is None:
        _default_service = LLMService(
            base_url=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=os.environ.get("VLLM_MODEL", "qwen3.5-9b"),
            provider=os.environ.get("LLM_PROVIDER", "openai"),
            model_max_context=int(os.environ.get("LLM_MAX_CTX", "16384")),
        )
    return _default_service

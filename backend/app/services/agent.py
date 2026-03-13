"""
Agent Service — thin wrapper around BatchAgent/agent_main.py for FastAPI use.

Similar in structure to backend/app/services/llm.py.
The key difference is that this service delegates ALL agent logic to the
already-tested UniversalAgent class from agent_main.py. No agent logic is
reimplemented here.

Usage:
    svc = AgentService(base_url=..., api_key=..., model=...)
    # blocking run
    success = await svc.run(goal="...", task_id="abc")
    # streaming run — tokens forwarded to `on_token` as they arrive
    success = await svc.run_stream(goal="...", on_token=my_cb, task_id="abc")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

# ── resolve project root so BatchAgent is importable ──────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # AIserver/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the working, tested classes from agent_main directly
from BatchAgent.agent_main import (
    AgentConfig,
    UniversalAgent,
    check_api_and_context,
)
from BatchAgent.mini_batch_agent_libs import (
    now_stamp, ensure_dirs, read_file, estimate_tokens, truncate_to_tokens,
)
from BatchAgent.tools_registry import get_base_tools, compile_tools_for_provider
from BatchAgent.prompt_registry import PromptRegistry


class AgentService:
    """
    Async service that wraps UniversalAgent for use inside FastAPI.

    Instantiate once per request (or share one instance with per-call overrides).
    All agent execution logic lives in agent_main.UniversalAgent — this class
    only handles bootstrapping (client, config, tools, prompts) and wires in
    the optional SSE streaming callback.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "EMPTY",
        model: str = "qwen3.5-9b",
        provider: str = "openai",
        model_max_context: int = 16384,
        max_output_tokens: int = 4096,
        tool_strategy: str = "native_all",
        domain: str = "general",
        location: str = "California, United States",
        output_dir: str = "./agent_workspace",
        verbose: bool = False,
        timeout: float = 1200.0,
        serper_api_key: str = "",
        temperature: float = 0.1,
        memory_strategy: str = "sliding_window",
        backend: str = "vllm",
        enable_thinking: bool = True,
        enable_turn_limits: bool = True,
        max_turns: int = 15,
    ):
        self.model = model
        self.provider = provider
        self.model_max_context = model_max_context
        self.max_output_tokens = max_output_tokens
        self.tool_strategy = tool_strategy
        self.domain = domain
        self.location = location
        self.output_dir = output_dir
        self.verbose = verbose
        self.serper_api_key = serper_api_key
        self.temperature = temperature
        self.memory_strategy = memory_strategy
        self.backend = backend
        self.enable_thinking = enable_thinking
        self.enable_turn_limits = enable_turn_limits
        self.max_turns = max_turns

        # Build async LLM client (OpenAI-compat or Anthropic)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Internal: bootstrap AgentConfig + compiled tools + prompts
    # ─────────────────────────────────────────────────────────────────────────
    def _bootstrap(
        self,
        stream_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        allowlist: Optional[List[str]] = None,
    ) -> tuple[AgentConfig, List[Dict[str, Any]], str, Callable, Path, Path]:
        """
        Create AgentConfig, tools, system prompt, and content_injector.

        Returns (config, compiled_tools, system_prompt, content_injector,
                 session_dir, workspace_dir).

        Session logs (per-turn response.md, parsed_actions.json, etc.) are written
        by agent_main.UniversalAgent into session_dir, which is located at:
            <output_dir>/agent_logs/<timestamp>/
        This keeps all run artefacts co-located with the workspace output.
        """
        workspace_dir = Path(self.output_dir).resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Co-locate session logs with the workspace so everything is in one place.
        # agent_main.UniversalAgent writes per-turn files here automatically:
        #   <timestamp>/0000/full_prompt_context.json
        #   <timestamp>/0000/latest_instruction.md
        #   <timestamp>/0000/response.md
        #   <timestamp>/0000/parsed_actions.json
        #   <timestamp>/0000/tool_execution_results.md  (when tools are called)
        log_base = workspace_dir / "agent_logs"
        log_base.mkdir(parents=True, exist_ok=True)
        session_dir = log_base / now_stamp()
        session_dir.mkdir(parents=True, exist_ok=True)

        # Keep a legacy .agent/ dir for RL trajectory storage (agent_main uses it)
        agent_dir = Path(".agent")
        ensure_dirs(agent_dir)

        config = AgentConfig(
            client=self.client,
            model=self.model,
            session_dir=session_dir,
            workspace_dir=workspace_dir,
            max_context=self.model_max_context,
            max_output=self.max_output_tokens,
            require_approval=False,
            agent_dir=agent_dir,
            tool_strategy=self.tool_strategy,
            provider=self.provider,
            verbose=self.verbose,
            serper_api_key=self.serper_api_key,
            domain=self.domain,
            location=self.location,
            temperature=self.temperature,
            memory_strategy=self.memory_strategy,
            enable_turn_limits=self.enable_turn_limits,
            max_turns=self.max_turns,
            backend=self.backend,
            enable_thinking=self.enable_thinking,
            stream_callback=stream_callback,  # ← wires SSE hook into agent
        )

        base_tools = get_base_tools(
            strategy=self.tool_strategy,
            enable_parallel=False,
            domain=self.domain,
        )
        compiled_tools = compile_tools_for_provider(
            base_tools, self.provider, self.tool_strategy
        )
        system_prompt = PromptRegistry.get_system_prompt(
            self.tool_strategy, base_tools, self.domain, self.location
        )

        def content_injector(files: list) -> str:
            md = ""
            for f in list(dict.fromkeys(files)):
                c = read_file(str(f))
                if c and not c.startswith("[MISSING FILE]"):
                    if estimate_tokens(c) > 8000:
                        c = truncate_to_tokens(c, 8000)
                    md += f"### File: {f}\n```text\n{c}\n```\n"
            return md

        return config, compiled_tools, system_prompt, content_injector, session_dir, workspace_dir

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    async def run(
        self,
        goal: str,
        allowlist: Optional[List[str]] = None,
    ) -> tuple[bool, Path, Path, str]:
        """
        Run a full ReAct agent task (blocking, no streaming).

        Returns (success, session_dir, workspace_dir, final_result_text).
        session_dir contains all per-turn debug logs written by UniversalAgent.
        """
        allowlist = allowlist or []
        config, compiled_tools, system_prompt, content_injector, session_dir, workspace_dir = \
            self._bootstrap(stream_callback=None, allowlist=allowlist)
        task_prompt = PromptRegistry.format_task(
            goal, allowlist, [], str(workspace_dir), content_injector
        )
        agent = UniversalAgent(
            config=config,
            system_message=system_prompt,
            tools=compiled_tools,
        )
        success, final_result = await agent.execute_task(
            task_goal=goal,
            task_idx=0,
            allowlist=allowlist,
            prompt_md=task_prompt,
        )
        return success, session_dir, workspace_dir, final_result

    async def run_stream(
        self,
        goal: str,
        on_event: Callable[[Dict[str, Any]], Awaitable[None]],
        allowlist: Optional[List[str]] = None,
    ) -> tuple[bool, Path, Path, str]:
        """
        Run a full ReAct agent task with token-level streaming.

        Each LLM event is forwarded to `on_event` as it is generated.
        Returns (success, session_dir, workspace_dir, final_result_text).
        """
        allowlist = allowlist or []
        config, compiled_tools, system_prompt, content_injector, session_dir, workspace_dir = \
            self._bootstrap(stream_callback=on_event, allowlist=allowlist)
        task_prompt = PromptRegistry.format_task(
            goal, allowlist, [], str(workspace_dir), content_injector
        )
        agent = UniversalAgent(
            config=config,
            system_message=system_prompt,
            tools=compiled_tools,
        )
        success, final_result = await agent.execute_task(
            task_goal=goal,
            task_idx=0,
            allowlist=allowlist,
            prompt_md=task_prompt,
        )
        return success, session_dir, workspace_dir, final_result


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton factory (for FastAPI Depends)
# ─────────────────────────────────────────────────────────────────────────────
_default_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """
    Returns a module-level singleton AgentService built from environment vars.

      VLLM_BASE_URL   – vLLM / OpenAI-compatible endpoint
      VLLM_API_KEY    – API key (default: 'EMPTY' for local vLLM)
      VLLM_MODEL      – model name (default: 'qwen3.5-9b')
      LLM_PROVIDER    – 'openai' | 'anthropic' (default: 'openai')
      LLM_MAX_CTX     – model max context length (default: 16384)
      AGENT_STRATEGY  – tool strategy (default: 'native_all')
      SERPER_API_KEY  – web search API key
    """
    global _default_agent_service
    if _default_agent_service is None:
        _default_agent_service = AgentService(
            base_url=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"),
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            model=os.environ.get("VLLM_MODEL", "qwen3.5-9b"),
            provider=os.environ.get("LLM_PROVIDER", "openai"),
            model_max_context=int(os.environ.get("LLM_MAX_CTX", "16384")),
            tool_strategy=os.environ.get("AGENT_STRATEGY", "native_all"),
            serper_api_key=os.environ.get("SERPER_API_KEY", ""),
            backend=os.environ.get("VLLM_BACKEND", "vllm"),
            enable_thinking=os.environ.get("ENABLE_THINKING", "true").lower() == "true",
        )
    return _default_agent_service

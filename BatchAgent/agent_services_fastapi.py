from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from BatchAgent.agent_main import AgentConfig, UniversalAgent
from BatchAgent.mini_batch_agent_libs import ensure_dirs, now_stamp
from BatchAgent.prompt_registry import PromptRegistry
from BatchAgent.tools_registry import compile_tools_for_provider, get_base_tools

_cwd_lock = asyncio.Lock()


class AgentService:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai",
        model_max_context: int = 16384,
        max_output_tokens: int = 4096,
        tool_strategy: str = "hybrid",
        domain: str = "general",
        location: str = "California, United States",
        current_time: str = "",
        output_dir: str = "./agent_workspace",
        verbose: bool = False,
        serper_api_key: str = "",
        temperature: float = 0.1,
        memory_strategy: str = "sliding_window",
        enable_turn_limits: bool = True,
        max_turns: int = 15,
        backend: str = "vllm",
        enable_thinking: bool = True,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.model_max_context = model_max_context
        self.max_output_tokens = max_output_tokens
        self.tool_strategy = tool_strategy
        self.domain = domain
        self.location = location
        self.current_time = current_time
        self.output_dir = output_dir
        self.verbose = verbose
        self.serper_api_key = serper_api_key
        self.temperature = temperature
        self.memory_strategy = memory_strategy
        self.enable_turn_limits = enable_turn_limits
        self.max_turns = max_turns
        self.backend = backend
        self.enable_thinking = enable_thinking

    def _build_client(self) -> Any:
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic

            return AsyncAnthropic(api_key=self.api_key)
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(timeout=1200.0),
        )

    def _build_task_prompt(self, goal: str, workspace_dir: Path) -> str:
        def _content_injector(_: List[str]) -> str:
            return ""

        return PromptRegistry.format_task(goal, [], [], workspace_dir.name, _content_injector)

    async def _run_internal(
        self,
        *,
        goal: str,
        allowlist: Optional[List[str]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        continue_session_dir: Optional[str] = None,
        continue_workspace_dir: Optional[str] = None,
        resume_messages: Optional[List[Dict[str, str]]] = None,
        resume_rl_trajectory: Optional[List[Dict[str, str]]] = None,
        start_turn_index: int = 0,
    ) -> Tuple[bool, Path, Path, str]:
        allowlist = allowlist or []
        workspace_dir = Path(continue_workspace_dir or self.output_dir).resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        agent_dir = (workspace_dir / ".agent").resolve()
        ensure_dirs(agent_dir)
        session_dir = Path(continue_session_dir).resolve() if continue_session_dir else (workspace_dir / now_stamp())
        session_dir.mkdir(parents=True, exist_ok=True)

        client = self._build_client()

        config = AgentConfig(
            client=client,
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
            current_time=self.current_time,
            temperature=self.temperature,
            memory_strategy=self.memory_strategy,
            backend=self.backend,
            enable_turn_limits=self.enable_turn_limits,
            max_turns=self.max_turns,
            stream_callback=on_event,
            enable_thinking=self.enable_thinking,
        )

        base_tools = get_base_tools(strategy=self.tool_strategy)
        compiled_tools = compile_tools_for_provider(
            base_tools=base_tools,
            provider=self.provider,
            strategy=self.tool_strategy,
        )
        system_prompt = PromptRegistry.get_system_prompt(
            strategy=self.tool_strategy,
            base_tools_list=base_tools,
            domain=self.domain,
            location=self.location,
            current_time=self.current_time,
        )

        agent = UniversalAgent(config=config, system_message=system_prompt, tools=compiled_tools)
        task_prompt = self._build_task_prompt(goal, workspace_dir)

        prev_cwd = Path.cwd()
        async with _cwd_lock:
            os.chdir(workspace_dir)
            try:
                success, final_result = await agent.execute_task(
                    task_goal=goal,
                    task_idx=0,
                    allowlist=allowlist,
                    prompt_md=task_prompt,
                    resume_messages=resume_messages,
                    start_turn_index=start_turn_index,
                    resume_rl_trajectory=resume_rl_trajectory,
                )
            finally:
                os.chdir(prev_cwd)

        return success, session_dir, workspace_dir, final_result

    async def run(
        self,
        *,
        goal: str,
        allowlist: Optional[List[str]] = None,
        continue_session_dir: Optional[str] = None,
        continue_workspace_dir: Optional[str] = None,
        resume_messages: Optional[List[Dict[str, str]]] = None,
        resume_rl_trajectory: Optional[List[Dict[str, str]]] = None,
        start_turn_index: int = 0,
    ) -> Tuple[bool, Path, Path, str]:
        return await self._run_internal(
            goal=goal,
            allowlist=allowlist,
            on_event=None,
            continue_session_dir=continue_session_dir,
            continue_workspace_dir=continue_workspace_dir,
            resume_messages=resume_messages,
            resume_rl_trajectory=resume_rl_trajectory,
            start_turn_index=start_turn_index,
        )

    async def run_stream(
        self,
        *,
        goal: str,
        on_event: Callable[[Dict[str, Any]], Awaitable[None]],
        allowlist: Optional[List[str]] = None,
        continue_session_dir: Optional[str] = None,
        continue_workspace_dir: Optional[str] = None,
        resume_messages: Optional[List[Dict[str, str]]] = None,
        resume_rl_trajectory: Optional[List[Dict[str, str]]] = None,
        start_turn_index: int = 0,
    ) -> Tuple[bool, Path, Path, str]:
        return await self._run_internal(
            goal=goal,
            allowlist=allowlist,
            on_event=on_event,
            continue_session_dir=continue_session_dir,
            continue_workspace_dir=continue_workspace_dir,
            resume_messages=resume_messages,
            resume_rl_trajectory=resume_rl_trajectory,
            start_turn_index=start_turn_index,
        )

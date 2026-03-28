from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from BatchAgent.agent_main_v2 import AgentConfig, UniversalAgent, check_api_and_context
from BatchAgent.mini_batch_agent_libs import ensure_dirs
from BatchAgent.prompt_registry_v2 import PromptRegistry
from BatchAgent.tools.tools_registry import compile_tools_for_provider, get_base_tools
from BatchAgent.tools.tool_registry_runtime import configure_global_tool_registry

_cwd_lock = asyncio.Lock()


class AgentService:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        provider: str = "openai",
        model_max_context: int = 0,   # 0 = auto-detect via API
        max_output_tokens: int = 0,   # 0 = derive from detected context
        tool_strategy: str = "hybrid",
        domain: str = "general",
        location: str = "California, United States",
        current_time: str = "",
        output_dir: str = "./agent_workspace",
        verbose: bool = False,
        serper_api_key: str = "",
        tavily_api_key: str = "",
        enable_youtube: bool = False,
        temperature: float = 0.1,
        memory_strategy: str = "sliding_window",
        enable_turn_limits: bool = True,
        max_turns: int = 15,
        backend: str = "vllm",
        enable_thinking: bool = True,
        # v2-only: OCR / document tools
        ocr_server: str = "http://localhost:8002/v1",
        ocr_model: str = "allenai/olmOCR-2-7B-1025-FP8",
        ocr_workspace: str = "./output_old/tmp_ocr",
        # v2-only: embedding for document RAG
        embedding_base_url: str = "http://localhost:8081/v1",
        embedding_api_key: str = "EMPTY",
        embedding_model: str = "text-embeddings-inference",
        # v2-only: registry feature flags
        enable_memory: bool = False,
        enable_web: bool = True,
        enable_document: bool = True,
        enable_code_tools: bool = True,
        enable_mutation: bool = True,
        enable_bash: bool = True,
        enable_parallel: bool = False,
        enable_meta: bool = True,
        # Reranker (TEI /v1/rerank compatible)
        reranker_base_url: str = "",
        reranker_api_key: str = "",
        reranker_model: str = "",
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
        self.tavily_api_key = tavily_api_key
        self.enable_youtube = enable_youtube
        self.temperature = temperature
        self.memory_strategy = memory_strategy
        self.enable_turn_limits = enable_turn_limits
        self.max_turns = max_turns
        self.backend = backend
        self.enable_thinking = enable_thinking
        # v2-only fields
        self.ocr_server = ocr_server
        self.ocr_model = ocr_model
        self.ocr_workspace = ocr_workspace
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.enable_memory = enable_memory
        self.enable_web = enable_web
        self.enable_document = enable_document
        self.enable_code_tools = enable_code_tools
        self.enable_mutation = enable_mutation
        self.enable_bash = enable_bash
        self.enable_parallel = enable_parallel
        self.enable_meta = enable_meta
        self.reranker_base_url = reranker_base_url
        self.reranker_api_key = reranker_api_key
        self.reranker_model = reranker_model

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

    def _build_task_prompt(
        self, goal: str, workspace_dir: Path, allowlist: List[str]
    ) -> str:
        def _content_injector(_: List[str]) -> str:
            return ""

        # Pass the full absolute path (not just .name) and the allowlist so the
        # agent sees its permitted output files in the task prompt.
        return PromptRegistry.format_task(
            goal, allowlist, [], str(workspace_dir), _content_injector
        )

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
        # workspace_dir IS the session folder: {user_root}/.agent/sessions/{timestamp}_{task_id}/
        # The agent's CWD, output files, and turn logs all live here.
        workspace_dir = Path(continue_workspace_dir or self.output_dir).resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # agent_dir is shared at the user root level: {user_root}/.agent/
        # Path structure: workspace_dir -> sessions/{ts_id}/ -> .agent/ -> user_root/
        agent_dir = workspace_dir.parent.parent
        ensure_dirs(agent_dir)

        # session_dir == workspace_dir: one session folder = one chat session
        session_dir = workspace_dir

        client = self._build_client()

        # Pre-flight connectivity check — raises before launching the agent so
        # callers can emit a clean error SSE instead of looping on LLM failures.
        if self.provider != "anthropic":
            try:
                if hasattr(client, "models"):
                    await client.models.list()
                else:
                    await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                    )
            except Exception as e:
                raise ConnectionError(f"Cannot connect to LLM API: {e}") from e

        # Auto-detect the model's actual context window unless the caller
        # pinned it explicitly.  max_output scales proportionally (ctx/4).
        if self.model_max_context > 0:
            detected_ctx = self.model_max_context
        else:
            detected_ctx = await check_api_and_context(
                client, self.provider, self.model, 0
            )
        detected_max_output = (
            self.max_output_tokens
            if self.max_output_tokens > 0
            else min(32768, max(4096, detected_ctx // 4))
        )

        # Configure the global tool registry before building tools/prompt
        configure_global_tool_registry(
            strategy=self.tool_strategy,
            profile=self.domain,
            domain=self.domain,
            enable_memory=self.enable_memory,
            enable_web=self.enable_web,
            enable_document=self.enable_document,
            enable_code_tools=self.enable_code_tools,
            enable_mutation=self.enable_mutation,
            enable_bash=self.enable_bash,
            enable_parallel=self.enable_parallel,
            enable_meta=self.enable_meta,
        )

        base_tools = get_base_tools(
            strategy=self.tool_strategy,
            enable_parallel=self.enable_parallel,
            domain=self.domain,
        )
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

        # Shared document cache lives at the user-root level: {user_root}/.cache/documents
        # Structure: workspace_dir = {user_root}/.agent/sessions/{ts_id}/
        # so user_root = workspace_dir.parent.parent.parent
        user_root = workspace_dir.parent.parent.parent
        document_cache_dir = str(user_root / ".cache" / "documents")

        config = AgentConfig(
            client=client,
            model=self.model,
            session_dir=session_dir,
            workspace_dir=workspace_dir,
            max_context=detected_ctx,
            max_output=detected_max_output,
            require_approval=False,
            agent_dir=agent_dir,
            tool_strategy=self.tool_strategy,
            provider=self.provider,
            verbose=self.verbose,
            serper_api_key=self.serper_api_key,
            tavily_api_key=self.tavily_api_key,
            enable_youtube=self.enable_youtube,
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
            ocr_server=self.ocr_server,
            ocr_model=self.ocr_model,
            ocr_workspace=self.ocr_workspace,
            document_cache_dir=document_cache_dir,
            embedding_base_url=self.embedding_base_url,
            embedding_api_key=self.embedding_api_key,
            embedding_model=self.embedding_model,
            reranker_base_url=self.reranker_base_url,
            reranker_api_key=self.reranker_api_key,
            reranker_model=self.reranker_model,
        )

        agent = UniversalAgent(config=config, system_message=system_prompt, tools=compiled_tools)
        task_prompt = self._build_task_prompt(goal, workspace_dir, allowlist)

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
        try:
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
        except ConnectionError as e:
            await on_event({"type": "error", "detail": str(e)})
            # Return a failure tuple so producer sends a clean done event
            # instead of a traceback-filled error SSE.
            workspace_dir = Path(continue_workspace_dir or self.output_dir).resolve()
            return False, workspace_dir, workspace_dir, str(e)

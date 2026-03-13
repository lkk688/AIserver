import asyncio
from BatchAgent.agent_service import _make_service, RunRequest
from BatchAgent.agent_main import UniversalAgent

async def test():
    req = RunRequest(
        goal="Say exactly: Agent streaming works! Then call finish_task.",
        tool_strategy="text_only",
        verbose=True,
        max_output=256,
        backend="openai",
        enable_thinking=False
    )
    svc = _make_service(req)
    
    # Run _bootstrap
    config, compiled_tools, system_prompt, content_injector, session_dir, workspace_dir = svc._bootstrap(stream_callback=None, allowlist=[])
    
    print("\n--- Bootstrapped config ---")
    print(f"session_dir: {session_dir}")
    print(f"workspace_dir: {workspace_dir}")
    
    # Run Agent
    agent = UniversalAgent(
        config=config,
        system_message=system_prompt,
        tools=compiled_tools,
    )
    
    print("\n--- Running agent ---")
    try:
        success, final_result = await agent.execute_task(
            task_goal=req.goal,
            task_idx=0,
            allowlist=[],
            prompt_md="Say exactly: Agent streaming works! Then call finish_task."
        )
        print(f"\n--- Result ---")
        print(f"Success: {success}")
        print(f"Final Result: {final_result}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())

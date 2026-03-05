import os
import asyncio
from rich.console import Console
from typing import Any
# 导入不同的异步客户端
from openai import AsyncOpenAI
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 导入我们刚刚写的 Wrapper 和 Action 类
from BatchAgent.llm_wrapper import complete_with_continuation_async, ActionToolCall

console = Console()

# ==========================================
# 1. 定义不同平台的 Native Tools Schema
# ==========================================

# OpenAI / vLLM 的标准工具格式 (嵌套在 type 和 function 里面)
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    }
]

# Anthropic (Claude) 的标准工具格式 (扁平化)
ANTHROPIC_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the internet for real-time information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"]
        }
    }
]

# ==========================================
# 2. 通用测试执行器
# ==========================================

async def run_test(name: str, client: Any, model: str, provider: str, tools: list):
    console.print(f"\n[bold magenta]=== 测试开始: {name} ({model}) ===[/bold magenta]")
    
    # 我们故意问一个需要查资料的问题，逼迫模型调用 web_search 工具
    messages = [
        {"role": "system", "content": "You are an AI assistant. If asked about current events, you MUST use the `web_search` tool."},
        {"role": "user", "content": "请帮我搜索一下2026年最近关于 SpaceX 星舰试飞的新闻。"}
    ]
    
    try:
        content, actions = await complete_with_continuation_async(
            client=client,
            model=model,
            messages=messages,
            provider=provider,
            tools=tools,
            tool_strategy="text", #"auto", # 开启 auto 模式，优先解析原生 JSON
            max_output_tokens=1024,
            stream=True if provider == "openai" else False # Anthropic 测试非流式，OpenAI 测试流式拼接
        )
        
        console.print("\n[bold green]返回的正文 (Content):[/bold green]")
        console.print(content if content else "[No text generated, directly called tool]")
        
        console.print("\n[bold green]解析出的动作 (Parsed Actions):[/bold green]")
        if not actions:
            console.print("[red]❌ 测试失败：没有解析出任何工具调用！[/red]")
        else:
            for act in actions:
                if isinstance(act, ActionToolCall):
                    console.print(f"✅ 成功捕获原生 ToolCall! 工具名: [cyan]{act.name}[/cyan], 参数: [yellow]{act.args}[/yellow]")
                else:
                    console.print(f"捕获到其他 Action: {act}")
                    
    except Exception as e:
        console.print(f"[red]调用出错: {e}[/red]")


# ==========================================
# 3. Main 函数，分别拉起三个平台的测试
# ==========================================

async def main():
    # ---------------------------------------------------------
    # 测试 1: 本地 vLLM (Qwen3.5-9B) - 使用 OpenAI 兼容协议
    # ---------------------------------------------------------
    vllm_client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1", # 换成你刚才测出 130T/s 的那个地址
        api_key="EMPTY", 
    )
    await run_test(
        name="Local vLLM (Qwen)", 
        client=vllm_client, 
        model="qwen3.5-9b", 
        provider="openai", 
        tools=OPENAI_TOOLS
    )

    # ---------------------------------------------------------
    # 测试 2: 官方 OpenAI (例如 gpt-4o-mini)
    # ---------------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        openai_client = AsyncOpenAI(api_key=openai_key)
        await run_test(
            name="Official OpenAI", 
            client=openai_client, 
            model="gpt-4o-mini", 
            provider="openai", 
            tools=OPENAI_TOOLS
        )
    else:
        console.print("\n[dim]跳过 OpenAI 测试 (未设置 OPENAI_API_KEY)[/dim]")

    # ---------------------------------------------------------
    # 测试 3: 官方 Anthropic (例如 claude-3-5-sonnet-20241022)
    # ---------------------------------------------------------
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key and AsyncAnthropic:
        anthropic_client = AsyncAnthropic(api_key=anthropic_key)
        await run_test(
            name="Official Anthropic", 
            client=anthropic_client, 
            model="claude-3-5-sonnet-20241022", 
            provider="anthropic", 
            tools=ANTHROPIC_TOOLS
        )
    else:
        console.print("\n[dim]跳过 Anthropic 测试 (未设置 ANTHROPIC_API_KEY 或未安装 anthropic 库)[/dim]")

if __name__ == "__main__":
    asyncio.run(main())

"""
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen \
    --trust-remote-code

python BatchAgent/test_wrapper_providers.py
"""
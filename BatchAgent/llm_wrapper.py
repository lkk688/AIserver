import time
import re
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from rich.console import Console
import sys
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


console = Console()

# 假设这些是你已有的工具函数
from BatchAgent.mini_batch_agent_libs import (
    estimate_tokens, compress_messages, compute_safe_max_tokens, now_stamp, write_jsonl
)
from BatchAgent.mini_batch_agent import AgentAction, ActionWriteFile, ActionReplaceText, ActionToolCall, parse_text_actions

from typing import List, Dict, Any

# ==========================================
# 1. Base Tool Definitions (Single Source of Truth)
# ==========================================
BASE_TOOLS = [
    {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing file with new content. Use this for new files or when changes are too complex for search_and_replace.",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file"},
            "content": {"type": "string", "description": "The complete file content to write"}
        },
        "required": ["path", "content"]
    },
    {
        "name": "search_code",
        "description": "Search for a string or regex pattern in the codebase.",
        "properties": {
            "query": {"type": "string", "description": "The text pattern to search for"}
        },
        "required": ["query"]
    },
    {
        "name": "find_file",
        "description": "Find files matching a glob pattern.",
        "properties": {
            "pattern": {"type": "string", "description": "The glob pattern to search for, e.g. '*.py'"}
        },
        "required": ["pattern"]
    },
    {
        "name": "read_file_chunk",
        "description": "Read the contents of a file along with line numbers.",
        "properties": {
            "filepath": {"type": "string", "description": "Relative path to the file to open"},
            "start_line": {"type": "integer", "description": "Optional: Start line number (1-indexed)"},
            "end_line": {"type": "integer", "description": "Optional: End line number"}
        },
        "required": ["filepath"]
    },
    {
        "name": "list_directory",
        "description": "List the contents of a directory using ls -la.",
        "properties": {
            "dir_path": {"type": "string", "description": "Directory path to list, defaults to '.'"}
        },
        "required": [] # list_directory technically might not strictly require it if default is used, but good to define
    },
    {
        "name": "run_bash_command",
        "description": "Execute a terminal command. Only use this for reading status, logs, or debugging outputs.",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute."}
        },
        "required": ["command"]
    },
    {
        "name": "web_search",
        "description": "Search the internet. Use categories to route to specific reliable sources.",
        "properties": {
            "query": {"type": "string", "description": "The exact search query."},
            "category": {
                "type": "string", 
                "enum": ["general", "news", "code", "academic"],
                "description": "Choose 'news' for current events, 'code' for github/docs, 'academic' for papers. Default to 'general'."
            }
        },
        "required": ["query"]
    },
    {
        "name": "read_url",
        "description": "Fetch and read the full text content of a specific webpage URL. Use this after web_search if you need more details from a specific source.",
        "properties": {
            "url": {"type": "string", "description": "The exact URL to fetch (must start with http:// or https://)."}
        },
        "required": ["url"]
    }
]

# ==========================================
# 2. Dynamic Tool Compiler
# ==========================================
def get_compiled_tools(provider: str) -> List[Dict[str, Any]]:
    """
    Compiles the BASE_TOOLS into the specific format required by the LLM provider.
    """
    compiled = []
    for tool in BASE_TOOLS:
        if provider == "anthropic":
            # Anthropic Format
            compiled.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["properties"],
                    "required": tool.get("required", [])
                }
            })
        else:
            # OpenAI / vLLM Format
            compiled.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["properties"],
                        "required": tool.get("required", [])
                    }
                }
            })
    return compiled

# ==========================================
# Helper Modules for LLM Interaction
# ==========================================

def _detect_repetition(text: str, tail_lines: int = 30) -> bool:
    """Detects infinite loops in LLM generation tail."""
    if not text or len(text) < 200: return False
    lines = text.splitlines()
    if len(lines) < tail_lines: return False
    
    tail = lines[-tail_lines:]
    tail_str = "\n".join(tail)
    
    if len(tail) >= 12:
        chunk = "\n".join(tail[-5:-1])
        if chunk.strip() and tail_str.count(chunk) >= 3:
            return True
    return False

def _stitch_text(full_content: str, new_content: str) -> str:
    """Cleans up conversational filler when stitching continuations."""
    original_len = len(new_content)
    is_inside_code = (full_content.count("```") % 2 == 1)
    is_inside_write_file = (len(re.findall(r'<<<CONTENT', full_content)) > len(re.findall(r'CONTENT>{2,3}', full_content)))
    
    if is_inside_code or is_inside_write_file:
        new_content = re.sub(r'^\s*```\w*\n', '', new_content)
        if not new_content.strip().startswith(('#', 'def ', 'class ', 'print', 'import ', '<')):
            new_content = re.sub(r'^(Here is the rest.*?|Sure.*?|Continuing.*?)\n', '', new_content, flags=re.IGNORECASE)
            
    if is_inside_code and new_content.lstrip().startswith("## "):
        console.print("[red]Detected hallucinated header in code block. Truncating.[/red]")
        new_content = new_content.split("## ")[0]

    return new_content

def _parse_native_dict_to_action(name: str, args_dict: dict, allowlist: List[str]) -> AgentAction:
    """Translates a native parsed tool dictionary into our AgentAction protocol."""
    if name == "write_file":
        return ActionWriteFile(path=args_dict.get("path", ""), content=args_dict.get("content", ""))
    elif name == "search_and_replace":
        return ActionReplaceText(
            path=args_dict.get("path", ""), 
            old_text=args_dict.get("old_text", ""), 
            new_text=args_dict.get("new_text", "")
        )
    else:
        # Generic fallback for web_search, search_code, run_bash_command, etc.
        return ActionToolCall(name=name, args=args_dict)

# ==========================================
# Network / Execution Layers
# ==========================================

async def _execute_openai_async(
    client: Any, model: str, messages: List[Dict[str, str]], 
    temperature: float, max_tokens: int, stream: bool,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    Handles OpenAI/vLLM format. 
    Includes complex state machine for reconstructing streamed tool calls.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools: kwargs["tools"] = tools
    if stream: kwargs["stream_options"] = {"include_usage": True}
        
    resp = await client.chat.completions.create(**kwargs)
    
    content = ""
    finish_reason = "stop"
    usage_info = {}
    native_tool_calls = [] # Structured as: [{"name": str, "arguments": str}]
    
    if stream:
        # State machine for streaming tool calls
        tc_dict = {}
        async for chunk in resp:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = {"prompt_tokens": chunk.usage.prompt_tokens, "completion_tokens": chunk.usage.completion_tokens}
                continue
                
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
                
            # Handle streaming tool calls (OpenAI yields chunks of JSON arguments)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_dict:
                        tc_dict[idx] = {"name": tc.function.name, "arguments": ""}
                    if tc.function.arguments:
                        tc_dict[idx]["arguments"] += tc.function.arguments
                        
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                
        # Convert stitched dictionary to list
        native_tool_calls = list(tc_dict.values())
        
    else:
        # Non-streaming parsing
        msg = resp.choices[0].message
        content = msg.content or ""
        finish_reason = resp.choices[0].finish_reason
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            
        if msg.tool_calls:
            for tc in msg.tool_calls:
                native_tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments})
                
    return content, finish_reason, usage_info, native_tool_calls

async def _execute_anthropic_async(
    client: Any, model: str, messages: List[Dict[str, str]], 
    temperature: float, max_tokens: int, 
    tools: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """Handles Anthropic specific API formatting with native tool calls."""
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr_msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} 
                for m in messages if m["role"] != "system"]
    
    if usr_msgs: usr_msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    sys_msg_blocks = [{"type": "text", "text": sys_msg, "cache_control": {"type": "ephemeral"}}]
    
    config_kwargs = {
        "model": model, "system": sys_msg_blocks, "messages": usr_msgs,
        "temperature": temperature, "max_tokens": max_tokens
    }
    if tools: config_kwargs["tools"] = tools
    
    resp = await client.messages.create(**config_kwargs)
    
    content = ""
    native_tool_calls = []
    
    for block in resp.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            # Anthropic returns parsed dict directly instead of JSON string
            native_tool_calls.append({"name": block.name, "arguments": json.dumps(block.input)})
            
    finish_reason = "length" if resp.stop_reason == "max_tokens" else ("tool_calls" if resp.stop_reason == "tool_use" else str(resp.stop_reason))
    usage_info = {"prompt_tokens": resp.usage.input_tokens, "completion_tokens": resp.usage.output_tokens} if hasattr(resp, 'usage') else {}
    
    return content, finish_reason, usage_info, native_tool_calls

# ==========================================
# Core Wrapper Function
# ==========================================

async def complete_with_continuation_async(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    model_max_context: int = 16384,
    provider: str = "openai",
    stream: bool = True,
    session_dir: Optional[Path] = None,
    tools: Optional[List[Dict[str, Any]]] = None, # Unified Tools Schema
    tool_strategy: str = "auto", # ['native', 'text', 'auto']
    allowlist: Optional[List[str]] = None
) -> Tuple[str, List[AgentAction]]:
    """
    Supercharged Wrapper supporting Native JSON Function Calling, Text-based Fallback,
    Concurrency, and Anti-Looping.
    """
    full_content = ""
    current_messages = list(messages)
    max_loops = 5
    allowlist = allowlist or []
    final_actions: List[AgentAction] = []
    
    # Decide if we inject tools into the API payload based on strategy
    active_tools = tools if tool_strategy in ["native", "auto"] else None
    
    for i in range(max_loops):
        console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
        # 1. Adaptive Token Compression
        input_text = "\n".join(m.get("content", "") for m in current_messages)
        input_est = estimate_tokens(input_text)
        min_output = 1024
        max_allowed_input = model_max_context - 1000 - min_output
        
        if int(input_est * 1.1) > max_allowed_input > 0:
            console.print(f"[yellow]Compressing messages (est {input_est} > limit).[/yellow]")
            current_messages = compress_messages(current_messages, max_allowed_tokens=int(max_allowed_input / 1.1))
            input_est = estimate_tokens("\n".join(m.get("content", "") for m in current_messages))

        safe_tokens = compute_safe_max_tokens(input_est, model_max_context, max_output_tokens, min_output)

        # 2. API Call with Retries
        content, finish_reason, usage_info, native_tcs = "", "stop", {}, []
        start_time = time.time()
        
        for attempt in range(3):
            try:
                if provider == "anthropic":
                    content, finish_reason, usage_info, native_tcs = await _execute_anthropic_async(
                        client, model, current_messages, temperature, safe_tokens, active_tools
                    )
                else:
                    content, finish_reason, usage_info, native_tcs = await _execute_openai_async(
                        client, model, current_messages, temperature, safe_tokens, stream, active_tools
                    )
                break
            except Exception as e:
                err_str = str(e)
                if 'max_tokens' in err_str or 'context length' in err_str:
                    safe_tokens = max(1024, safe_tokens // 2)
                    console.print(f"[red]Context overflow. Retrying max_tokens={safe_tokens}[/red]")
                    await asyncio.sleep(1)
                    continue
                console.print(f"[red]LLM Call failed (attempt {attempt+1}): {e}[/red]")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                # Hard fail, parse what we have and abort
                return full_content, parse_text_actions(full_content, allowlist)

        # 3. Metrics
        elapsed = time.time() - start_time
        if not usage_info:
            usage_info = {"prompt_tokens": input_est, "completion_tokens": estimate_tokens(content)}
        
        tok_speed = usage_info["completion_tokens"] / elapsed if elapsed > 0 else 0
        console.print(f"[bold blue][LLM][/bold blue] [dim]{usage_info['prompt_tokens']}P, {usage_info['completion_tokens']}C | {tok_speed:.1f} T/s | {elapsed:.1f}s[/dim]")

        # 4. Stitching & Guardrails
        if i > 0: content = _stitch_text(full_content, content)
        full_content += content
        
        if _detect_repetition(full_content):
            console.print("[bold red]Repetition loop detected! Fusing circuit breaker.[/bold red]")
            break
            
        # 5. Process Native Tool Calls (If Any)
        if native_tcs:
            for tc in native_tcs:
                try:
                    args_dict = json.loads(tc["arguments"])
                    action = _parse_native_dict_to_action(tc["name"], args_dict, allowlist)
                    final_actions.append(action)
                except json.JSONDecodeError:
                    console.print(f"[red]Failed to parse native tool JSON: {tc['arguments']}[/red]")
            
            # If we received native tool calls, we consider the turn complete (no need to parse text)
            if final_actions:
                break

        # 6. Continuation Check
        if finish_reason == "length":
            console.print("[yellow]Output truncated. Continuing automatically...[/yellow]")
            current_messages.append({"role": "assistant", "content": content})
            current_messages.append({
                "role": "user", 
                "content": "You were cut off. IMMEDIATELY continue exactly where you left off. DO NOT repeat the last line."
            })
        else:
            break

    # ==========================================
    # Multi-tier Parsing Strategy
    # ==========================================
    # If we didn't capture any native actions, OR strategy explicitly forced 'text', apply Fallback
    if not final_actions and tool_strategy in ["auto", "text"]:
        console.print("[dim]No native tool calls detected. Applying Text-based parsing fallback...[/dim]")
        final_actions = parse_text_actions(full_content, allowlist)
        
    return full_content, final_actions
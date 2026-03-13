import time
import re
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
from pathlib import Path
from rich.console import Console
import sys
from collections import Counter
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


console = Console()

from BatchAgent.mini_batch_agent_libs import (
    estimate_tokens, compress_messages, compute_safe_max_tokens, now_stamp, write_jsonl, robust_json_loads
)
from BatchAgent.mini_batch_agent import AgentAction, ActionWriteFile, ActionReplaceText, ActionToolCall #, parse_text_actions
from BatchAgent.tool_handler import parse_text_actions
from typing import List, Dict, Any

# ==========================================
# 1. Base Tool Definitions (Single Source of Truth)
# ==========================================
# BASE_TOOLS = [
#     # {
#     #     "name": "write_file",
#     #     "description": "Create a new file or completely overwrite an existing file with new content. Use this for new files or when changes are too complex for search_and_replace.",
#     #     "properties": {
#     #         "path": {"type": "string", "description": "Relative path to the file"},
#     #         "content": {"type": "string", "description": "The complete file content to write"}
#     #     },
#     #     "required": ["path", "content"]
#     # },
#     {
#         "name": "search_code",
#         "description": "Search for a string or regex pattern in the codebase.",
#         "properties": {
#             "query": {"type": "string", "description": "The text pattern to search for"}
#         },
#         "required": ["query"]
#     },
#     {
#         "name": "find_file",
#         "description": "Find files matching a glob pattern.",
#         "properties": {
#             "pattern": {"type": "string", "description": "The glob pattern to search for, e.g. '*.py'"}
#         },
#         "required": ["pattern"]
#     },
#     {
#         "name": "read_file_chunk",
#         "description": "Read the contents of a file along with line numbers.",
#         "properties": {
#             "filepath": {"type": "string", "description": "Relative path to the file to open"},
#             "start_line": {"type": "integer", "description": "Optional: Start line number (1-indexed)"},
#             "end_line": {"type": "integer", "description": "Optional: End line number"}
#         },
#         "required": ["filepath"]
#     },
#     {
#         "name": "list_directory",
#         "description": "List the contents of a directory using ls -la.",
#         "properties": {
#             "dir_path": {"type": "string", "description": "Directory path to list, defaults to '.'"}
#         },
#         "required": [] # list_directory technically might not strictly require it if default is used, but good to define
#     },
#     {
#         "name": "run_bash_command",
#         "description": "Execute a terminal command. Only use this for reading status, logs, or debugging outputs.",
#         "properties": {
#             "command": {"type": "string", "description": "The bash command to execute."}
#         },
#         "required": ["command"]
#     },
#     {
#         "name": "web_search",
#         "description": "Search the internet. Use categories to route to specific reliable sources.",
#         "properties": {
#             "query": {"type": "string", "description": "The exact search query."},
#             "category": {
#                 "type": "string", 
#                 "enum": ["general", "news", "code", "academic"],
#                 "description": "Choose 'news' for current events, 'code' for github/docs, 'academic' for papers. Default to 'general'."
#             }
#         },
#         "required": ["query"]
#     },
#     {
#         "name": "read_url",
#         "description": "Fetch and read the full text content of a specific webpage URL. Use this after web_search if you need more details from a specific source.",
#         "properties": {
#             "url": {"type": "string", "description": "The exact URL to fetch (must start with http:// or https://)."}
#         },
#         "required": ["url"]
#     },
#     {
#         "name": "submit_plan",
#         "description": "Break down a complex task into smaller subtasks. The framework will orchestrate and potentially parallelize them.",
#         "properties": {
#             "subtasks": {
#                 "type": "array",
#                 "description": "List of subtasks.",
#                 "items": {
#                     "type": "object",
#                     "properties": {
#                         "task_id": {"type": "string", "description": "Unique ID, e.g., 'task_1'"},
#                         "description": {"type": "string", "description": "Detailed instruction for this subtask"},
#                         "depends_on": {
#                             "type": "array", 
#                             "items": {"type": "string"}, 
#                             "description": "List of task_ids this subtask depends on. Empty if it can run immediately."
#                         }
#                     },
#                     "required": ["task_id", "description", "depends_on"]
#                 }
#             }
#         },
#         "required": ["subtasks"]
#     },
#     {
#         "name": "finish_task",
#         "description": "Call this tool ONLY when you have fully completed the user's goal and verified the results.",
#         "properties": {
#             "summary": {"type": "string", "description": "A brief summary of what was accomplished."}
#         },
#         "required": ["summary"]
#     }
# ]

# # ==========================================
# # 2. Dynamic Tool Compiler
# # ==========================================
# def get_compiled_tools(provider: str) -> List[Dict[str, Any]]:
#     """
#     Compiles the BASE_TOOLS into the specific format required by the LLM provider.
#     """
#     compiled = []
#     for tool in BASE_TOOLS:
#         if provider == "anthropic":
#             # Anthropic Format
#             compiled.append({
#                 "name": tool["name"],
#                 "description": tool["description"],
#                 "input_schema": {
#                     "type": "object",
#                     "properties": tool["properties"],
#                     "required": tool.get("required", [])
#                 }
#             })
#         else:
#             # OpenAI / vLLM Format
#             compiled.append({
#                 "type": "function",
#                 "function": {
#                     "name": tool["name"],
#                     "description": tool["description"],
#                     "parameters": {
#                         "type": "object",
#                         "properties": tool["properties"],
#                         "required": tool.get("required", [])
#                     }
#                 }
#             })
#     return compiled

# ==========================================
# Helper Modules for LLM Interaction
# ==========================================
def _detect_repetition(text: str, window_size: int = 50, threshold: int = 4) -> bool:
    """
    Advanced Repetition Detector.
    Catches exact looping phrases AND structural loops (like incrementing numbers in the same sentence).
    """
    if len(text) < 500:
        return False

    # 1. 检测最常见的“尾部文字无限循环”
    # 取最后的 500 个字符，按行分割
    tail_lines = text[-500:].strip().split('\n')
    if len(tail_lines) >= 4:
        # 如果最后 4 行完全一模一样
        if len(set(tail_lines[-4:])) == 1 and len(tail_lines[-1]) > 5:
            return True

    # 2. 检测“模式重复”（结构一样，只有数字不同）
    # 把文本中的数字全部替换为占位符 <NUM>
    normalized_text = re.sub(r'\d+', '<NUM>', text)
    
    # 将标准化的文本分成 Token (这里为了简单，用空格分词)
    tokens = normalized_text.split()
    
    if len(tokens) < window_size * 2:
        return False
        
    # 使用 N-Gram 统计频率 (N = 15 words)
    n = 15
    if len(tokens) < n:
        return False
        
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n)]
    ngram_counts = Counter(ngrams)
    
    # 如果某一段 15 个词的结构在文本中重复出现了超过 5 次，必定是模型退化了！
    most_common = ngram_counts.most_common(1)
    if most_common and most_common[0][1] > 5:
        # Debug 打印，让你知道它因为什么被熔断了
        print(f"\n[bold red]⚠️ Repetition Circuit Breaker Fused![/bold red]")
        print(f"[dim]Detected repeating pattern: '{most_common[0][0]}' ({most_common[0][1]} times)[/dim]")
        return True
        
    return False

# deprecated    
def _detect_repetition_v1(text: str, tail_lines: int = 30) -> bool:
    """Detects infinite loops in LLM generation tail, ignoring list numbers."""
    if not text or len(text) < 200: return False
    lines = text.splitlines()
    if len(lines) < tail_lines: return False
    
    tail = lines[-tail_lines:]
    
    # [NEW] Remove leading numbers (e.g. "15. ", "- ") for pattern matching
    clean_tail = [re.sub(r'^(\d+\.|-|\*)\s*', '', line).strip() for line in tail if line.strip()]
    
    if len(clean_tail) >= 12:
        # Check if the last 4 clean lines appear multiple times
        chunk = "\n".join(clean_tail[-5:-1])
        clean_tail_str = "\n".join(clean_tail)
        if chunk.strip() and clean_tail_str.count(chunk) >= 3:
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
import sys
import json
from typing import Any, List, Dict, Tuple, Optional

async def _execute_openai_async(
    client: Any, model: str, messages: List[Dict[str, str]], 
    temperature: float, max_tokens: int, stream: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    
    kwargs = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": stream,
    }
    
    # ── Inject specific parameters for vllm/llama.cpp ──
    if backend in ["llama.cpp", "vllm"] and enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        kwargs["stop"] = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
        
    if tools: kwargs["tools"] = tools
    if stream: kwargs["stream_options"] = {"include_usage": True}
        
    resp = await client.chat.completions.create(**kwargs)
    
    content = ""
    finish_reason = "stop"
    usage_info = {}
    native_tool_calls = []
    tc_dict = {}
    
    if stream:
        chunk_counter = 0
        
        # State machine for suppressing XML tool prints & handling thinking
        in_think = False
        in_tool = False
        buffer = ""
        tool_args_buffer = ""
        tool_name_buffer = ""
        
        async for chunk in resp:
            chunk_counter += 1
            
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = {"prompt_tokens": chunk.usage.prompt_tokens, "completion_tokens": chunk.usage.completion_tokens}
                continue
                
            delta = chunk.choices[0].delta
            
            # --- 1. Extract Llama.cpp / Custom backends reasonings ---
            reasoning = delta.model_dump().get("reasoning_content")
            if reasoning:
                if verbose:
                    sys.stdout.write(f"\033[90m{reasoning}\033[0m")
                    sys.stdout.flush()
                if on_event:
                    await on_event({"type": "think", "data": reasoning})
            
            # --- 2. Extract standard content ---
            if delta.content:
                text_chunk = delta.content
                content += text_chunk
                buffer += text_chunk
                
                # Tag detector
                if not in_think and "<think>" in buffer:
                    in_think = True
                    buffer = buffer.split("<think>")[-1]
                    
                if in_think and "</think>" in buffer:
                    in_think = False
                    buffer = buffer.split("</think>")[-1]
                    
                if not in_tool and "<tool_call>" in buffer:
                    in_tool = True
                    buffer = buffer.split("<tool_call>")[-1]
                    
                if in_tool and "</tool_call>" in buffer:
                    in_tool = False
                    buffer = ""
                    tool_args_buffer = ""
                    tool_name_buffer = ""
                
                # Strip XML control tags from the visible text before emitting
                clean_chunk = re.sub(r'</?think>|</?tool_call>', '', text_chunk)

                # Text processing logic
                if in_think:
                    if verbose:
                        sys.stdout.write(f"\033[90m{clean_chunk}\033[0m")
                        sys.stdout.flush()
                    if on_event and clean_chunk:
                        await on_event({"type": "think", "data": clean_chunk})
                elif in_tool:
                    # Accumulate tool text silently
                    tool_args_buffer += text_chunk

                    # Try to extract the tool name if we haven't yet (simple regex logic)
                    if not tool_name_buffer and ">" in tool_args_buffer:
                        # e.g., <web_search> or <write_file>
                        match = re.search(r"<([a-zA-Z0-9_]+)>", tool_args_buffer)
                        if match:
                            tool_name_buffer = match.group(1)
                            if verbose:
                                console.print(f"\n[bold magenta]🛠️ Parsing Tool: {tool_name_buffer}...[/bold magenta]")

                    if on_event:
                        await on_event({"type": "tool", "status": "streaming", "data": text_chunk})
                else:
                    # Normal message token — strip any stray tag fragments before emitting
                    if clean_chunk:
                        if verbose:
                            sys.stdout.write(clean_chunk)
                            sys.stdout.flush()
                        if on_event:
                            await on_event({"type": "message", "data": clean_chunk})
                
                # =================================================================
                # [NEW] 实时流式熔断 (Real-time Stream Circuit Breaker)
                # 每收到 40 个字块，且总长度大于 500 时，抽样检测一次是否发疯
                # =================================================================
                if chunk_counter % 40 == 0 and len(content) > 500:
                    if _detect_repetition(content):
                        if verbose:
                            print("\n\n[bold red]⚠️ [Stream Interrupted] Repetition Loop Detected! Connection severed.[/bold red]")
                        finish_reason = "repetition"  # 标记特殊结束原因
                        break # 强行跳出 async for，中断网络流，省时省钱！
                        
            # (后续解析 Tool Call 的逻辑保持不变)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_dict:
                        func_name = tc.function.name if tc.function and tc.function.name else "unknown_tool"
                        tc_dict[idx] = {"name": func_name, "arguments": ""}
                        if on_event:
                            await on_event({"type": "tool", "name": func_name, "status": "started"})
                        if verbose:
                            console.print(f"\n[bold magenta]🛠️ Calling Tool: {func_name}...[/bold magenta]")
                    
                    if tc.function and tc.function.arguments:
                        chunk_arg = tc.function.arguments
                        tc_dict[idx]["arguments"] += chunk_arg
                        if on_event:
                            await on_event({"type": "tool", "name": tc_dict[idx]["name"], "args_delta": chunk_arg, "status": "streaming"})
                            
                # Optionally print fully completed arguments when finish_reason hits, but let's wait until outside the loop for summary if needed.
                        
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
                
        if verbose and content and finish_reason != "repetition":
            print()
            
        native_tool_calls = list(tc_dict.values())
        
    else:
        # Non-streaming parsing
        msg = resp.choices[0].message
        content = msg.content or ""
        finish_reason = resp.choices[0].finish_reason or "stop"
        
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {
                "prompt_tokens": resp.usage.prompt_tokens, 
                "completion_tokens": resp.usage.completion_tokens
            }
            
        if msg.tool_calls:
            for tc in msg.tool_calls:
                native_tool_calls.append({
                    "name": tc.function.name, 
                    "arguments": tc.function.arguments
                })
                
    return content, finish_reason, usage_info, native_tool_calls

async def _execute_anthropic_async(
    client: Any, model: str, messages: List[Dict[str, str]], 
    temperature: float, max_tokens: int, 
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False 
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    Handles Anthropic specific API formatting with native tool calls.
    Utilizes Ephemeral Caching for massive cost reduction on large contexts.
    """
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    
    # assemble User/Assistant Messages
    usr_msgs = []
    for m in messages:
        if m["role"] != "system":
            # Anthropic differentiate user and assistant
            role = "assistant" if m["role"] == "assistant" else "user"
            usr_msgs.append({"role": role, "content": [{"type": "text", "text": m["content"]}]})
    
    # Apply Prompt Caching (cache last User message)
    if usr_msgs and usr_msgs[-1]["role"] == "user": 
        usr_msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        
    sys_msg_blocks = [{"type": "text", "text": sys_msg, "cache_control": {"type": "ephemeral"}}]
    
    config_kwargs = {
        "model": model, 
        "system": sys_msg_blocks, 
        "messages": usr_msgs,
        "temperature": temperature, 
        "max_tokens": max_tokens
    }
    if tools: config_kwargs["tools"] = tools
    
    # used the standard create message endpoint for better streaming support, even though it doesn't natively support streaming, we can still parse the content and tool calls in a streaming-like fashion by processing the response blocks as they come in.
    resp = await client.messages.create(**config_kwargs)
    
    content = ""
    native_tool_calls = []
    
    # parse the content and tool calls from the response, stitching tool calls if they come in multiple chunks
    for block in resp.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            # Anthropic return dict，we need to serialize it to string for uniformity with OpenAI's tool_calls
            native_tool_calls.append({
                "name": block.name, 
                "arguments": json.dumps(block.input)
            })
            
    # (OpenAI format mapping)
    if resp.stop_reason == "max_tokens":
        finish_reason = "length"
    elif resp.stop_reason == "tool_use":
        finish_reason = "tool_calls"
    else:
        finish_reason = "stop"
        
    usage_info = {
        "prompt_tokens": getattr(resp.usage, "input_tokens", 0), 
        "completion_tokens": getattr(resp.usage, "output_tokens", 0)
    } if hasattr(resp, 'usage') else {}
    
    if verbose and content:
        print(content) #print the full content at the end for Anthropic, since it doesn't support true streaming
    
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
    stream: bool = True,           # Enable/Disable Streaming API
    verbose: bool = False,         # Enable/Disable terminal printing
    session_dir: Optional[Path] = None,
    tools: Optional[List[Dict[str, Any]]] = None, # Unified Tools Schema
    tool_strategy: str = "auto",   # ['native', 'text', 'auto']
    allowlist: Optional[List[str]] = None,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,  # SSE/streaming callback
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, List[AgentAction]]:
    """
    Supercharged Wrapper supporting Native JSON Function Calling, Text-based Fallback,
    Concurrency, Streaming, Anti-Looping, and Robust Continuation Sanitization.
    """
    full_content = ""
    current_messages = list(messages)
    max_loops = 5
    allowlist = allowlist or []
    final_actions: List[AgentAction] = []
    
    # Decide if we inject tools into the API payload based on strategy
    #active_tools = tools if tool_strategy in ["native", "auto"] else None
    active_tools = tools if tool_strategy in ["native_all", "auto", "hybrid"] else None
    
    for i in range(max_loops):
        if i > 0: console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
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
                # IMPORTANT: Your underlying _execute functions should respect the `stream` 
                # and `verbose` flags to print token-by-token to the terminal if requested.
                if provider == "anthropic":
                    content, finish_reason, usage_info, native_tcs = await _execute_anthropic_async(
                        client, model, current_messages, temperature, safe_tokens, active_tools, verbose
                    )
                else:
                    content, finish_reason, usage_info, native_tcs = await _execute_openai_async(
                        client, model, current_messages, temperature, safe_tokens, stream, active_tools, verbose,
                        on_event=on_event, backend=backend, enable_thinking=enable_thinking
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

        if on_event and usage_info:
            await on_event({
                "type": "usage",
                "prompt_tokens": usage_info.get("prompt_tokens", 0),
                "completion_tokens": usage_info.get("completion_tokens", 0),
                "elapsed_s": round(elapsed, 2),
            })

        # 4. Continuation Sanitization & Stitching
        if i > 0: 
            # --- [FIX 1: THE CONTINUATION SANITIZER] ---
            # Reasoning models (like Qwen-Reasoning) will forcefully insert <think> blocks 
            # or markdown fences when asked to continue. We must amputate them to avoid SyntaxErrors.
            
            # Amputate full reasoning blocks
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            clean_content = clean_content.lstrip()
            
            # Amputate markdown fences at the start of the continuation
            if clean_content.startswith("```python"):
                clean_content = clean_content[9:].lstrip('\n')
            elif clean_content.startswith("```"):
                clean_content = clean_content[3:].lstrip('\n')
                
            content = clean_content
            # Apply traditional stitch to fix overlapping words
            content = _stitch_text(full_content, content)
            
        full_content += content
        
        # Guardrail: Circuit Breaker for repetitive generation loops
        # if _detect_repetition(full_content):
        #     console.print("[bold red]Repetition loop detected! Fusing circuit breaker.[/bold red]")
        #     break
        # --- [FIX: 如果检测到发疯，直接熔断并抛弃最后的狂暴内容] ---
        if _detect_repetition(full_content):
            console.print("[bold red]Repetition loop detected! Fusing circuit breaker.[/bold red]")
            # 稍微往回退一点，把最近生成的几百个字符切掉（因为通常检测到的时候，尾巴已经是垃圾了）
            safe_length = max(0, len(full_content) - 1000)
            full_content = full_content[:safe_length] + "\n\n[SYSTEM: OUTPUT TRUNCATED DUE TO REPETITION LOOP]"
            break
            
        # 5. Process Native Tool Calls (If Any)
        if native_tcs:
            for tc in native_tcs:
                tool_name = tc.get("name", "")
                
                # --- [FIX 2: ROBUST JSON PARSING & ERROR INJECTION] ---
                # We use the robust JSON parser which includes json-repair
                args_dict = robust_json_loads(tc["arguments"], tool_name=tool_name)
                
                if args_dict is not None:
                    try:
                        action = _parse_native_dict_to_action(tool_name, args_dict, allowlist)
                        final_actions.append(action)
                    except Exception as e:
                        console.print(f"[red]Error mapping JSON to action for {tool_name}: {e}[/red]")
                else:
                    # Inject a hallucinated error action to force the LLM to realize its JSON broke
                    console.print(f"[bold red]Failed to parse native tool JSON completely for {tool_name}.[/bold red]")
                    final_actions.append(ActionToolCall(
                        name="json_parse_error", 
                        args={
                            "error": f"The JSON tool call for '{tool_name}' was malformed or improperly escaped (e.g., unescaped quotes or newlines). "
                                     f"Please try again, or use the Format B (WRITE_FILE) markdown text format instead."
                        }
                    ))
            
            # If we received ANY native tool actions (even error injections), we complete the turn
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
    # Apply text-based parsing for any strategy that isn't exclusively native-JSON.
    # 'text_only' : LLM outputs XML tags, must parse — previously broken (checked for "text" not "text_only")
    # 'hybrid'    : may mix XML tool tags with native JSON
    # 'auto'/'text': older aliases
    # 'native_all': already parsed above via native_tcs; skip text parsing
    if not final_actions and tool_strategy not in ("native_all",):
        if verbose:
            console.print("[dim]No native tool calls detected. Applying text-based parsing fallback...[/dim]")
        final_actions = parse_text_actions(full_content, allowlist)

    return full_content, final_actions


# ==========================================
# Simple Single-Shot Async Wrapper (No Continuation)
# ==========================================

async def complete_with_async(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    model_max_context: int = 16384,
    provider: str = "openai",
    stream: bool = True,
    verbose: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Simple single-shot async wrapper for LLM API calls.

    Unlike complete_with_continuation_async, this function:
    - Does NOT loop on finish_reason == 'length' (no auto-continuation)
    - Does NOT parse tool calls or agent actions
    - Returns (content: str, usage_info: dict)

    The optional `on_token` async callback receives each streamed token as it
    arrives, enabling FastAPI SSE endpoints to forward tokens to clients in
    real-time without any extra buffering.

    Args:
        client:            Async OpenAI-compatible client (or Anthropic client)
        model:             Model name
        messages:          List of chat messages (role/content dicts)
        temperature:       Sampling temperature
        max_output_tokens: Maximum completion tokens to request
        model_max_context: Total context window size (for token budget calc)
        provider:          'openai' or 'anthropic'
        stream:            Enable streaming API (token-by-token)
        verbose:           Print tokens to terminal as they arrive
        on_token:          Optional async callback called for each streamed token.
                           Signature: async def on_token(token: str) -> None

    Returns:
        (content, usage_info)  where usage_info has 'prompt_tokens',
        'completion_tokens', 'elapsed_seconds', 'tokens_per_second',
        and 'finish_reason' keys.
    """
    # --- 1. Adaptive Token Budget ---
    input_text = "\n".join(m.get("content", "") for m in messages)
    input_est = estimate_tokens(input_text)
    min_output = 256
    max_allowed_input = model_max_context - 1000 - min_output

    if int(input_est * 1.1) > max_allowed_input > 0:
        console.print(f"[yellow]Compressing messages (est {input_est} > limit).[/yellow]")
        messages = compress_messages(messages, max_allowed_tokens=int(max_allowed_input / 1.1))
        input_est = estimate_tokens("\n".join(m.get("content", "") for m in messages))

    safe_tokens = compute_safe_max_tokens(input_est, model_max_context, max_output_tokens, min_output)

    # --- 2. Single API Call with Retries ---
    content: str = ""
    finish_reason: str = "stop"
    usage_info: Dict[str, Any] = {}
    start_time = time.time()

    for attempt in range(3):
        try:
            if provider == "anthropic":
                content, finish_reason, usage_info, _ = await _execute_anthropic_async(
                    client, model, messages, temperature, safe_tokens,
                    tools=None, verbose=verbose
                )
            else:
                content, finish_reason, usage_info, _ = await _execute_openai_async(
                    client, model, messages, temperature, safe_tokens,
                    stream=stream, tools=None, verbose=verbose, on_event=on_event,
                    backend=backend, enable_thinking=enable_thinking
                )
            break
        except Exception as e:
            err_str = str(e)
            if "max_tokens" in err_str or "context length" in err_str:
                safe_tokens = max(1024, safe_tokens // 2)
                console.print(f"[red]Context overflow. Retrying max_tokens={safe_tokens}[/red]")
                await asyncio.sleep(1)
                continue
            console.print(f"[red]LLM Call failed (attempt {attempt + 1}): {e}[/red]")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            break  # Hard fail – return whatever we have

    # --- 3. Metrics ---
    elapsed = time.time() - start_time
    if not usage_info:
        usage_info = {
            "prompt_tokens": input_est,
            "completion_tokens": estimate_tokens(content),
        }

    tok_speed = usage_info["completion_tokens"] / elapsed if elapsed > 0 else 0
    console.print(
        f"[bold blue][LLM][/bold blue] [dim]"
        f"{usage_info['prompt_tokens']}P, {usage_info['completion_tokens']}C | "
        f"{tok_speed:.1f} T/s | {elapsed:.1f}s | finish={finish_reason}[/dim]"
    )

    usage_info["elapsed_seconds"] = round(elapsed, 2)
    usage_info["tokens_per_second"] = round(tok_speed, 1)
    usage_info["finish_reason"] = finish_reason

    return content, usage_info
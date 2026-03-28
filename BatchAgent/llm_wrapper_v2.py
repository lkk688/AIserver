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


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM backend is unreachable after all retries."""

from BatchAgent.mini_batch_agent_libs import (
    estimate_tokens, compress_messages, compute_safe_max_tokens, now_stamp, write_jsonl, robust_json_loads
)
from BatchAgent.mini_batch_agent_base import AgentAction, ActionWriteFile, ActionReplaceText, ActionToolCall

# Import the new decoupled parser
from BatchAgent.tools.text_action_parser import parse_text_actions

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

    # 1. Detect infinite tail loops (last 4 lines are identical)
    tail_lines = text[-500:].strip().split('\n')
    if len(tail_lines) >= 4:
        if len(set(tail_lines[-4:])) == 1 and len(tail_lines[-1]) > 5:
            return True

    # 2. Detect structural repetition using N-Grams
    normalized_text = re.sub(r'\d+', '<NUM>', text)
    tokens = normalized_text.split()
    
    if len(tokens) < window_size * 2:
        return False
        
    n = 15
    if len(tokens) < n:
        return False
        
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n)]
    ngram_counts = Counter(ngrams)
    
    # If a specific 15-word phrase repeats more than 5 times, it's a loop
    most_common = ngram_counts.most_common(1)
    if most_common and most_common[0][1] > 5:
        print(f"\n[bold red]⚠️ Repetition Circuit Breaker Fused![/bold red]")
        print(f"[dim]Detected repeating pattern: '{most_common[0][0]}' ({most_common[0][1]} times)[/dim]")
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


def _has_unclosed_tool_markup(text: str) -> bool:
    """Determines if a tool call was cut off in the middle of being streamed."""
    lowered = (text or "").lower()
    if lowered.count("<tool_call>") > lowered.count("</tool_call>"):
        return True
    tool_tags = ("write_file", "search_and_replace", "finish_task")
    for tag in tool_tags:
        if lowered.count(f"<{tag}>") > lowered.count(f"</{tag}>"):
            return True
    return False


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
    tools: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    backend: str = "openai",
    enable_thinking: bool = True,
) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    Executes a standard OpenAI-compatible API call, capturing tool streams and <think> blocks.
    """
    kwargs = {
        "model": model, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": stream,
    }
    
    # Enable reasoning features for vLLM and llama.cpp backends
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
            
            # Extract reasoning out of API properties if supported natively
            reasoning = delta.model_dump().get("reasoning_content")
            if reasoning:
                if verbose:
                    console.print(reasoning, end="", style="dim", highlight=False, markup=False)
                if on_event:
                    await on_event({"type": "think", "data": reasoning})
            
            # Standard chunk body processing
            if delta.content:
                text_chunk = delta.content
                content += text_chunk
                buffer += text_chunk
                
                # Tag detector variables to correctly stream blocks
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
                
                clean_chunk = re.sub(r'</?think>|</?tool_call>', '', text_chunk)

                if in_think:
                    if verbose:
                        console.print(clean_chunk, end="", style="dim", highlight=False, markup=False)
                    if on_event and clean_chunk:
                        await on_event({"type": "think", "data": clean_chunk})
                elif in_tool:
                    tool_args_buffer += text_chunk
                    if not tool_name_buffer and ">" in tool_args_buffer:
                        match = re.search(r"<([a-zA-Z0-9_]+)>", tool_args_buffer)
                        if match:
                            tool_name_buffer = match.group(1)
                            if verbose:
                                console.print(f"\n[bold magenta]🛠️ Parsing Tool: {tool_name_buffer}...[/bold magenta]")
                    if on_event:
                        await on_event({"type": "tool", "status": "streaming", "data": text_chunk})
                else:
                    if clean_chunk:
                        if verbose:
                            console.print(clean_chunk, end="", highlight=False, markup=False)
                        if on_event:
                            await on_event({"type": "message", "data": clean_chunk})
                
                # Live stream breaker to save resources against infinitely looping bad models
                if chunk_counter % 40 == 0 and len(content) > 500:
                    if _detect_repetition(content):
                        if verbose:
                            print("\n\n[bold red]⚠️ [Stream Interrupted] Repetition Loop Detected! Connection severed.[/bold red]")
                        finish_reason = "repetition" 
                        break 
                        
            # Handle native explicit JSON tool calls out-of-band
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


# ==========================================
# Core Wrapper Function V2
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
    verbose: bool = False,         
    session_dir: Optional[Path] = None,
    tools: Optional[List[Dict[str, Any]]] = None, 
    tool_strategy: str = "hybrid",   # Choices: ['native_all', 'hybrid', 'text_only']
    allowlist: Optional[List[str]] = None,
    dynamic_tools_registry: Optional[Dict[str, Dict[str, Any]]] = None, # Used strictly during text parsing
    on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None, 
    backend: str = "vllm", # Set explicitly for testing vllm below
    enable_thinking: bool = True,
) -> Tuple[str, List[AgentAction]]:
    """
    Supercharged Wrapper V2 supporting Native JSON Function Calling, Text-based Fallback,
    Concurrency, Streaming, Anti-Looping, and Robust Continuation Sanitization.
    
    Tool Strategies Config:
        - `native_all`: Strict API native tools. Text tags are ignored.
        - `hybrid`: Passes tool schemas to provider, but attempts to parse mutated tags (<write_file>) using the text parser.
        - `text_only`: Ignores provider tool integration entirely. Evaluates everything using text_action_parser.
    """
    full_content = ""
    current_messages = list(messages)
    max_loops = 5
    allowlist = allowlist or []
    final_actions: List[AgentAction] = []
    
    # We strip Active Tools for `text_only` because we don't want the LLM to emit functional JSON, only text format.
    active_tools = tools if tool_strategy in ["native_all", "hybrid"] else None
    
    for i in range(max_loops):
        if i > 0: console.print(f"[dim]Generation loop {i+1}/{max_loops}...[/dim]")
        
        # 1. Token budgeting
        input_text = "\n".join(m.get("content", "") for m in current_messages)
        input_est = estimate_tokens(input_text)
        min_output = 1024
        max_allowed_input = model_max_context - 1000 - min_output
        
        if int(input_est * 1.1) > max_allowed_input > 0:
            console.print(f"[yellow]Compressing messages (est {input_est} > limit).[/yellow]")
            current_messages = compress_messages(current_messages, max_allowed_tokens=int(max_allowed_input / 1.1))
            input_est = estimate_tokens("\n".join(m.get("content", "") for m in current_messages))

        safe_tokens = compute_safe_max_tokens(input_est, model_max_context, max_output_tokens, min_output)

        # 2. Execution Loop
        content, finish_reason, usage_info, native_tcs = "", "stop", {}, []
        start_time = time.time()
        
        for attempt in range(3):
            try:
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
                raise LLMUnavailableError(f"LLM backend unavailable after 3 attempts: {e}") from e

        elapsed = time.time() - start_time
        if verbose: console.print()

        # Emit a usage event so upstream token_stats accumulators are updated.
        if on_event and usage_info:
            await on_event({
                "type": "usage",
                "prompt_tokens": int(usage_info.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage_info.get("completion_tokens", 0) or 0),
                "elapsed_s": elapsed,
            })
        
        # 3. Continuation Sanitizer
        if i > 0: 
            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).lstrip()
            if clean_content.startswith("```python"):
                clean_content = clean_content[9:].lstrip('\n')
            elif clean_content.startswith("```"):
                clean_content = clean_content[3:].lstrip('\n')
            content = _stitch_text(full_content, clean_content)
            
        full_content += content
        
        # Guardrail on catastrophic model degeneracy
        if _detect_repetition(full_content):
            console.print("[bold red]Repetition loop detected! Fusing circuit breaker.[/bold red]")
            safe_length = max(0, len(full_content) - 1000)
            full_content = full_content[:safe_length] + "\n\n[SYSTEM: OUTPUT TRUNCATED DUE TO REPETITION LOOP]"
            break
            
        completion_tokens = int(usage_info.get("completion_tokens", 0) or 0)
        near_budget_cap = completion_tokens >= int(max(128, safe_tokens * 0.9))

        # Detect "think-only" response: model spent its whole budget on <think>...</think>
        # but produced no actual tool call or action text.  This happens when the output
        # budget is so small the model never gets past the reasoning phase.
        _content_without_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        think_only_no_action = bool(
            re.search(r'<think>', content, re.DOTALL)
            and not _content_without_think
        )

        likely_truncated = (
            finish_reason == "length"
            or think_only_no_action
            or (finish_reason in ("stop", None, "") and (_has_unclosed_tool_markup(content) or near_budget_cap))
        )

        # 4. JSON Native Handlers
        # Accept native JSON tool calls for all strategies including hybrid.
        # The hybrid XML text parser still runs afterward as a fallback for non-JSON mutations.
        if native_tcs and not likely_truncated:
            for tc in native_tcs:
                tool_name = tc.get("name", "")
                args_dict = robust_json_loads(tc["arguments"], tool_name=tool_name)
                
                if args_dict is not None:
                    try:
                        action = _parse_native_dict_to_action(tool_name, args_dict, allowlist)
                        final_actions.append(action)
                    except Exception as e:
                        console.print(f"[red]Error mapping JSON to action for {tool_name}: {e}[/red]")
                else:
                    console.print(f"[bold red]Failed to parse native tool JSON completely for {tool_name}.[/bold red]")
                    if tool_strategy != "hybrid":
                        # Only add parse error if NOT hybrid — in hybrid, mutations fall through to XML
                        final_actions.append(ActionToolCall(
                            name="json_parse_error",
                            args={"error": f"The JSON tool call for '{tool_name}' was malformed."}
                        ))
                    else:
                        # In hybrid mode, flag as truncated so the continuation loop retries
                        # with an explicit "malformed JSON" warning injected into the context.
                        likely_truncated = True
                        current_messages.append({"role": "assistant", "content": content})
                        current_messages.append({
                            "role": "user",
                            "content": (
                                f"⚠️ Your `{tool_name}` tool call contained malformed JSON and could not be parsed. "
                                "Please call it again with valid JSON. Make sure all string values use `\\n` "
                                "for newlines and `\\\"` for quotes inside strings."
                            ),
                        })
                        break  # restart the continuation loop with the error injected
            
            if final_actions:
                break

        # 5. Check Cutoff Loop
        if likely_truncated:
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
    # - hybrid: ALWAYS run the XML text parser to pick up mutations,
    #   even if native JSON tools were already parsed for observations.
    # - text_only: Text/XML parser is the sole strategy.
    # - native_all: Text parser only as a fallback if native JSON produced nothing.
    if tool_strategy == "hybrid":
        if verbose:
            console.print(f"[dim]Applying hybrid fallback parser for mutations or textual tool calls...[/dim]")
        parsed_text_actions = parse_text_actions(full_content, allowlist, dynamic_tools_registry)
        # Merge: append any text-parsed actions not already captured by native JSON parse
        for act in parsed_text_actions:
            if act not in final_actions:
                final_actions.append(act)
    elif tool_strategy == "text_only" or (tool_strategy == "native_all" and not final_actions):
        if verbose:
            console.print(f"[dim]Applying {tool_strategy} fallback parser for mutations or textual tool calls...[/dim]")
        parsed_text_actions = parse_text_actions(full_content, allowlist, dynamic_tools_registry)
        for act in parsed_text_actions:
            if act not in final_actions:
                final_actions.append(act)

    return full_content, final_actions

# ==========================================
# Integration Testing for Different Tool Strategies
# ==========================================
if __name__ == "__main__":
    from openai import AsyncOpenAI
    
    # Configure parameters as supplied by the user instructions
    API_BASE = "http://100.110.236.127:8000/v1"
    API_KEY = "EMPTY"
    MODEL_NAME = "qwen3.5-9b"
    
    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "Lists files.",
                "parameters": {
                    "type": "object",
                    "properties": {"dir_path": {"type": "string"}},
                }
            }
        }
    ]

    async def run_integration_tests():
        client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
        console.print(f"\\n[bold green]Testing Connection to vLLM ({API_BASE} using {MODEL_NAME})[/bold green]\\n")
        
        # Testing 1: "text_only" strategy
        console.print("=> [cyan]Testing Strategy: text_only[/cyan]")
        msg = [{"role": "user", "content": "Please output a raw XML tool call to list the current directory '.' wrapped in a <tool_call> block. Do not write text."}]
        _, result_actions = await complete_with_continuation_async(
            client=client, model=MODEL_NAME, messages=msg, tool_strategy="text_only", tools=sample_tools, verbose=True
        )
        console.print(f"Result parsed actions ({len(result_actions)}): {result_actions}")
        
        # Testing 2: "native_all" strategy
        console.print("\\n=> [cyan]Testing Strategy: native_all[/cyan]")
        msg = [{"role": "user", "content": "Please emit purely Native JSON for the list_directory tool with dir_path: './foo'."}]
        _, result_actions = await complete_with_continuation_async(
            client=client, model=MODEL_NAME, messages=msg, tool_strategy="native_all", tools=sample_tools, verbose=True
        )
        console.print(f"Result parsed actions ({len(result_actions)}): {result_actions}")
        
        # Testing 3: "hybrid" strategy
        console.print("\\n=> [cyan]Testing Strategy: hybrid[/cyan]")
        msg = [{"role": "user", "content": "Please output a raw <write_file><path>test.txt</path><content>data123</content></write_file> chunk."}]
        _, result_actions = await complete_with_continuation_async(
            client=client, model=MODEL_NAME, messages=msg, tool_strategy="hybrid", tools=sample_tools, verbose=True, allowlist=["test.txt"]
        )
        console.print(f"Result parsed actions ({len(result_actions)}): {result_actions}")

    try:
        asyncio.run(run_integration_tests())
    except Exception as e:
        console.print(f"[red]Test connection failed. Is the vLLM server running? Error: {e}[/red]")

import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
import sys
import httpx
import argparse
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
import time
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------
# Import our newly refactored robust components
# ---------------------------------------------------------
from BatchAgent.llm_wrapper import complete_with_continuation_async #, get_compiled_tools, BASE_TOOLS
from BatchAgent.tool_handler import UniversalToolHandler
from BatchAgent.mini_batch_agent_libs import (
    now_stamp, ensure_dirs, run_shell, _determine_verify_cmd, build_debug_prompt,
    _handle_missing_modules, top_level_tree, read_file, estimate_tokens, truncate_to_tokens, robust_json_loads, sha1_text
)
from BatchAgent.mini_batch_agent_base import ActionApplyDiff, ActionToolCall, ActionWriteFile, ActionReplaceText
from BatchAgent.tools_registry import get_base_tools, compile_tools_for_provider #get_active_tools
from BatchAgent.prompt_registry import PromptRegistry

from BatchAgent.domain_tools import DOMAIN_REGISTRY

console = Console()

# ==========================================
# 1. Configuration & Data Structures
# ==========================================
@dataclass
class AgentConfig:
    client: Any
    model: str
    session_dir: Path
    workspace_dir: Path
    max_context: int
    max_output: int
    require_approval: bool
    agent_dir: Path
    tool_strategy: str  # [NEW] 'native_all', 'hybrid', 'text_only'
    parallel_thinking: bool = False  # [NEW] switch for enabling the Batch Brainstorming feature
    provider: str = "openai"
    sandbox_container: Optional[str] = None
    verbose: bool = False
    max_retries: int = 4
    serper_api_key: str = ""
    domain: str = "general"
    location: str = "California, United States"
    current_time: str = ""
    temperature: float = 0.1  # [NEW] optimized for strict formatting 
    memory_strategy: str = "sliding_window"  # [NEW] 'sliding_window', 'summarize'
    backend: str = "vllm" # [NEW] 'vllm', 'llama.cpp', 'openai', 'anthropic'
    enable_thinking: bool = True # [NEW] Switch for internal reasoning injection
    enable_turn_limits: bool = False
    max_turns: int = 15  # configurable max ReAct turns
    embedding_base_url: str = "http://100.83.246.7:8003/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model: str = "custom_bge_gpu1"
    # Optional: async token callback for SSE streaming (set by agent_service / API layer)
    stream_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
# ==========================================
# 3. Universal Agent Class (Single-Track ReAct)
# ==========================================
class UniversalAgent:
    """
    Modular, RL-friendly Autonomous Agent with Batched Parallel Thinking.
    """
    def __init__(self, config: Any, system_message: str, tools: List[Dict[str, Any]]):
        self.config = config
        self.system_message = system_message
        self.tools = tools
        self.messages: List[Dict[str, str]] = []
        self.rl_trajectory: List[Dict[str, str]] = []
        
        # [NEW] dynamic tools
        self.dynamic_tools_schema: List[Dict[str, Any]] = []
        self.dynamic_tools_mapping: Dict[str, str] = {} # tool_name -> script_path

    def _log_rl(self, role: str, content: str):
        self.rl_trajectory.append({"role": role, "content": content})

    def _save_trajectory_to_disk(self, task_idx: int, reward: float):
        log_file = self.config.session_dir / "rl_trajectory.jsonl"
        entry = {
            "task_id": f"{self.config.session_dir.name}_task_{task_idx}",
            "reward": reward,
            "messages": self.rl_trajectory
        }
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            console.print(f"[dim]RL Trajectory saved (Reward: {reward})[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to write RL trajectory log: {e}[/red]")

    async def _execute_tools_concurrently(self, tool_actions: List[ActionToolCall], turn_dir: Path, allowlist: List[str]) -> str:
        """Executes information-gathering tools concurrently."""
        if not tool_actions: return ""
        console.print(f"[cyan]🚀 Executing {len(tool_actions)} observation tools concurrently...[/cyan]")
        
        handler = UniversalToolHandler(self.config, turn_dir, allowlist)
        
        async def run_single_tool(action: ActionToolCall) -> str:
            # Run in thread pool to prevent blocking the async loop
            _, result_str = await asyncio.to_thread(handler.execute, [action], "")
            return result_str

        tasks = [run_single_tool(action) for action in tool_actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined_results = []
        for action, res in zip(tool_actions, results):
            if isinstance(res, Exception):
                err = f"### Result for {action.name}\n```text\nException: {str(res)}\n```"
                combined_results.append(err)
                if self.config.verbose: console.print(f"[red]{err}[/red]")
            else:
                combined_results.append(res)
                if self.config.verbose:
                    console.print(Panel(res[:500] + "..." if len(res)>500 else res, title=f"Tool Output: {action.name}", border_style="magenta"))
                
        return "\n\n".join(combined_results)

    
    async def _prune_memory(self):
        """Intelligent context window management avoiding context loss."""
        if len(self.messages) > 12:
            if getattr(self.config, 'memory_strategy', 'sliding_window') == 'summarize':
                # We need to compress messages from index 2 up to -10
                msgs_to_compress = self.messages[2:-10]
                text_to_compress = "\n\n".join([f"Role: {m['role']}\n{m['content']}" for m in msgs_to_compress])
                
                console.print("\n[magenta]🧠 Compress memory: Summarizing previous actions to preserve context...[/magenta]")
                prompt = (
                    "Please concisely summarize the following sequence of actions, tool results, and findings. "
                    "Focus only on WHAT tools were used, WHAT data was discovered, and WHAT files were changed. "
                    "Do NOT include conversational filler.\n\n"
                    f"{text_to_compress}"
                )
                
                from BatchAgent.llm_wrapper import complete_with_async
                try:
                    summary, _ = await complete_with_async(
                        client=self.config.client, model=self.config.model, provider=self.config.provider,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, max_output_tokens=1000,
                        stream=False, backend=self.config.backend, enable_thinking=False
                    )
                    
                    compressed_msg = {
                        "role": "user", 
                        "content": f"[SYSTEM: The following is a summary of earlier actions to preserve context]\n{summary}"
                    }
                    self.messages = [self.messages[0], self.messages[1], compressed_msg] + self.messages[-10:]
                    console.print("[dim]🧹 Memory window gracefully compressed via LLM summarization.[/dim]")
                except Exception as e:
                    console.print(f"[red]Failed to summarize memory ({e}), falling back to sliding window.[/red]")
                    self.messages = [self.messages[0], self.messages[1]] + self.messages[-10:]
            else:
                # Keep: System Prompt (0), User Goal (1)
                # Keep: latest 10 messages
                self.messages = [self.messages[0], self.messages[1]] + self.messages[-10:]
                console.print("[dim]🧹 Memory window gracefully pruned (kept last 10 turns).[/dim]")

    async def _execute_batch_brainstorm(self, action: ActionToolCall, allowlist: List[str]) -> str:
        """
        Batch LLM Calls。
        leverage vLLM or any async-compatible LLM wrapper to run multiple parallel calls with different temperatures or random seeds, to generate a diverse set of strategies for the same problem statement.
        """
        problem = action.args.get('problem_statement', 'Help me solve this task.')
        n_vars = min(4, max(2, int(action.args.get('n_variations', 3))))
        
        console.print(f"\n[bold magenta]🧠 Launching {n_vars} Batch LLM Calls for Parallel Thinking...[/bold magenta]")
        
        # pure prompts for brainstorming, no tool calls allowed to maximize creativity and minimize parsing errors
        bs_messages = self.messages + [{
            "role": "user", 
            "content": f"Please brainstorm a unique strategy to solve: '{problem}'. Do NOT write final code. Just provide the step-by-step logic."
        }]
        
        async def _fetch_idea(idx: int) -> str:
            # adjust temperature for each branch to encourage diversity. The first branch is the most focused (lowest temp), and later branches get more creative freedom.
            div_temp = min(0.9, getattr(self.config, 'temperature', 0.2) + 0.3 + (idx * 0.15))
            
            # import complete_with_continuation_async
            from BatchAgent.llm_wrapper import complete_with_continuation_async
            
            content, _ = await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=bs_messages,
                temperature=div_temp,
                max_output_tokens=1024, # limit the length of each brainstormed idea to save tokens
                model_max_context=self.config.max_context,
                provider=self.config.provider,
                stream=False,   # disable streaming for batch calls to simplify aggregation
                verbose=False,
                tool_strategy="text", # text only for brainstorming to avoid parsing issues and encourage free-form thinking
                allowlist=allowlist,
                backend=self.config.backend,
                enable_thinking=self.config.enable_thinking
            )
            return f"### Approach {idx + 1} (Temp {div_temp:.2f}):\n{content}\n"

        bs_start = time.time()
        # launch parallel vLLM calls and gather results
        ideas = await asyncio.gather(*(_fetch_idea(i) for i in range(n_vars)), return_exceptions=True)
        
        valid_ideas = [idea for idea in ideas if not isinstance(idea, Exception)]
        combined_ideas = "\n\n".join(valid_ideas)
        
        console.print(f"[dim]Batch Brainstorming completed in {time.time() - bs_start:.1f}s[/dim]")
        
        return (
            f"### Parallel Brainstorming Results\n{combined_ideas}\n\n"
            f"Please review these options, select the best approach (or a combination), and proceed with your format (Format A or B)."
        )

    def _verify_mutations(self, content: str, mutation_actions: List[Any], allowlist: List[str], turn_dir: Path) -> Tuple[bool, str]:
        """
        sandbox verification
        return: (is_success, feedback_string)
        """
        # 1. extract modified file paths from mutation actions for verification context
        modified_files = []
        for a in mutation_actions:
            if hasattr(a, 'path'): modified_files.append(a.path)
            elif isinstance(a, ActionApplyDiff):
                paths = re.findall(r'^\+\+\+ b/(.+)$', getattr(a, 'diff_text', ''), re.MULTILINE)
                modified_files.extend(paths)
        modified_files = list(set(modified_files))
        
        # 2. infer the verification command from the agent's response or use a default one based on modified files
        auto_verify_cmd = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
        v_cmd = auto_verify_cmd.group(1).strip() if auto_verify_cmd else None
        verify_cmd = _determine_verify_cmd(allowlist, modified_files, v_cmd, self.config)
        
        if not verify_cmd:
            console.print("[yellow]No verification command found/needed. Assuming Success.[/yellow]")
            # [FIX] give agent clear next steps instead of just returning True, to prevent it from getting stuck in a loop of rewriting files without understanding the verification process
            return (
                "✅ File modifications successfully applied to disk.\n"
                "**System Prompt**: No code execution was required for these files. "
                "If you have completed the user's analytical or writing task, you MUST invoke the `<tool_call><finish_task>...` tool IMMEDIATELY in your next response to conclude the task. "
                "Do NOT rewrite the files unless you found an error."
            )
            
        # 3. check approoval
        if getattr(self.config, 'require_approval', False):
            console.print(f"\n[bold red]⚠️ Agent intends to run a command: [/bold red] `[white]{verify_cmd}[/white]`")
            from rich.prompt import Confirm
            if not Confirm.ask("Do you approve executing this command?"):
                return f"⚠️ User bypassed verification for command: {verify_cmd}"

        # 4. execute the verification command in a sandboxed environment and capture output
        console.print(f"[cyan]-> Running Verification Sandbox...[/cyan]")
        code, out = run_shell(verify_cmd, cap=20000, sandbox_container=getattr(self.config, 'sandbox_container', None))
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        # [FIX] not just return True。The Agent will judge the results！
        if code == 0:
            console.print("[bold green]✅ Verification Script Executed (Exit 0)[/bold green]")
            if getattr(self.config, 'verbose', False): console.print(f"[dim]{out}[/dim]")
            return (
                f"### Verification Execution Result (Exit Code: 0)\n"
                f"```text\n{out}\n```\n"
                f"**System Prompt**: The script executed without crashing. Please carefully review the output above against the ORIGINAL USER GOAL.\n"
                f"- Did the output explicitly fulfill ALL constraints requested by the user? (e.g., quantity of items, specific formats?)\n"
                f"- If the task is only partially complete, write more code or take further actions.\n"
                f"- ONLY if every single requirement is met, use the `finish_task` tool to conclude."
            )
        else:
            console.print(f"[bold red]❌ Verification Failed (exit={code})[/bold red]")
            smart_error = build_debug_prompt(out, root_dir=str(self.config.workspace_dir))
            
            return (
                f"⚠️ [Verification Failed] Exit Code: {code}\n"
                f"Command: {verify_cmd}\n"
                f"{smart_error}\n"
                f"Please analyze the error. You may use observation tools to investigate, then provide a fix."
            )

    # =========================================================
    # THE MAIN LOOP (Now incredibly clean and readable)
    # =========================================================
    async def execute_task(
        self,
        task_goal: str,
        task_idx: int,
        allowlist: List[str],
        prompt_md: str,
        resume_messages: Optional[List[Dict[str, str]]] = None,
        start_turn_index: int = 0,
        resume_rl_trajectory: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[bool, str]:
        
        console.print(f"\n[bold green]=== Starting Agent Task {task_idx+1} ===[/bold green]")
        if resume_messages:
            restored = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in resume_messages
                if isinstance(m, dict) and m.get("role") and m.get("content") is not None
            ]
            if restored and restored[0].get("role") == "system":
                restored[0]["content"] = self.system_message
            else:
                restored.insert(0, {"role": "system", "content": self.system_message})
            self.messages = restored
            if resume_rl_trajectory:
                self.rl_trajectory = [
                    {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                    for m in resume_rl_trajectory
                    if isinstance(m, dict) and m.get("role") and m.get("content") is not None
                ]
            else:
                self.rl_trajectory = list(self.messages)
        else:
            self.messages = [{"role": "system", "content": self.system_message}]
            self._log_rl("system", self.system_message)

        MAX_REACT_TURNS = getattr(self.config, 'max_turns', 15)
        from BatchAgent.llm_wrapper import complete_with_continuation_async # Local import

        final_result_text = "Task aborted or failed."
        # Loop-detection: track (tool_name, arg_hash) fingerprints from last 3 turns
        _recent_fingerprints: list = []
        _recent_mutation_fingerprints: list = []
        consecutive_json_parse_turns = 0
        last_write_only_signature = None
        write_only_repeat_count = 0

        def _tool_fingerprint(action: ActionToolCall):
            name = str(getattr(action, "name", "") or "").strip()
            args = getattr(action, "args", {}) or {}
            if name == "read_file_chunk":
                raw_path = str(args.get("filepath", "")).strip().replace("\\", "/")
                normalized_path = re.sub(r"^\./+", "", raw_path).lower()
                try:
                    start_line = int(args.get("start_line", 1))
                except Exception:
                    start_line = 1
                try:
                    end_line = int(args.get("end_line", start_line + 200))
                except Exception:
                    end_line = start_line + 200
                span = 25
                start_bucket = (max(1, start_line) // span) * span
                end_bucket = (max(start_line, end_line) // span) * span
                return (name, normalized_path, start_bucket, end_bucket)
            try:
                normalized_args = json.dumps(args, sort_keys=True)
            except Exception:
                normalized_args = str(args)
            return (name, normalized_args)

        for current_turn in range(MAX_REACT_TURNS):
            await self._prune_memory()
            
            absolute_turn = start_turn_index + current_turn
            display_turn = absolute_turn + 1
            turn_dir = self.config.session_dir / f"{task_idx * 100 + absolute_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)

            # ===============================================================
            # [NEW] 1. complete Prompt for debug
            # ===============================================================
            # (turn_dir / "full_prompt_context.json").write_text(
            #     json.dumps(self.messages, indent=2, ensure_ascii=False), 
            #     encoding="utf-8"
            # )
            # # Human readable version of the prompt for easier debugging
            # (turn_dir / "latest_instruction.md").write_text(self.messages[-1]["content"], encoding="utf-8")
            
            console.print(f"\n[bold yellow]>>> ReAct Turn {display_turn} / {MAX_REACT_TURNS}[/bold yellow]")

            # Notify streaming clients of a new ReAct turn so the UI can create a new message bubble
            if self.config.stream_callback:
                await self.config.stream_callback({
                    "type": "turn_start",
                    "turn": display_turn,
                    "max_turns": MAX_REACT_TURNS,
                })

            # [NEW] Inject temporal awareness if enabled
            current_user_prompt = prompt_md if current_turn == 0 else "" # Initial prompt is only for the first turn
            if self.config.enable_turn_limits:
                current_user_prompt = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{current_user_prompt}"
            
            if current_turn == 0:
                self._log_rl("user", prompt_md)
                self.messages.append({"role": "user", "content": current_user_prompt})
                (turn_dir / "user_prompt.md").write_text(prompt_md, encoding="utf-8")
            
            # --- 1. LLM Generation ---
            content, actions = await complete_with_continuation_async(
                client=self.config.client, model=self.config.model, messages=self.messages,
                temperature=getattr(self.config, 'temperature', 0.1), max_output_tokens=self.config.max_output,
                model_max_context=self.config.max_context, provider=self.config.provider,
                stream=True, verbose=self.config.verbose, session_dir=self.config.session_dir,
                tools=self.tools, tool_strategy=self.config.tool_strategy, allowlist=allowlist,
                on_event=getattr(self.config, 'stream_callback', None),  # SSE forwarding hook
                backend=getattr(self.config, 'backend', 'vllm'),
                enable_thinking=getattr(self.config, 'enable_thinking', True)
            )
            
            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)
            final_result_text = content # keep track of last message

            # ===============================================================
            # [NEW] 2. extract the actions for this turn and log them in a structured way for better interpretability and debugging. This also allows us to analyze the agent's decision-making process and tool usage patterns over time.
            # ===============================================================
            action_logs = []
            for a in actions:
                if hasattr(a, 'name') and hasattr(a, 'args'): # ActionToolCall
                    action_logs.append({"type": "Native/XML Tool", "name": a.name, "args": a.args})
                else: # File Mutations
                    action_logs.append({"type": a.__class__.__name__, "path": getattr(a, 'path', 'N/A')})
            
            (turn_dir / "parsed_actions.json").write_text(
                json.dumps(action_logs, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
            
            # --- 2. Check for Conclusion ---
            finish_task_action = next((a for a in actions if getattr(a, 'name', '') == 'finish_task'), None)
            if finish_task_action:
                console.print("[bold green]🏁 Agent explicitly finished the task.[/bold green]")
                self._save_trajectory_to_disk(task_idx, reward=1.0)
                summary = finish_task_action.args.get('summary', final_result_text)
                return True, summary

            json_parse_errors = [
                a for a in actions
                if isinstance(a, ActionToolCall)
                and getattr(a, "name", "") == "json_parse_error"
                and "write_file" in str(getattr(a, "args", {}).get("error", "")).lower()
            ]
            if json_parse_errors:
                consecutive_json_parse_turns += 1
                if self.config.tool_strategy == "native_all" and consecutive_json_parse_turns >= 2:
                    self.config.tool_strategy = "hybrid"
                    switched_base_tools = get_base_tools(
                        strategy=self.config.tool_strategy,
                        enable_parallel=getattr(self.config, "parallel_thinking", False),
                        domain=getattr(self.config, "domain", "general"),
                    )
                    self.tools = compile_tools_for_provider(
                        switched_base_tools,
                        self.config.provider,
                        self.config.tool_strategy,
                    )
                    self.system_message = PromptRegistry.get_system_prompt(
                        strategy=self.config.tool_strategy,
                        base_tools_list=switched_base_tools,
                        domain=getattr(self.config, "domain", "general"),
                        location=getattr(self.config, "location", "California, United States"),
                        current_time=getattr(self.config, "current_time", ""),
                    )
                    self.messages[0]["content"] = self.system_message
                    parse_feedback = (
                        "System switched tool strategy from native_all to hybrid after repeated write_file JSON parsing failures. "
                        "Continue with native tools for observation and XML write_file/search_and_replace for file mutations."
                    )
                    self.messages.append({"role": "assistant", "content": content})
                    feedback_prompt = parse_feedback
                    if self.config.enable_turn_limits:
                        feedback_prompt = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"
                    self._log_rl("user", parse_feedback)
                    self.messages.append({"role": "user", "content": feedback_prompt})
                    (turn_dir / "strategy_switch_feedback.md").write_text(feedback_prompt, encoding="utf-8")
                    continue
            else:
                consecutive_json_parse_turns = 0
                
            if not actions:
                console.print("[yellow]Agent outputted nothing useful. Forcing format switch.[/yellow]")
                
                # [FIX 2: dynamic error feedback based on tool strategy] Instead of a generic message, provide specific guidance based on the current tool strategy to help the agent course-correct more effectively.
                if self.config.tool_strategy == "native_all":
                    feedback = (
                        "⚠️ System Warning: No valid tool calls detected.\n"
                        "You MUST use your native JSON function calling capabilities to execute tools (e.g. `web_search`, `write_file`). "
                        "Do NOT output JSON in plain text code blocks."
                    )
                elif self.config.tool_strategy == "text_only":
                    feedback = (
                        "⚠️ System Warning: No valid XML tool calls detected.\n"
                        "Please use the exact XML tags (e.g., `<tool_call><web_search>...</web_search></tool_call>`) to use tools, "
                        "or use Format B to write files."
                    )
                else: # hybrid
                    feedback = (
                        "⚠️ System Warning: No valid tool calls or file modifications were detected.\n"
                        "Use native JSON tools for searching/reading, and **Format B (WRITE_FILE)** directly in the Markdown text for writing code.\n"
                        "If you are finished, invoke the `finish_task` tool."
                    )

                self.messages.append({"role": "assistant", "content": content})
                
                feedback_prompt = feedback
                if self.config.enable_turn_limits:
                    feedback_prompt = f"[Current Turn: {current_turn + 1} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"

                self._log_rl("user", feedback)
                self.messages.append({"role": "user", "content": feedback_prompt})
                (turn_dir / "user_feedback.md").write_text(feedback_prompt, encoding="utf-8")
                continue
                
            # --- 3. Action Segregation ---
            info_actions = [a for a in actions if isinstance(a, ActionToolCall)]
            mutation_actions = [a for a in actions if isinstance(a, (ActionWriteFile, ActionApplyDiff, ActionReplaceText))]
            feedback_blocks = []
            if json_parse_errors:
                feedback_blocks.append(
                    "Tool JSON parsing failed for at least one call. If writing files, switch to XML write format in hybrid mode."
                )

            # =====================================================================
            # [NEW]load domain-specific Hot-Swapping tools
            # =====================================================================
            load_domain_action = next((a for a in info_actions if getattr(a, 'name', '') == 'load_domain_tools'), None)
            
            if load_domain_action:
                new_domain = load_domain_action.args.get('domain', '')
                console.print(f"\n[bold green]🔌 Agent is hot-swapping domain plugin: [white]{new_domain}[/white][/bold green]")
                
                if new_domain in DOMAIN_REGISTRY:
                    # 1. update Config
                    self.config.domain = new_domain
                    
                    # 2. get the new domain's tools and recompile them for the current provider
                    new_base_tools = get_base_tools(
                        strategy=self.config.tool_strategy, 
                        enable_parallel=getattr(self.config, 'parallel_thinking', False), 
                        domain=new_domain
                    )
                    self.tools = compile_tools_for_provider(new_base_tools, self.config.provider, self.config.tool_strategy)
                    
                    # 3. re-generate System Prompt to include the new tools' usage instructions
                    self.system_message = PromptRegistry.get_system_prompt(
                        self.config.tool_strategy,
                        new_base_tools,
                        new_domain,
                        self.config.location,
                        getattr(self.config, "current_time", ""),
                    )
                    
                    # 4. change the first message in the conversation to the new system prompt, so that the agent can immediately pick up on the new tools and instructions in the next turn without needing a full reset or restart
                    self.messages[0]["content"] = self.system_message
                    
                    tool_names = [t['name'] for t in DOMAIN_REGISTRY[new_domain]]
                    load_feedback = (
                        f"✅ Successfully loaded the '{new_domain}' plugin!\n"
                        f"**New tools available**: {', '.join(tool_names)}.\n"
                        f"Please proceed to use these new tools to complete your task."
                    )
                else:
                    load_feedback = f"❌ Failed to load domain. '{new_domain}' is not a valid domain plugin."
                
                # record this hot-swapping event in the trajectory with the new system prompt and feedback, to help the agent understand the change in its capabilities and instructions
                self.messages.append({"role": "assistant", "content": content})
                
                feedback_prompt = load_feedback
                if self.config.enable_turn_limits:
                    feedback_prompt = f"[Current Turn: {current_turn + 1} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"

                self._log_rl("user", load_feedback)
                self.messages.append({"role": "user", "content": feedback_prompt})
                (turn_dir / "load_feedback.md").write_text(feedback_prompt, encoding="utf-8")
                continue
            # =====================================================================

            # --- 4. Handle Parallel Brainstorming (Optional / Configurable) ---
            # only config enabled enable_parallel_thinking, add this tool
            brainstorm_action = next((a for a in info_actions if getattr(a, 'name', '') == 'brainstorm_solutions'), None)
            if brainstorm_action and getattr(self.config, 'enable_parallel_thinking', False):
                bs_feedback = await self._execute_batch_brainstorm(brainstorm_action, allowlist)
                self.messages.append({"role": "assistant", "content": content})
                
                feedback_prompt = bs_feedback
                if self.config.enable_turn_limits:
                    feedback_prompt = f"[Current Turn: {current_turn + 1} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"

                self._log_rl("user", bs_feedback)
                self.messages.append({"role": "user", "content": feedback_prompt})
                (turn_dir / "bs_feedback.md").write_text(feedback_prompt, encoding="utf-8")
                continue # get all the ideas, then let the agent decide how to proceed in the next turn, instead of rushing into tool execution or file mutations

            # --- 5. Execute Standard Tools (Concurrently) ---
            standard_info_actions = [a for a in info_actions if a.name not in ['brainstorm_solutions', 'finish_task', 'load_domain_tools', 'register_custom_tool', 'json_parse_error']]

            # Loop detection: skip actions whose (name, args) fingerprint appeared in the last 2 turns
            if standard_info_actions:
                deduped, skipped = [], []
                for a in standard_info_actions:
                    fp = _tool_fingerprint(a)
                    if _recent_fingerprints.count(fp) >= 1:
                        skipped.append(a.name)
                    else:
                        deduped.append(a)
                if skipped:
                    console.print(f"[yellow]⚠️ Loop detected — skipping repeated calls: {skipped}[/yellow]")
                    feedback_blocks.append(
                        f"⚠️ **Loop detected**: You already called {skipped} with identical arguments in a previous turn. "
                        "The results were already provided. Use the information you have to proceed — "
                        "either synthesize what you know into a `write_file` call or call `finish_task`."
                    )
                standard_info_actions = deduped

            if standard_info_actions:
                # Record fingerprints for this turn
                for a in standard_info_actions:
                    fp = _tool_fingerprint(a)
                    _recent_fingerprints.append(fp)
                if len(_recent_fingerprints) > 12:
                    _recent_fingerprints[:] = _recent_fingerprints[-12:]

                concurrent_results = await self._execute_tools_concurrently(standard_info_actions, turn_dir, allowlist)
                feedback_blocks.append(f"### Tool Execution Results\n{concurrent_results}")
            
            # =====================================================================
            # [NEW] Create and register Custom Tools on the Fly
            # =====================================================================
            register_action = next((a for a in info_actions if getattr(a, 'name', '') == 'register_custom_tool'), None)
            
            if register_action:
                args = register_action.args
                t_name = args.get("tool_name", "custom_tool")
                
                console.print(f"\n[bold green]🛠️ Agent created a new tool: [white]{t_name}[/white][/bold green]")
                
                # Create the tool schema based on the agent's input. We expect the agent to provide a clear schema with name, description, properties, and required args. This allows the agent to effectively teach itself new capabilities during the task.
                new_schema = {
                    "name": t_name,
                    "description": args.get("description", ""),
                    "properties": args.get("schema_properties", {}),
                    "required": args.get("required_args", [])
                }
                
                # save to dynamic tools registry (in-memory for now, can be extended to disk if needed)
                self.dynamic_tools_schema.append(new_schema)
                self.dynamic_tools_mapping[t_name] = args.get("script_path", "")
                
                current_base_tools = get_base_tools(self.config.tool_strategy, getattr(self.config, 'parallel_thinking', False), getattr(self.config, 'domain', 'auto'))
                current_base_tools.extend(self.dynamic_tools_schema)
                
                # update the useful tools that will be passed to the LLM in the next turn, so that the agent can start using the new tool it just created right away in the next turn's response
                self.tools = compile_tools_for_provider(current_base_tools, self.config.provider, self.config.tool_strategy)
                
                # update system prompt to reflect the new tool in the instructions, so that the agent can understand how to use it properly in the next turn. This is crucial for enabling the agent to effectively leverage its newly created capabilities without needing explicit human intervention or retraining.
                self.system_message = PromptRegistry.get_system_prompt(
                    self.config.tool_strategy,
                    current_base_tools,
                    getattr(self.config, 'domain', 'auto'),
                    self.config.location,
                    getattr(self.config, "current_time", ""),
                )
                self.messages[0]["content"] = self.system_message
                
                feedback = (
                    f"✅ Successfully compiled and registered the new tool `{t_name}`!\n"
                    f"The tool is now loaded into your context. You can call it immediately just like any other native tool."
                )
                
                self.messages.append({"role": "assistant", "content": content})
                
                feedback_prompt = feedback
                if self.config.enable_turn_limits:
                    feedback_prompt = f"[Current Turn: {current_turn + 1} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"

                self._log_rl("user", feedback)
                self.messages.append({"role": "user", "content": feedback_prompt})
                (turn_dir / "custom_tool_feedback.md").write_text(feedback_prompt, encoding="utf-8")
                continue
            
            # --- 6. Execute Mutations & Verify ---
            if mutation_actions:
                deduped_mutations = []
                skipped_mutations = []
                current_mutation_fingerprints = []

                for a in mutation_actions:
                    if isinstance(a, ActionWriteFile):
                        fp = ("write_file", str(getattr(a, "path", "")).strip().lower(), sha1_text(getattr(a, "content", "")))
                    elif isinstance(a, ActionReplaceText):
                        fp = (
                            "search_and_replace",
                            str(getattr(a, "path", "")).strip().lower(),
                            sha1_text(f"{getattr(a, 'old_text', '')}::{getattr(a, 'new_text', '')}")
                        )
                    elif isinstance(a, ActionApplyDiff):
                        fp = ("apply_diff", sha1_text(getattr(a, "diff_text", "")))
                    else:
                        fp = ("mutation", sha1_text(str(a)))

                    if _recent_mutation_fingerprints.count(fp) >= 1:
                        skipped_mutations.append(fp[0])
                        continue

                    deduped_mutations.append(a)
                    current_mutation_fingerprints.append(fp)

                if skipped_mutations:
                    feedback_blocks.append(
                        "⚠️ Repeated identical file mutation detected and skipped. "
                        "The same write/edit was already applied in a previous turn. "
                        "Use the existing file result and call finish_task if the goal is completed."
                    )

                mutation_actions = deduped_mutations

            if mutation_actions:
                handler = UniversalToolHandler(self.config, turn_dir, allowlist, dynamic_tools_mapping=self.dynamic_tools_mapping)
                has_mutation, mutation_res = handler.execute(mutation_actions, content)
                feedback_blocks.append(mutation_res)

                if self.config.stream_callback:
                    file_records = list(getattr(handler, "written_file_records", []) or [])
                    if not file_records and handler.written_files:
                        from BatchAgent.minio_uploader import upload_file
                        for fpath in handler.written_files:
                            fname = Path(fpath).name
                            session_name = getattr(self.config, "session_dir", Path(".")).name
                            object_name = f"agent/{session_name}/{fname}"
                            url = upload_file(fpath, object_name=object_name)
                            inline_content = None
                            try:
                                ext = Path(fpath).suffix.lower()
                                if ext in {".md", ".markdown", ".txt"} or os.path.getsize(fpath) < 200 * 1024:
                                    with open(fpath, "r", encoding="utf-8", errors="replace") as _f:
                                        inline_content = _f.read()
                            except Exception:
                                pass
                            file_records.append({
                                "name": fname,
                                "url": url or "",
                                "local_path": fpath,
                                "content": inline_content,
                                "size": os.path.getsize(fpath) if os.path.exists(fpath) else 0,
                            })
                    for rec in file_records:
                        await self.config.stream_callback({
                            "type": "file_written",
                            "name": rec.get("name") or "",
                            "url": rec.get("url") or "",
                            "local_path": rec.get("local_path") or "",
                            "content": rec.get("content"),
                            "size": int(rec.get("size") or 0),
                        })

                if has_mutation:
                    if current_mutation_fingerprints:
                        _recent_mutation_fingerprints.extend(current_mutation_fingerprints)
                        if len(_recent_mutation_fingerprints) > 8:
                            _recent_mutation_fingerprints[:] = _recent_mutation_fingerprints[-8:]
                    # [FIX] get the output from sandbox verification and provide it as feedback to the agent, instead of just returning True/False. This allows the agent to understand the results of its actions and learn from any mistakes, rather than just blindly trying again without guidance.
                    v_feedback = self._verify_mutations(content, mutation_actions, allowlist, turn_dir)
                    if v_feedback:
                        feedback_blocks.append(v_feedback)
                    write_only_actions = [a for a in mutation_actions if isinstance(a, ActionWriteFile)]
                    is_write_only_turn = bool(write_only_actions) and len(write_only_actions) == len(mutation_actions) and not standard_info_actions
                    if is_write_only_turn:
                        signature = tuple(sorted(str(a.path).strip().lower() for a in write_only_actions))
                        if len(set(signature)) == 1:
                            if signature == last_write_only_signature:
                                write_only_repeat_count += 1
                            else:
                                last_write_only_signature = signature
                                write_only_repeat_count = 1
                            if write_only_repeat_count >= 2:
                                target_file = signature[0]
                                summary = f"Completed task by writing output file: {target_file}"
                                self._save_trajectory_to_disk(task_idx, reward=1.0)
                                return True, summary
                        else:
                            last_write_only_signature = None
                            write_only_repeat_count = 0
                    else:
                        last_write_only_signature = None
                        write_only_repeat_count = 0
                    # v_success, v_feedback = self._verify_mutations(content, mutation_actions, allowlist, turn_dir)
                    # if v_success:
                    #     self._save_trajectory_to_disk(task_idx, reward=1.0)
                    #     return True
                    # else:
                    #     feedback_blocks.append(v_feedback)

            # --- 7. Loop Feedback ---
            combined_feedback = "\n\n".join(feedback_blocks)
            if not mutation_actions and standard_info_actions:
                combined_feedback += "\nAnalyze the results and continue."
            if not combined_feedback.strip():
                combined_feedback = (
                    "No new actions were executed in this turn. "
                    "If the requested output file is already correct, call finish_task now."
                )
            
            # ===============================================================
            # [NEW] 3. tool execution results (Feedback)
            # ===============================================================
            if combined_feedback:
                (turn_dir / "tool_execution_results.md").write_text(combined_feedback, encoding="utf-8")

            self.messages.append({"role": "assistant", "content": content})
            
            feedback_prompt = combined_feedback
            if self.config.enable_turn_limits:
                feedback_prompt = f"[Current Turn: {display_turn} / Max Turns: {MAX_REACT_TURNS}]\n{feedback_prompt}"

            self._log_rl("user", combined_feedback)
            self.messages.append({"role": "user", "content": feedback_prompt})
            (turn_dir / "user_feedback.md").write_text(feedback_prompt, encoding="utf-8")

        console.print("[bold red]Max ReAct turns exceeded. Task aborted.[/bold red]")
        self._save_trajectory_to_disk(task_idx, reward=-1.0)
        return False, final_result_text

    
# ==========================================
# 4. Helper: API and Context Checker
# ==========================================
async def check_api_and_context(client: Any, provider: str, model: str, default_ctx: int) -> int:
    """
    Checks if the API is reachable and attempts to auto-detect the maximum context length.
    """
    try:
        if provider == "anthropic":
            # Anthropic Claude 3 models generally support 200k
            return 200000
        
        # For OpenAI / vLLM
        if hasattr(client, 'models'):
            models = await client.models.list()
            for m in models.data:
                if m.id == model:
                    # vLLM sometimes exposes max_model_len
                    if hasattr(m, 'max_model_len') and m.max_model_len:
                        return int(m.max_model_len)
        
        # Fallback dummy check to ensure API is alive
        await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return default_ctx
        
    except Exception as e:
        console.print(f"\n[bold red]CRITICAL ERROR: Unable to connect to LLM API ([/bold red]{e}[bold red])[/bold red]")
        console.print("[yellow]Please ensure your vLLM server is running or your API Key is correct.[/yellow]")
        sys.exit(1)


# ==========================================
# 5. Main Entry Point
# ==========================================
async def main_async():
    parser = argparse.ArgumentParser(description="Universal Autonomous ReAct Agent")
    parser.add_argument("--goal", help="Task goal/description")
    parser.add_argument("--allowlist", help="Comma-separated list of files to allow editing", default="")
    parser.add_argument("--context", help="Comma-separated list of read-only context files", default="")
    parser.add_argument("--notes", help="Extra notes/constraints", default="")
    
    # Provider & Env
    parser.add_argument("--tool-strategy", 
                        choices=["native_all", "hybrid", "text_only"], 
                        default="hybrid",
                        help="Choose how the LLM interacts with tools.")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="LLM Provider")
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1")) #100.110.236.127 http://127.0.0.1:8000/v1
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-base-url", default=os.environ.get("EMBEDDING_BASE_URL", "http://100.83.246.7:8003/v1"))
    parser.add_argument("--embedding-api-key", default=os.environ.get("EMBEDDING_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model", default=os.environ.get("EMBEDDING_MODEL", "custom_bge_gpu1"))
    parser.add_argument("--serper-key", default=os.environ.get("SERPER_API_KEY", ""), help="API key for Web Search")
    
    # System Features (New Additions)
    parser.add_argument("--parallel-thinking", action="store_true", help="Enable Batched LLM calls for branching thoughts (Free for local vLLM, expensive for APIs).")
    parser.add_argument("--output-dir", default="./agent_workspace", help="Default directory for agent to write files")
    parser.add_argument("--sandbox", default=None, help="Docker container name for sandbox execution (e.g. 'my_container')")
    parser.add_argument("--require-approval", action="store_true", help="Require user confirmation before writing files or executing commands")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of LLM responses and tool outputs")
    parser.add_argument("--max-context", type=int, default=16384, help="Fallback context length if auto-detect fails")
    
    parser.add_argument("--domain", default="auto", 
                        choices=["general", "medical", "academic", "news", 
                                 "software_eng", "math", "science", "language", 
                                 "business", "assistant", "sales_support"], 
                        help="Inject domain-specific tools.")
    parser.add_argument("--location", default="California, United States", help="Location for time-sensitive tasks")
    parser.add_argument("--enable-turn-limits", action="store_true", help="Inject turn limits into prompts to remind the agent of remaining turns.")
    
    args = parser.parse_args()

    # --- Directory Setup ---
    workspace_dir = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = (workspace_dir / ".agent").resolve()
    ensure_dirs(agent_dir)
    session_dir = agent_dir / "sessions" / now_stamp()
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL: Change working directory so all relative paths generated by LLM land here
    os.chdir(workspace_dir)
    os.environ["EMBEDDING_BASE_URL"] = args.embedding_base_url
    os.environ["EMBEDDING_API_KEY"] = args.embedding_api_key
    os.environ["EMBEDDING_MODEL"] = args.embedding_model

    # --- Initialize API Client ---
    if args.provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=args.api_key)
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=args.base_url, 
            api_key=args.api_key, 
            http_client=httpx.AsyncClient(timeout=1200.0)
        )

    # --- Pre-flight Checks & UI ---
    console.print(Text(
        " ╔══════════════════════════════════════════════════════════════╗\n"
        " ║             UNIVERSAL AUTONOMOUS AGENT ENGINE              ║\n"
        " ╚══════════════════════════════════════════════════════════════╝",
        style="bold green"
    ))
    
    with Status("[bold cyan]Connecting to LLM API and scanning context limits...[/bold cyan]", spinner="bouncingBar"):
        detected_ctx = await check_api_and_context(client, args.provider, args.model, args.max_context)
    
    console.print(f"[green]✔ API Connection Established[/green]")
    console.print(f"[green]✔ Effective Context Limit: {detected_ctx} tokens[/green]")
    console.print(f"[green]✔ Workspace Directory: {workspace_dir}[/green]")
    console.print(f"[green]✔ Embedding Endpoint: {args.embedding_base_url}[/green]")
    console.print(f"[green]✔ Embedding Model: {args.embedding_model}[/green]")
    if args.sandbox: console.print(f"[yellow]⚠ Sandbox Execution Enabled: Container '{args.sandbox}'[/yellow]")
    if args.require_approval: console.print(f"[yellow]⚠ Human-in-the-loop Active: Requires approval for mutations[/yellow]")

    # --- Configuration ---
    config = AgentConfig(
        client=client,
        model=args.model,
        session_dir=session_dir,
        workspace_dir=workspace_dir,
        max_context=detected_ctx,
        max_output=4096,
        require_approval=args.require_approval,
        agent_dir=agent_dir,
        tool_strategy=args.tool_strategy,
        parallel_thinking=args.parallel_thinking,
        provider=args.provider,
        sandbox_container=args.sandbox,
        verbose=args.verbose,
        serper_api_key=args.serper_key,
        domain=args.domain,
        location=args.location,
        current_time=os.environ.get("BATCHAGENT_CURRENT_TIME", ""),
        enable_turn_limits=args.enable_turn_limits,
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=args.embedding_api_key,
        embedding_model=args.embedding_model,
    )

    # --- Goal Collection ---
    goal = args.goal
    if not goal:
        print()
        goal = Prompt.ask("[bold magenta]🎯 Enter your Goal / Task[/bold magenta]").strip()

    allowlist = [x.strip() for x in args.allowlist.split(",") if x.strip()]
    context_files = [x.strip() for x in args.context.split(",") if x.strip()]

    # --- Agent Initialization & Execution ---
    # =======================================================
    # Based on the selected strategy, setup System Prompt and Tools
    # =======================================================
    # 1. 获取基础工具 Schema (同时支持根据参数开启并行能力)
    base_tools = get_base_tools(
        strategy=config.tool_strategy, 
        enable_parallel=getattr(config, 'parallel_thinking', False)
    )
    
    # 2. 编译为 API 所需格式 (如果是 text_only，这里会自动变成 [])
    compiled_tools = compile_tools_for_provider(
        base_tools=base_tools, 
        provider=config.provider, 
        strategy=config.tool_strategy
    )
    
    # 3. 生成系统提示词 (传入 base_tools 供它动态生成 XML 说明)
    system_prompt = PromptRegistry.get_system_prompt(
        strategy=config.tool_strategy, 
        base_tools_list=base_tools,
        domain=config.domain,
        location=config.location,
        current_time=getattr(config, "current_time", ""),
    )
    # compiled_tools = get_active_tools(config.tool_strategy, config.provider)
    
    #compiled_tools = get_compiled_tools(config.provider)
    #system_prompt = PromptRegistry.get_system_prompt(BASE_TOOLS)
    #task_prompt = PromptRegistry.format_task(goal, allowlist, context_files, args.notes, workspace_dir.name)
    def content_injector(files: list) -> str:
        files_md = ""
        for f in list(dict.fromkeys(files)):
            content = read_file(str(f))
            if content and not content.startswith("[MISSING FILE]"):
                if estimate_tokens(content) > 8000:
                    content = truncate_to_tokens(content, 8000)
                files_md += f"### File: {f}\n```text\n{content}\n```\n"
        return files_md

    task_prompt = PromptRegistry.format_task(goal, [], [], workspace_dir.name, content_injector)
    
    
    agent = UniversalAgent(config=config, system_message=system_prompt, tools=compiled_tools)
    
    success = await agent.execute_task(
        task_goal=goal,
        task_idx=0,
        allowlist=allowlist,
        prompt_md=task_prompt
    )

    if success:
        console.print("\n[bold green]🏁 Agent Operation Completed Successfully![/bold green]")
    else:
        console.print("\n[bold red]💀 Agent Operation Terminated / Failed.[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation manually interrupted by user. Shutting down...[/bold red]")

"""
python agent_main.py --provider anthropic --model claude-3-5-sonnet-20241022 --tool-strategy native_all

python agent_main.py --model qwen2.5-coder-7b --tool-strategy native_all

python agent_main.py --model qwen2.5-coder-7b --tool-strategy hybrid


VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --trust-remote-code

export SERPER_API_KEY="your_serper_api_key_here"   
python BatchAgent/agent_main.py --verbose

Search the web for the latest Python 3.13 features, and then write a python script named test_313.py demonstrating the new features.

Create a tool that fetches the current price of a cryptocurrency from online API, register the tool, and then use it to tell me the price of BTC and ETH.

Create a tool that fetches the English word definition online like a dictionary, test the code, and then use this tool to find some definitions of popular GRE words (<10).

Search the web for the recent crude oil price and its trend, then write a brief analysis report in markdown format.

create 10 questions based on the book of "art of problem solving - prealgebra chapter 1" into one document


python BatchAgent/agent_main.py \
  --base-url "http://100.110.236.127:8000/v1" \
  --api-key "EMPTY" \
  --model "qwen3.5-9b" \
  --embedding-base-url "http://100.83.246.7:8003/v1" \
  --embedding-api-key "EMPTY" \
  --embedding-model "custom_bge_gpu1"
"""

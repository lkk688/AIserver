import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import httpx
import argparse
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------
# Import our newly refactored robust components
# ---------------------------------------------------------
from BatchAgent.llm_wrapper import complete_with_continuation_async #, get_compiled_tools, BASE_TOOLS
from BatchAgent.tool_handler import UniversalToolHandler
from BatchAgent.mini_batch_agent_libs import (
    now_stamp, ensure_dirs, run_shell, _determine_verify_cmd, build_debug_prompt,
    _handle_missing_modules, top_level_tree, read_file, estimate_tokens, truncate_to_tokens, robust_json_loads
)
from BatchAgent.mini_batch_agent import ActionApplyDiff, ActionToolCall, ActionWriteFile, ActionReplaceText
from BatchAgent.tools_registry import get_active_tools
from BatchAgent.prompt_registry import PromptRegistry

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
    provider: str = "openai"
    sandbox_container: Optional[str] = None
    verbose: bool = False
    max_retries: int = 4
    serper_api_key: str = ""




# ==========================================
# 3. Universal Agent Class (Single-Track ReAct)
# ==========================================
class UniversalAgent:
    """
    An Object-Oriented Autonomous Agent implementing a Single-Track Parallel ReAct Loop.
    """
    def __init__(self, config: AgentConfig, system_message: str, tools: List[Dict[str, Any]]):
        self.config = config
        self.system_message = system_message
        self.tools = tools
        self.messages: List[Dict[str, str]] = []
        self.rl_trajectory: List[Dict[str, str]] = []

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

    async def execute_task(self, task_goal: str, task_idx: int, allowlist: List[str], prompt_md: str) -> bool:
        """The core ReAct State Machine."""
        console.print(f"\n[bold green]=== Starting Agent Task {task_idx+1} ===[/bold green]")
        
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt_md}
        ]
        self._log_rl("system", self.system_message)
        self._log_rl("user", prompt_md)

        MAX_REACT_TURNS = 15
        current_turn = 0

        while current_turn < MAX_REACT_TURNS:
            # --- [FIX: Sliding Window Memory] ---
            # If multiple round, cut the memory to avoid the (Death Spiral)
            # Keep System(0), Goal(1)，latest 4 message (latest 2 Turn)
            if len(self.messages) > 8:
                self.messages = [self.messages[0], self.messages[1]] + self.messages[-4:]
                console.print("[dim]🧹 Memory window pruned to prevent context overflow and restore token budget.[/dim]")
            # --------------------------------------------------------
            
            turn_dir = self.config.session_dir / f"{task_idx * 100 + current_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            (turn_dir / "prompt.md").write_text(self.messages[-1]["content"], encoding="utf-8")
            
            console.print(f"\n[bold yellow]>>> ReAct Turn {current_turn + 1} / {MAX_REACT_TURNS}[/bold yellow]")
            
            # --- 1. LLM Generation ---
            # Note: `stream` can be toggled by config.verbose inside llm_wrapper if supported
            content, actions = await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=self.messages,
                max_output_tokens=self.config.max_output,
                model_max_context=self.config.max_context, # Now uses auto-detected context
                provider=self.config.provider,
                session_dir=self.config.session_dir,
                tools=self.tools,
                tool_strategy="auto",
                allowlist=allowlist
            )
            
            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)
            
            # Print full response if verbose
            if self.config.verbose:
                console.print(Panel(content, title="LLM Response", border_style="blue"))
            
            # --- 2. Conclusion Check ---
            # --- 3. Action Segregation ---
            finish_action = next((a for a in actions if getattr(a, 'name', '') == 'finish_task'), None)
            if finish_action:
                console.print(f"[bold green]🏁 Agent explicitly finished the task: {finish_action.args.get('summary')}[/bold green]")
                self._save_trajectory_to_disk(task_idx, reward=1.0)
                return True
                
            if not actions:
                console.print("[yellow]Agent outputted nothing useful. Forcing format switch.[/yellow]")
                feedback = (
                    "⚠️ System Warning: No valid tool calls or file modifications were detected in your last response.\n"
                    "If you attempted to use a native JSON tool call, it may have failed parsing due to length or syntax errors.\n"
                    "Please fallback to using **Format B (WRITE_FILE)** directly in the Markdown text instead of using JSON tools for writing code.\n"
                    "If you are finished, use the `<tool_call><finish_task><summary>done</summary></finish_task></tool_call>` format."
                )
                self.messages.append({"role": "assistant", "content": content})
                self.messages.append({"role": "user", "content": feedback})
                self._log_rl("user", feedback)
                current_turn += 1
                continue
                
            # --- 3. Action Segregation ---
            info_actions = [a for a in actions if isinstance(a, ActionToolCall)]
            mutation_actions = [a for a in actions if isinstance(a, (ActionWriteFile, ActionApplyDiff, ActionReplaceText))]
            
            feedback_blocks = []
            
            # 3A. Execute Info Actions (Concurrently)
            if info_actions:
                concurrent_results = await self._execute_tools_concurrently(info_actions, turn_dir, allowlist)
                feedback_blocks.append(f"### Tool Execution Results\n{concurrent_results}")
                
            # 3B. Execute Mutation Actions (Sequentially)
            has_mutation = False
            if mutation_actions:
                # --- UI Polish: Print mutations cleanly ---
                console.print("\n[cyan]📝 Planning Code Mutations:[/cyan]")
                for act in mutation_actions:
                    if isinstance(act, ActionWriteFile):
                        path = getattr(act, 'path', 'Unknown')
                        size = len(getattr(act, 'content', ''))
                        console.print(f"  [dim]- WRITE_FILE: {path} ({size} chars)[/dim]")
                    elif isinstance(act, ActionApplyDiff):
                        size = len(getattr(act, 'diff_text', ''))
                        console.print(f"  [dim]- APPLY_DIFF: ({size} chars)[/dim]")

                # Human-in-the-loop Approval check
                if self.config.require_approval:
                    if not Confirm.ask("[bold red]⚠️ Approve these file modifications?[/bold red]"):
                        console.print("[yellow]Changes rejected by user.[/yellow]")
                        feedback_blocks.append("### User Override\nThe user REJECTED your file modifications. Please explain your approach or ask for clarification.")
                        mutation_actions = [] # Clear mutations
                
                if mutation_actions:
                    console.print("[cyan]💾 Applying to Disk (Atomic)...[/cyan]")
                    handler = UniversalToolHandler(self.config, turn_dir, allowlist)
                    # We pass the actions to the handler. 
                    # Ensure the handler itself doesn't print the raw content!
                    has_mutation, mutation_res = handler.execute(mutation_actions, content)
                    feedback_blocks.append(mutation_res)
            
            # --- 4. Environment Feedback & Verification ---
            if has_mutation:
                console.print(f"[cyan]-> Files Mutated. Triggering Verification Sandbox (Target: {self.config.sandbox_container or 'Local'})...[/cyan]")
                
                # Robust extraction of modified files
                modified_files = []
                for a in mutation_actions:
                    if hasattr(a, 'path'): modified_files.append(a.path)
                    elif isinstance(a, ActionApplyDiff):
                        paths = re.findall(r'^\+\+\+ b/(.+)$', a.diff_text, re.MULTILINE)
                        modified_files.extend(paths)
                modified_files = list(set(modified_files))
                
                auto_verify_cmd = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
                v_cmd = auto_verify_cmd.group(1).strip() if auto_verify_cmd else None
                verify_cmd = _determine_verify_cmd(allowlist, modified_files, v_cmd, self.config)
                
                if not verify_cmd:
                    console.print("[yellow]No verification command found/needed. Assuming Success.[/yellow]")
                    self._save_trajectory_to_disk(task_idx, reward=1.0)
                    return True
                
                # Approval for running commands
                if self.config.require_approval:
                    console.print(f"\n[bold red]⚠️ Agent intends to run a command: [/bold red] `[white]{verify_cmd}[/white]`")
                    if not Confirm.ask("Do you approve executing this command?"):
                        console.print("[yellow]Command rejected by user.[/yellow]")
                        self._save_trajectory_to_disk(task_idx, reward=1.0)
                        return True

                # Run Verification
                code, out = run_shell(verify_cmd, cap=20000, sandbox_container=self.config.sandbox_container)
                (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
                
                if code == 0:
                    console.print("[bold green]✅ Verification PASSED![/bold green]")
                    if self.config.verbose: console.print(f"[dim]{out}[/dim]")
                    self._save_trajectory_to_disk(task_idx, reward=1.0)
                    return True
                else:
                    console.print(f"[bold red]❌ Verification Failed (exit={code})[/bold red]")
                    
                    # [FIX 3]: filter the output to extract only relevant error messages, and provide actionable feedback to the agent
                    smart_error_context = build_debug_prompt(out, root_dir=str(self.config.workspace_dir))
                    
                    error_feedback = (
                        f"⚠️ [Verification Failed] Exit Code: {code}\n"
                        f"Command: {verify_cmd}\n"
                        f"{smart_error_context}\n"
                        f"Please analyze the error. Use tools (`search_code`, `read_file_chunk`) "
                        f"to investigate if necessary, then provide a fix using Format A (Diff) or Format B (WRITE_FILE)."
                    )
                    feedback_blocks.append(error_feedback)
            
            # --- 5. Append Observations and Loop ---
            combined_feedback = "\n\n".join(feedback_blocks)
            if not has_mutation and info_actions:
                combined_feedback += "\nAnalyze the results and continue."
                
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": combined_feedback})
            self._log_rl("user", combined_feedback)
            
            current_turn += 1

        console.print("[bold red]Max ReAct turns exceeded. Task aborted.[/bold red]")
        self._save_trajectory_to_disk(task_idx, reward=-1.0)
        return False


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
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--serper-key", default=os.environ.get("SERPER_API_KEY", ""), help="API key for Web Search")
    
    # System Features (New Additions)
    parser.add_argument("--output-dir", default="./agent_workspace", help="Default directory for agent to write files")
    parser.add_argument("--sandbox", default=None, help="Docker container name for sandbox execution (e.g. 'my_container')")
    parser.add_argument("--require-approval", action="store_true", help="Require user confirmation before writing files or executing commands")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of LLM responses and tool outputs")
    parser.add_argument("--max-context", type=int, default=16384, help="Fallback context length if auto-detect fails")
    
    args = parser.parse_args()

    # --- Directory Setup ---
    agent_dir = Path(".agent")
    ensure_dirs(agent_dir)
    session_dir = agent_dir / "sessions" / now_stamp()
    session_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL: Change working directory so all relative paths generated by LLM land here
    os.chdir(workspace_dir)

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
        provider=args.provider,
        sandbox_container=args.sandbox,
        verbose=args.verbose,
        serper_api_key=args.serper_key
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
    compiled_tools = get_active_tools(config.tool_strategy, config.provider)
    system_prompt = PromptRegistry.get_system_prompt(config.tool_strategy, compiled_tools)
    
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
python BatchAgent/agent_main.py

Search the web for the latest Python 3.13 features, and then write a python script named test_313.py demonstrating the new features.
"""
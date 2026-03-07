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
from BatchAgent.llm_wrapper import complete_with_continuation_async, get_compiled_tools, BASE_TOOLS
from BatchAgent.tool_handler import UniversalToolHandler
from BatchAgent.mini_batch_agent_libs import (
    now_stamp, ensure_dirs, run_shell, _determine_verify_cmd, 
    _handle_missing_modules, top_level_tree, read_file, estimate_tokens, truncate_to_tokens, robust_json_loads
)
from BatchAgent.mini_batch_agent import ActionApplyDiff, ActionToolCall, ActionWriteFile, ActionReplaceText

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
    provider: str = "openai"
    sandbox_container: Optional[str] = None
    verbose: bool = False
    max_retries: int = 4
    serper_api_key: str = ""

# ==========================================
# 2. Prompt Registry
# ==========================================
class PromptRegistry:
    """
    Centralized manager for all LLM prompts.
    Defines strict output formats, verification rules, and tool usages.
    """
    @classmethod
    def get_system_prompt(cls, tools_list: list) -> str:
        """Dynamically generates the system prompt based on available tools."""
        base_prompt = (
            "You are an elite, general-purpose AI Agent. You can write code, author documentation, and answer complex questions.\n"
            "You operate in a structured generation environment.\n\n"
            "## Output Format (STRICT)\n"
            "You MUST output in ONE of these three formats per response. Never mix them.\n\n"
            "### Format A: Unified Diff (For small code edits)\n"
            "1. Start with `## Action` followed by a SINGLE fenced diff block (`diff --git ...`).\n\n"
            "### Format B: WRITE_FILE (For new files, full rewrites, or writing Documents)\n"
            "WRITE_FILE: path/to/file.ext\n"
            "<<<CONTENT\n...\nCONTENT>>>\n\n"
            "### Format C: Direct Response (For Q&A, Summaries, and Tutorials)\n"
            "Output directly in Markdown.\n\n"
            "## Information Gathering & Anti-Hallucination Rules (CRITICAL)\n"
            "1. **Never Guess Code**: Check the `Provided File Context` section first.\n"
            "2. **Read Before Act**: If a requested file is NOT in the context, you MUST use `read_file_chunk` or `find_file` before modifying it.\n"
            "3. **Search for Unknowns**: Use `web_search` for unknown libraries, APIs, or events.\n\n"
            "## Verification & Environment Rules (CRITICAL)\n"
            "1. **Mandatory Verification**: If you write executable code, include a verification command on a standalone line:\n"
            "   `Verification: <command>` (e.g., `Verification: python3 script.py`)\n"
            "2. **Environment Limitations**: If the environment lacks dependencies (e.g., requires Python 3.13 but environment has 3.12):\n"
            "   - Write the correct code anyway.\n"
            "   - Use Format C to provide a step-by-step setup tutorial.\n"
            "   - Do NOT output a `Verification:` command that is guaranteed to fail.\n\n"
        )

        tool_section = "## Interactive Tools\nIf you lack context, output an XML tool call to pause and wait for data:\n"
        for t in tools_list:
            name = t.get("name")
            desc = t.get("description", "")
            # Construct dummy XML example
            args_xml = "".join([f"<{k}>...</{k}>" for k in t.get("properties", {}).keys()])
            tool_section += f"`<tool_call><{name}>{args_xml}</{name}></tool_call>`\n  - {desc}\n"

        return base_prompt + tool_section

    @staticmethod
    def format_task(goal: str, allowlist: List[str], context_files: List[str], notes: str, workspace_dir: str) -> str:
        """Builds the main Turn Prompt for the user message."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        
        all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False
        format_hint = ""
        if not allowlist or all_new_files:
            format_hint = "\n> **HINT**: You are creating NEW files or answering a query. Use **Format B (WRITE_FILE)** or **Format C**.\n"
        elif len(allowlist) > 1:
            format_hint = "\n> **HINT**: You are modifying multiple files. Use **Format B (WRITE_FILE)** to avoid diff truncation.\n"

        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Directory: `./` (inside `{workspace_dir}/`)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n{format_hint}\n"
            f"## Notes & Constraints\n{notes if notes else '(none)'}\n\n"
            f"## Output Contract Reminder\n"
            f"1. Choose ONE format: Format A (Diff), Format B (Write), or Format C (Direct Answer).\n"
            f"2. If generating code, remember the `Verification: <cmd>` rule.\n"
        )
        
        files_md = ""
        for f in list(dict.fromkeys(allowlist + context_files)):
            content = read_file(str(f))
            if content and not content.startswith("[MISSING FILE]"):
                if estimate_tokens(content) > 8000:
                    content = truncate_to_tokens(content, 8000)
                files_md += f"### File: {f}\n```text\n{content}\n```\n"
                
        if files_md:
            base_md += f"\n## Provided File Context\n{files_md}"

        return base_md


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
                # Human-in-the-loop Approval check
                if self.config.require_approval:
                    console.print("\n[bold red]⚠️ Agent intends to modify files on your disk.[/bold red]")
                    for act in mutation_actions:
                        path = getattr(act, 'path', 'Unknown Path')
                        console.print(f"  - Modifying: [cyan]{path}[/cyan]")
                    if not Confirm.ask("Do you approve these changes?"):
                        console.print("[yellow]Changes rejected by user.[/yellow]")
                        feedback_blocks.append("### User Override\nThe user REJECTED your file modifications. Please explain your approach or ask for clarification.")
                        mutation_actions = [] # Clear mutations
                
                if mutation_actions:
                    console.print("[cyan]📝 Applying Code Mutations to Disk (Atomic)...[/cyan]")
                    handler = UniversalToolHandler(self.config, turn_dir, allowlist)
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
                    error_feedback = (
                        f"⚠️ [Verification Failed] Exit Code: {code}\n"
                        f"Command: {verify_cmd}\n"
                        f"Output:\n```text\n{out[-4000:]}\n```\n"
                        f"Please analyze the error. Use tools (`search_code`, `read_file_chunk`, `web_search`) "
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
    compiled_tools = get_compiled_tools(config.provider)
    system_prompt = PromptRegistry.get_system_prompt(BASE_TOOLS)
    task_prompt = PromptRegistry.format_task(goal, allowlist, context_files, args.notes, workspace_dir.name)

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
import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
import httpx
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------
# Import our newly refactored robust components
# ---------------------------------------------------------
from BatchAgent.llm_wrapper import complete_with_continuation_async, get_compiled_tools
from BatchAgent.tool_handler import UniversalToolHandler
from CodeAgent.codeagent_libs import (
    now_stamp, ensure_dirs, run_shell, _determine_verify_cmd, 
    _handle_missing_modules, top_level_tree, read_file, estimate_tokens, truncate_to_tokens
)

console = Console()

# ==========================================
# 1. Configuration & Data Structures
# ==========================================
@dataclass
class AgentConfig:
    client: Any
    model: str
    session_dir: Path
    max_context: int
    max_output: int
    auto_approve: bool
    agent_dir: Path
    provider: str = "openai"
    sandbox_container: Optional[str] = None
    rl_mode: bool = False
    max_retries: int = 4
    batch_size: int = 1
    serper_api_key: str = ""
    model_max_context: int = 16384

# ==========================================
# 2. Prompt Registry
# ==========================================
class PromptRegistry:
    SYSTEM = (
        "You are an elite, general-purpose AI Agent. You can write code, author documentation, and answer complex questions.\n"
        "You operate in a structured generation environment.\n"
        "\n"
        "## Output Format (STRICT)\n"
        "You MUST output in ONE of these three formats per response. Never mix them.\n"
        "\n"
        "### Format A: Unified Diff (For small code edits)\n"
        "1. Start with `## Action` followed by a SINGLE fenced diff block (`diff --git ...`).\n"
        "\n"
        "### Format B: WRITE_FILE (For new files, full rewrites, or writing Documents)\n"
        "Use when creating new files, writing markdown documentation, or when diffs are too complex.\n"
        "WRITE_FILE: path/to/file.ext\n"
        "<<<CONTENT\n"
        "... content here ...\n"
        "CONTENT>>>\n"
        "\n"
        "### Format C: Direct Response (For Q&A and Information)\n"
        "If the user is asking a question, asking for a summary, or asking you to search the web, "
        "simply answer directly in Markdown format.\n"
        "\n"
        "## Interactive Tools\n"
        "If you lack context or need internet access, output an XML tool call to pause and wait for the system to provide data:\n"
        "`<tool_call><web_search><query>your query</query><category>news</category></web_search></tool_call>`\n"
        "`<tool_call><read_url><url>https://example.com</url></read_url></tool_call>`\n" # <--- ADD THIS LINE
        "`<tool_call><search_code><query>query</query></search_code></tool_call>`\n"
    )

    @staticmethod
    def format_task(goal: str, allowlist: List[str], context_files: List[str], notes: str, max_context: int) -> str:
        """Builds the main Turn Prompt for the user message."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        cwd = Path.cwd().name
        
        base_md = (
            f"# Agent Task\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workspace Context\n"
            f"Directory: `./` (inside `{cwd}/`)\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n\n"
            f"## Notes & Constraints\n{notes if notes else '(none)'}\n\n"
        )
        
        # Inject file contents
        files_md = ""
        for f in list(dict.fromkeys(allowlist + context_files)):
            content = read_file(str(f))
            if content and not content.startswith("[MISSING FILE]"):
                if estimate_tokens(content) > 8000:
                    content = truncate_to_tokens(content, 8000)
                files_md += f"### File: {f}\n```python\n{content}\n```\n"
                
        if files_md:
            base_md += f"\n## File Context\n{files_md}"

        return base_md


# ==========================================
# 3. Universal Agent Class (Qwen-Agent Style)
# ==========================================
class UniversalAgent:
    """
    An Object-Oriented Autonomous Agent that encapsulates the ReAct Loop.
    Designed with an interface similar to Qwen-Agent's `Assistant`.
    """
    def __init__(self, config: AgentConfig, system_message: str = None):
        self.config = config
        self.system_message = system_message or PromptRegistry.SYSTEM
        
        # Dynamically compile tools based on the Provider (OpenAI vs Anthropic)
        self.tools = get_compiled_tools(config.provider)
        
        # Memory / State
        self.messages: List[Dict[str, str]] = []
        self.rl_trajectory: List[Dict[str, str]] = []

    def _log_rl(self, role: str, content: str):
        """Records trajectory for future RL (DPO/PPO) fine-tuning."""
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
        except Exception as e:
            console.print(f"[dim]Failed to write RL trajectory log: {e}[/dim]")

    async def execute_task(self, task_goal: str, task_idx: int, allowlist: List[str], context_files: List[str], global_notes: str) -> bool:
        """
        The core Autonomous ReAct Loop (Reasoning + Acting + Verifying).
        """
        console.print(f"\n[bold green]=== Starting Agent Task {task_idx+1} ===[/bold green]")
        
        # Initialize Prompt
        prompt_md = PromptRegistry.format_task(task_goal, allowlist, context_files, global_notes, self.config.max_context)
        
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt_md}
        ]
        self._log_rl("system", self.system_message)
        self._log_rl("user", prompt_md)

        MAX_REACT_TURNS = 10
        current_turn = 0

        while current_turn < MAX_REACT_TURNS:
            turn_dir = self.config.session_dir / f"{task_idx * 100 + current_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)
            (turn_dir / "prompt.md").write_text(self.messages[-1]["content"], encoding="utf-8")
            
            console.print(f"\n[bold yellow]>>> ReAct Turn {current_turn + 1} / {MAX_REACT_TURNS}[/bold yellow]")
            
            # 1. LLM Generation Phase (The "Brain")
            # Automatically handles Tool Calling formats, context truncation, and repetition guards
            content, actions = await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=self.messages,
                max_output_tokens=self.config.max_output,
                model_max_context=self.config.model_max_context,
                provider=self.config.provider,
                session_dir=self.config.session_dir,
                tools=self.tools,
                tool_strategy="auto", # Uses native JSON first, falls back to text
                allowlist=allowlist
            )
            
            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)
            
            # 2. Evaluation Phase
            if not actions:
                console.print(Panel(content, title="Agent Conclusion", border_style="blue"))
                console.print("[green]Agent finished thinking/answering without further tool actions.[/green]")
                self._save_trajectory_to_disk(task_idx, 1.0)
                return True
                
            # 3. Action Phase (The "Hands and Eyes")
            tool_handler = UniversalToolHandler(self.config, turn_dir, allowlist)
            has_mutation, tool_results_str = tool_handler.execute(actions, content)
            
            # 4. Environment Feedback & Verification Phase
            if has_mutation:
                console.print("[cyan]-> File Mutated. Triggering Verification Sandbox...[/cyan]")
                
                #modified_files = [a.path for a in actions if hasattr(a, 'path')]
                # [FIXED] 强健的修改文件提取逻辑
                modified_files = []
                for a in actions:
                    if hasattr(a, 'path'):
                        modified_files.append(a.path)
                    elif isinstance(a, ActionApplyDiff):
                        # 从 diff 文本中用正则提取目标文件路径 (+++ b/filepath)
                        paths = re.findall(r'^\+\+\+ b/(.+)$', a.diff_text, re.MULTILINE)
                        modified_files.extend(paths)
                modified_files = list(set(modified_files)) # 去重
                
                auto_verify_cmd = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
                v_cmd = auto_verify_cmd.group(1).strip() if auto_verify_cmd else None
                
                verify_cmd = _determine_verify_cmd(allowlist, modified_files, v_cmd, self.config)
                
                if not verify_cmd:
                    console.print("[yellow]No verification command found/needed. Assuming Success.[/yellow]")
                    self._save_trajectory_to_disk(task_idx, 1.0)
                    return True
                    
                code, out = run_shell(verify_cmd, cap=20000, sandbox_container=self.config.sandbox_container)
                (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
                
                if code == 0:
                    console.print("[bold green]✅ Verification PASSED![/bold green]")
                    self._save_trajectory_to_disk(task_idx, 1.0)
                    return True
                else:
                    console.print(f"[bold red]❌ Verification Failed (exit={code})[/bold red]")
                    
                    # Feed the Error back to the LLM as environment observation!
                    error_feedback = (
                        f"⚠️ [Verification Failed] Exit Code: {code}\n"
                        f"Command: {verify_cmd}\n"
                        f"Output:\n```text\n{out[-4000:]}\n```\n"
                        f"Please analyze the error. You may use tools (`search_code`, `read_file_chunk`, `web_search`) "
                        f"to investigate, then provide a fix using Format A (Diff) or Format B (WRITE_FILE)."
                    )
                    self.messages.append({"role": "assistant", "content": content})
                    self.messages.append({"role": "user", "content": error_feedback})
                    self._log_rl("user", error_feedback)
            
            else:
                # Information gathering actions (Web Search, File Read)
                self.messages.append({"role": "assistant", "content": content})
                feedback = f"Tool Results:\n```text\n{tool_results_str}\n```\nAnalyze the results and continue."
                self.messages.append({"role": "user", "content": feedback})
                self._log_rl("user", feedback)
                
            current_turn += 1

        console.print("[bold red]Max ReAct turns exceeded. Task aborted.[/bold red]")
        self._save_trajectory_to_disk(task_idx, -1.0) # Negative reward for failing to finish
        return False


# ==========================================
# 4. Main Entry Point
# ==========================================
async def main_async():
    import argparse
    parser = argparse.ArgumentParser(description="Universal Autonomous Agent")
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
    
    args = parser.parse_args()

    agent_dir = Path(".agent")
    ensure_dirs(agent_dir)
    session_dir = agent_dir / "sessions" / now_stamp()
    session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Async Client
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

    # 2. Build Configuration
    config = AgentConfig(
        client=client,
        model=args.model,
        session_dir=session_dir,
        max_context=16000,
        max_output=4096,
        auto_approve=True, # Auto-approve for Autonomous mode
        agent_dir=agent_dir,
        provider=args.provider,
        serper_api_key=args.serper_key
    )

    console.print(Panel(
        f"Session: {session_dir.name}\nProvider: {args.provider}\nModel: {args.model}\nMode: Autonomous ReAct Loop",
        title="Universal AI Agent Started", style="cyan"
    ))

    # 3. Interactive Goal Collection (if not provided in args)
    goal = args.goal
    if not goal:
        goal = Prompt.ask("[bold green]Enter your Goal[/bold green]").strip()

    allowlist = [x.strip() for x in args.allowlist.split(",") if x.strip()]
    context_files = [x.strip() for x in args.context.split(",") if x.strip()]

    # 4. Instantiate and Run the Agent
    agent = UniversalAgent(config=config)
    
    # In a full setup, you might run `plan_tasks` here and loop over subtasks.
    # For now, we execute the main goal as Task 0.
    success = await agent.execute_task(
        task_goal=goal,
        task_idx=0,
        allowlist=allowlist,
        context_files=context_files,
        global_notes=args.notes
    )

    if success:
        console.print("\n[bold green]🎉 Agent completed the task successfully![/bold green]")
    else:
        console.print("\n[bold red]💀 Agent failed to complete the task.[/bold red]")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent execution interrupted by user.[/yellow]")

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
    
python BatchAgent/agent_main.py

Search the web for the latest Python 3.13 features, and then write a python script named test_313.py demonstrating the new features.
"""
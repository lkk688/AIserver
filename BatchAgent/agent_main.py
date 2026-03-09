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
    parallel_thinking: bool = False  # [NEW] 开关：是否允许并发思维分叉
    provider: str = "openai"
    sandbox_container: Optional[str] = None
    verbose: bool = False
    max_retries: int = 4
    serper_api_key: str = ""
    domain: str = "general"
    location: str = "California, United States"




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
        
        # [NEW] 动态自定义工具库
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

    
    def _prune_memory(self):
        """智能滑动窗口：保留系统提示词、最初的 Goal、以及最近的推理轨迹"""
        # 放宽历史记忆的条数，从 8 条放宽到 12 条，给它更多的思考空间
        if len(self.messages) > 12:
            # 保留: System Prompt (0), User Goal (1)
            # 保留: 最近的 10 条交互 (确保它能记得自己刚刚发现了什么)
            self.messages = [self.messages[0], self.messages[1]] + self.messages[-10:]
            console.print("[dim]🧹 Memory window gracefully pruned (kept last 10 turns).[/dim]")

    async def _execute_batch_brainstorm(self, action: ActionToolCall, allowlist: List[str]) -> str:
        """
        核心重构：将复杂的子 Agent 降级为廉价且纯粹的 Batch LLM Calls。
        利用 vLLM 的连续批处理能力，并发请求，汇总为纯文本。
        """
        problem = action.args.get('problem_statement', 'Help me solve this task.')
        n_vars = min(4, max(2, int(action.args.get('n_variations', 3))))
        
        console.print(f"\n[bold magenta]🧠 Launching {n_vars} Batch LLM Calls for Parallel Thinking...[/bold magenta]")
        
        # 构造一个纯净的 Prompt，不要给工具，只要它“想”
        bs_messages = self.messages + [{
            "role": "user", 
            "content": f"Please brainstorm a unique strategy to solve: '{problem}'. Do NOT write final code. Just provide the step-by-step logic."
        }]
        
        async def _fetch_idea(idx: int) -> str:
            # 动态调整温度以增加策略多样性
            div_temp = min(0.9, getattr(self.config, 'temperature', 0.2) + 0.3 + (idx * 0.15))
            
            # 从外部引入你的 complete_with_continuation_async
            from BatchAgent.llm_wrapper import complete_with_continuation_async
            
            content, _ = await complete_with_continuation_async(
                client=self.config.client,
                model=self.config.model,
                messages=bs_messages,
                temperature=div_temp,
                max_output_tokens=1024, # 限制思维发散的长度
                model_max_context=self.config.max_context,
                provider=self.config.provider,
                stream=False,   # 屏蔽子调用的终端打印
                verbose=False,
                tool_strategy="text", # 强制纯文本模式
                allowlist=allowlist
            )
            return f"### Approach {idx + 1} (Temp {div_temp:.2f}):\n{content}\n"

        bs_start = time.time()
        # 万箭齐发，榨干 vLLM
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
        独立的沙盒验证模块
        返回: (is_success, feedback_string)
        """
        # 1. 提取被修改的文件
        modified_files = []
        for a in mutation_actions:
            if hasattr(a, 'path'): modified_files.append(a.path)
            elif isinstance(a, ActionApplyDiff):
                paths = re.findall(r'^\+\+\+ b/(.+)$', getattr(a, 'diff_text', ''), re.MULTILINE)
                modified_files.extend(paths)
        modified_files = list(set(modified_files))
        
        # 2. 推断验证命令
        auto_verify_cmd = re.search(r"^Verification:\s*(.+)$", content, re.MULTILINE)
        v_cmd = auto_verify_cmd.group(1).strip() if auto_verify_cmd else None
        verify_cmd = _determine_verify_cmd(allowlist, modified_files, v_cmd, self.config)
        
        if not verify_cmd:
            console.print("[yellow]No verification command found/needed. Assuming Success.[/yellow]")
            # MUST return a string, not a tuple!
            return "✅ File modifications applied successfully. No verification command was specified or required for this action."
            
        # 3. 用户授权检查
        if getattr(self.config, 'require_approval', False):
            console.print(f"\n[bold red]⚠️ Agent intends to run a command: [/bold red] `[white]{verify_cmd}[/white]`")
            from rich.prompt import Confirm
            if not Confirm.ask("Do you approve executing this command?"):
                return f"⚠️ User bypassed verification for command: {verify_cmd}"

        # 4. 执行命令
        console.print(f"[cyan]-> Running Verification Sandbox...[/cyan]")
        code, out = run_shell(verify_cmd, cap=20000, sandbox_container=getattr(self.config, 'sandbox_container', None))
        (turn_dir / "verify_stdout.txt").write_text(out, encoding='utf-8')
        
        # [FIX] 不再武断地 return True。把裁判权交给 Agent！
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
    async def execute_task(self, task_goal: str, task_idx: int, allowlist: List[str], prompt_md: str) -> bool:
        
        console.print(f"\n[bold green]=== Starting Agent Task {task_idx+1} ===[/bold green]")
        self.messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": prompt_md}]
        self._log_rl("system", self.system_message)
        self._log_rl("user", prompt_md)

        MAX_REACT_TURNS = 15
        from BatchAgent.llm_wrapper import complete_with_continuation_async # Local import
        
        for current_turn in range(MAX_REACT_TURNS):
            self._prune_memory()
            
            turn_dir = self.config.session_dir / f"{task_idx * 100 + current_turn:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)

            # ===============================================================
            # [NEW] 1. 落盘：完整的 Prompt 上下文 (最关键的调试文件)
            # ===============================================================
            (turn_dir / "full_prompt_context.json").write_text(
                json.dumps(self.messages, indent=2, ensure_ascii=False), 
                encoding="utf-8"
            )
            # 保留一个人类易读的版本，只看最新的一条指令
            (turn_dir / "latest_instruction.md").write_text(self.messages[-1]["content"], encoding="utf-8")
            
            console.print(f"\n[bold yellow]>>> ReAct Turn {current_turn + 1} / {MAX_REACT_TURNS}[/bold yellow]")
            
            # --- 1. LLM Generation ---
            content, actions = await complete_with_continuation_async(
                client=self.config.client, model=self.config.model, messages=self.messages,
                temperature=getattr(self.config, 'temperature', 0.2), max_output_tokens=self.config.max_output,
                model_max_context=self.config.max_context, provider=self.config.provider,
                stream=True, verbose=self.config.verbose, session_dir=self.config.session_dir,
                tools=self.tools, tool_strategy=self.config.tool_strategy, allowlist=allowlist
            )
            
            (turn_dir / "response.md").write_text(content, encoding="utf-8")
            self._log_rl("assistant", content)

            # ===============================================================
            # [NEW] 2. 落盘：引擎解析出的 Action 列表
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
            if next((a for a in actions if getattr(a, 'name', '') == 'finish_task'), None):
                console.print("[bold green]🏁 Agent explicitly finished the task.[/bold green]")
                self._save_trajectory_to_disk(task_idx, reward=1.0)
                return True
                
            if not actions:
                console.print("[yellow]Agent outputted nothing useful. Forcing format switch.[/yellow]")
                
                # [FIX 2: 动态错误反馈，防止逻辑冲突]
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

                self.messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": feedback}])
                self._log_rl("user", feedback)
                current_turn += 1
                continue
                
            # --- 3. Action Segregation ---
            info_actions = [a for a in actions if isinstance(a, ActionToolCall)]
            mutation_actions = [a for a in actions if isinstance(a, (ActionWriteFile, ActionApplyDiff, ActionReplaceText))]
            feedback_blocks = []

            # =====================================================================
            # [NEW] 插件系统：动态加载领域工具 (Hot-Swapping)
            # =====================================================================
            load_domain_action = next((a for a in info_actions if getattr(a, 'name', '') == 'load_domain_tools'), None)
            
            if load_domain_action:
                new_domain = load_domain_action.args.get('domain', '')
                console.print(f"\n[bold green]🔌 Agent is hot-swapping domain plugin: [white]{new_domain}[/white][/bold green]")
                
                if new_domain in DOMAIN_REGISTRY:
                    # 1. 更新 Config 状态
                    self.config.domain = new_domain
                    
                    # 2. 重新获取并编译包含新领域工具的列表
                    new_base_tools = get_base_tools(
                        strategy=self.config.tool_strategy, 
                        enable_parallel=getattr(self.config, 'parallel_thinking', False), 
                        domain=new_domain
                    )
                    self.tools = compile_tools_for_provider(new_base_tools, self.config.provider, self.config.tool_strategy)
                    
                    # 3. 重新生成 System Prompt (不仅包含新工具的 XML 描述，还包含新的人设 Domain Profile)
                    self.system_message = PromptRegistry.get_system_prompt(self.config.tool_strategy, new_base_tools, new_domain)
                    
                    # 4. 【核心魔术】修改第一条历史记录，悄无声息地替换 Agent 的大脑设定！
                    self.messages[0]["content"] = self.system_message
                    
                    tool_names = [t['name'] for t in DOMAIN_REGISTRY[new_domain]]
                    load_feedback = (
                        f"✅ Successfully loaded the '{new_domain}' plugin!\n"
                        f"**New tools available**: {', '.join(tool_names)}.\n"
                        f"Please proceed to use these new tools to complete your task."
                    )
                else:
                    load_feedback = f"❌ Failed to load domain. '{new_domain}' is not a valid domain plugin."
                
                # 记录这轮操作并继续循环，让模型在下一轮立即使用新工具
                self.messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": load_feedback}])
                self._log_rl("user", load_feedback)
                current_turn += 1
                continue
            # =====================================================================

            # --- 4. Handle Parallel Brainstorming (Optional / Configurable) ---
            # 只有在 config 开启了 enable_parallel_thinking 时，才会注入这个工具并处理
            brainstorm_action = next((a for a in info_actions if getattr(a, 'name', '') == 'brainstorm_solutions'), None)
            if brainstorm_action and getattr(self.config, 'enable_parallel_thinking', False):
                bs_feedback = await self._execute_batch_brainstorm(brainstorm_action, allowlist)
                self.messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": bs_feedback}])
                self._log_rl("user", bs_feedback)
                continue # 拿到思路后进入下一轮，不执行其他工具

            # --- 5. Execute Standard Tools (Concurrently) ---
            # 过滤掉特殊的元工具
            standard_info_actions = [a for a in info_actions if a.name not in ['brainstorm_solutions', 'finish_task']]
            if standard_info_actions:
                concurrent_results = await self._execute_tools_concurrently(standard_info_actions, turn_dir, allowlist)
                feedback_blocks.append(f"### Tool Execution Results\n{concurrent_results}")
            
            # =====================================================================
            # [NEW] 创造者模式：注册并热重载自定义工具
            # =====================================================================
            register_action = next((a for a in info_actions if getattr(a, 'name', '') == 'register_custom_tool'), None)
            
            if register_action:
                args = register_action.args
                t_name = args.get("tool_name", "custom_tool")
                
                console.print(f"\n[bold green]🛠️ Agent created a new tool: [white]{t_name}[/white][/bold green]")
                
                # 构建原生 API 需要的 Schema 格式
                new_schema = {
                    "name": t_name,
                    "description": args.get("description", ""),
                    "properties": args.get("schema_properties", {}),
                    "required": args.get("required_args", [])
                }
                
                # 存入动态记忆库
                self.dynamic_tools_schema.append(new_schema)
                self.dynamic_tools_mapping[t_name] = args.get("script_path", "")
                
                # 热重载：重新编译工具并更新 System Prompt
                from tools_registry import compile_tools_for_provider
                from prompt_registry import PromptRegistry
                
                # 获取基础工具并追加刚刚创建的新工具
                from tools_registry import get_base_tools
                current_base_tools = get_base_tools(self.config.tool_strategy, getattr(self.config, 'parallel_thinking', False), getattr(self.config, 'domain', 'auto'))
                current_base_tools.extend(self.dynamic_tools_schema)
                
                # 更新底层大模型的可用工具
                self.tools = compile_tools_for_provider(current_base_tools, self.config.provider, self.config.tool_strategy)
                
                # 更新系统提示词中的教学 XML (如果它是 text_only 或 hybrid)
                self.system_message = PromptRegistry.get_system_prompt(self.config.tool_strategy, current_base_tools, getattr(self.config, 'domain', 'auto'))
                self.messages[0]["content"] = self.system_message
                
                feedback = (
                    f"✅ Successfully compiled and registered the new tool `{t_name}`!\n"
                    f"The tool is now loaded into your context. You can call it immediately just like any other native tool."
                )
                
                self.messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": feedback}])
                self._log_rl("user", feedback)
                current_turn += 1
                continue
            
            # --- 6. Execute Mutations & Verify ---
            if mutation_actions:
                handler = UniversalToolHandler(self.config, turn_dir, allowlist, dynamic_tools_mapping=self.dynamic_tools_mapping)
                has_mutation, mutation_res = handler.execute(mutation_actions, content)
                feedback_blocks.append(mutation_res)
                
                if has_mutation:
                    # [FIX] 收集沙盒输出，并不中断循环
                    v_feedback = self._verify_mutations(content, mutation_actions, allowlist, turn_dir)
                    if v_feedback:
                        feedback_blocks.append(v_feedback)
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
            
            # ===============================================================
            # [NEW] 3. 落盘：工具执行的返回结果 (Feedback)
            # ===============================================================
            if combined_feedback:
                (turn_dir / "tool_execution_results.md").write_text(combined_feedback, encoding="utf-8")

            self.messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": combined_feedback}])
            self._log_rl("user", combined_feedback)

        console.print("[bold red]Max ReAct turns exceeded. Task aborted.[/bold red]")
        self._save_trajectory_to_disk(task_idx, reward=-1.0)
        return False

    # [old version]
    async def execute_task_v1(self, task_goal: str, task_idx: int, allowlist: List[str], prompt_md: str) -> bool:
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

            # =====================================================================
            # [NEW] 核心架构：并行分支与懒加载 (Agentic Map-Reduce)
            # =====================================================================
            parallel_action = next((a for a in info_actions if getattr(a, 'name', '') == 'execute_parallel_branches'), None)
            inspect_action = next((a for a in info_actions if getattr(a, 'name', '') == 'inspect_branch_details'), None)
            
            if parallel_action:
                branches = parallel_action.args.get('branches', [])
                console.print(f"\n[bold magenta]🚀 Launching {len(branches)} Parallel Branches...[/bold magenta]")
                
                # 初始化全局分支记录字典 (如果还不存在的话)
                if not hasattr(self, 'branch_records'):
                    self.branch_records = {}

                async def _run_branch(branch: dict) -> str:
                    b_id = branch.get('branch_id', 'unknown')
                    b_inst = branch.get('instruction', '')
                    
                    # 1. 为子分支创建物理隔离的 Workspace，防止并行写文件冲突
                    b_workspace = self.config.workspace_dir / b_id
                    b_workspace.mkdir(parents=True, exist_ok=True)
                    
                    # 2. 继承配置，但指向新的工作区。为了防止无限递归，关闭子 Agent 的并发能力
                    b_config = AgentConfig(**{**self.config.__dict__, "workspace_dir": b_workspace, "verbose": False})
                    
                    # 3. 剥离并行工具，防止子 Agent 再次召唤孙子 Agent 导致死循环
                    b_tools = [t for t in self.tools if t.get('name') not in ['execute_parallel_branches', 'inspect_branch_details']]
                    
                    # 4. 构造子 Agent Prompt
                    b_goal = f"MAIN TASK CONTEXT: {task_goal}\n\nYOUR SPECIFIC BRANCH MISSION: {b_inst}\n\nIMPORTANT: You are running in an isolated directory `{b_id}/`. Write code, test it, and use `finish_task` to provide a detailed summary of your results, test outputs, and discoveries."
                    b_prompt = PromptRegistry.format_task(b_goal, [], [], b_workspace.name, content_injector)
                    
                    # 5. 执行子 Agent
                    sub_agent = UniversalAgent(config=b_config, system_message=self.system_message, tools=b_tools)
                    success = await sub_agent.execute_task(task_goal=b_goal, task_idx=task_idx+100, allowlist=[], prompt_md=b_prompt)
                    
                    # 6. 提取精华：子 Agent 的最后一个 finish_task summary 或最后的回复
                    last_msg = next((m['content'] for m in reversed(sub_agent.messages) if m['role'] == 'assistant'), "")
                    
                    # 7. 保存全量状态到内存，供后续 inspect_branch_details 懒加载
                    self.branch_records[b_id] = {
                        "instruction": b_inst,
                        "workspace": str(b_workspace),
                        "full_trajectory": "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in sub_agent.messages]),
                        "success": success
                    }
                    
                    # 8. 只返回轻量级摘要给主 Agent
                    status_emoji = "✅" if success else "❌"
                    return f"### Branch: {b_id} {status_emoji}\n**Instruction**: {b_inst}\n**Summary**: {last_msg[:800]}..."

                # 并发执行所有分支
                branch_start = time.time()
                branch_results = await asyncio.gather(*(_run_branch(b) for b in branches), return_exceptions=True)
                
                valid_summaries = [res for res in branch_results if not isinstance(res, Exception)]
                errors = [str(res) for res in branch_results if isinstance(res, Exception)]
                
                combined_report = "\n\n".join(valid_summaries)
                if errors: combined_report += f"\n\n**Framework Errors:**\n" + "\n".join(errors)
                
                console.print(f"[dim]Parallel Branches completed in {time.time() - branch_start:.1f}s[/dim]")
                
                feedback = (
                    f"### Parallel Execution Report\n"
                    f"The branches have completed. Below are their high-level summaries. Their files are saved in subdirectories named after their `branch_id`.\n\n"
                    f"{combined_report}\n\n"
                    f"**Next Steps**:\n"
                    f"- If you need to see the exact code, tests, or full search text from a branch, use `<tool_call><inspect_branch_details><branch_id>...</branch_id></inspect_branch_details>`.\n"
                    f"- Once you have enough info, you can copy the best code to the main directory using bash, or synthesize the final report using Format B / Format C."
                )
                
                self.messages.append({"role": "assistant", "content": content})
                self.messages.append({"role": "user", "content": feedback})
                self._log_rl("user", feedback)
                
                current_turn += 1
                continue

            elif inspect_action:
                # =====================================================================
                # [NEW] 懒加载查阅工具处理
                # =====================================================================
                b_id = inspect_action.args.get('branch_id', '')
                console.print(f"\n[cyan]📖 Agent is inspecting details of branch: {b_id}[/cyan]")
                
                if hasattr(self, 'branch_records') and b_id in self.branch_records:
                    record = self.branch_records[b_id]
                    # 我们不仅返回它的轨迹，还可以自动去它的目录下读取所有文件内容
                    import glob
                    files_content = ""
                    for filepath in glob.glob(f"{record['workspace']}/**/*", recursive=True):
                        if os.path.isfile(filepath) and not filepath.endswith('.pyc'):
                            try:
                                with open(filepath, 'r') as f:
                                    files_content += f"\n--- File: {os.path.basename(filepath)} ---\n```\n{f.read()[:4000]}\n```" # 截断防爆
                            except Exception: pass
                    
                    feedback = (
                        f"### Details for Branch: {b_id}\n"
                        f"**Workspace**: `{record['workspace']}/`\n"
                        f"**Generated Files**: {files_content if files_content else 'No files generated.'}\n\n"
                        f"**Execution Trajectory Snippet**:\n"
                        f"{record['full_trajectory'][-3000:]}" # 只给最后 3000 字符，通常包含最终代码和测试结果
                    )
                else:
                    feedback = f"Error: Branch ID '{b_id}' not found. Please check the ID."

                self.messages.append({"role": "assistant", "content": content})
                self.messages.append({"role": "user", "content": feedback})
                self._log_rl("user", feedback)
                
                current_turn += 1
                continue
            
            ## Back to Normal processing
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
                        default="native_all",
                        help="Choose how the LLM interacts with tools.")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="LLM Provider")
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
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
        parallel_thinking=args.parallel_thinking,
        provider=args.provider,
        sandbox_container=args.sandbox,
        verbose=args.verbose,
        serper_api_key=args.serper_key,
        domain=args.domain,
        location=args.location
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
        location=config.location
    )
    # compiled_tools = get_active_tools(config.tool_strategy, config.provider)
    # system_prompt = PromptRegistry.get_system_prompt(config.tool_strategy, compiled_tools)
    
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
"""
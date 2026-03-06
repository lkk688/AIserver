import os
import asyncio
from typing import Any, List, Tuple
from pathlib import Path
import sys
from rich.console import Console
from rich.panel import Panel

# Import different async clients
from openai import AsyncOpenAI
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import Wrapper and Action classes
from BatchAgent.llm_wrapper import complete_with_continuation_async
from BatchAgent.mini_batch_agent import (
    ActionToolCall, ActionWriteFile, ActionApplyDiff, ActionReplaceText
)

# If your codeagent_libs contains these, ensure they are importable
from CodeAgent.codeagent_libs import estimate_tokens, top_level_tree, truncate_to_tokens

console = Console()

# ==========================================
# 1. Native Tools Schema Definitions
# ==========================================

# Standard OpenAI / vLLM Tool Format
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

# Standard Anthropic Tool Format (Flattened)
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
# 2. Prompt Registry
# ==========================================
class PromptRegistry:
    """
    Centralized manager for all LLM prompts.
    Optimized to reduce token waste and clearly define output formats.
    """

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
        "If the user is just asking a question, asking for a summary, or asking you to search the web, "
        "simply answer directly in Markdown format. Do NOT use Format A or B unless saving to a file is required.\n"
        "\n"
        "## Interactive Tools\n"
        "If you lack context or need internet access, output an XML tool call to pause and wait for the system to provide data:\n"
        "`<tool_call><web_search>your query</web_search></tool_call>`\n"
    )

    @staticmethod
    def format_task(
        goal: str,
        allowlist: List[str],
        max_context: int,
        file_content_mock: str = "", 
    ) -> str:
        """Builds the main Turn Prompt for the user message."""
        allow_txt = "\n".join(f"- {p}" for p in allowlist) if allowlist else "- (none)"
        all_new_files = all(not Path(f).exists() for f in allowlist) if allowlist else False

        format_hint = ""
        if not allowlist or all_new_files:
            format_hint = "\n> **IMPORTANT**: You are creating NEW code. Use **Format B (WRITE_FILE)**.\n"
        elif len(allowlist) > 1:
            format_hint = "\n> **IMPORTANT**: Use **Format B (WRITE_FILE)** to create all files.\n"

        base_md = (
            f"# Turn Prompt\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Target Files (Allowlist)\n{allow_txt}\n"
            f"{format_hint}\n"
            f"## Output Contract\n"
            f"1. Return changes using Format A, Format B, OR Format C.\n"
            f"2. ALL files in the Target Files list must be addressed.\n"
        )

        # Inject mock file content for testing purposes
        if allowlist and file_content_mock:
            base_md += f"\n## Context\n### File: {allowlist[0]}\n```python\n{file_content_mock}\n```\n"

        return base_md


# ==========================================
# 3. Universal Test Executor
# ==========================================

async def run_scenario_test(
    scenario_name: str, 
    client: Any, 
    model: str, 
    provider: str, 
    tools: list, 
    goal: str, 
    allowlist: List[str], 
    mock_content: str = "",
    expected_action_type: Any = None
):
    console.print(f"\n[bold magenta]=== Running Scenario: {scenario_name} ===[/bold magenta]")
    
    # Use PromptRegistry to build the exact prompt the agent will use
    user_prompt = PromptRegistry.format_task(
        goal=goal,
        allowlist=allowlist,
        max_context=8192,
        file_content_mock=mock_content
    )
    
    messages = [
        {"role": "system", "content": PromptRegistry.SYSTEM},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        content, actions = await complete_with_continuation_async(
            client=client,
            model=model,
            messages=messages,
            provider=provider,
            tools=tools,
            tool_strategy="auto", # Prioritize native JSON tool calling
            max_output_tokens=1024,
            stream=True if provider == "openai" else False,
            allowlist=allowlist
        )
        
        console.print(Panel(content if content else "[Tool Called Directly, No Content]", title="Generated Content", border_style="green"))
        
        console.print("[bold cyan]Parsed Actions:[/bold cyan]")
        if not actions:
            if expected_action_type is None:
                console.print("✅ Expected no specific file actions (Format C response).")
            else:
                console.print("[red]❌ Test Failed: No actions parsed![/red]")
        else:
            for act in actions:
                if isinstance(act, ActionToolCall):
                    console.print(f"🛠️  [yellow]ToolCall Detected:[/yellow] {act.name}({act.args})")
                elif isinstance(act, ActionWriteFile):
                    console.print(f"📝 [blue]WriteFile Detected:[/blue] {act.path} (Length: {len(act.content)} chars)")
                elif isinstance(act, ActionApplyDiff):
                    console.print(f"✂️  [magenta]ApplyDiff Detected:[/magenta]\n{act.diff_text[:100]}...")
                
                # Check if it matches expectation
                if expected_action_type and isinstance(act, expected_action_type):
                    # Handle tuple of expected types gracefully
                    if isinstance(expected_action_type, tuple):
                        expected_names = " or ".join([t.__name__ for t in expected_action_type])
                        console.print(f"✅ Successfully matched expected action type: {expected_names}")
                    else:
                        console.print(f"✅ Successfully matched expected action type: {expected_action_type.__name__}")
                    
    except Exception as e:
        console.print(f"[red]Execution Error: {e}[/red]")


# ==========================================
# 4. Main Test Suite
# ==========================================

async def main():
    console.print("[bold green]Starting Universal Agent Capability Test Suite...[/bold green]")

    # ---------------------------------------------------------
    # Test Block 1: Local vLLM (Qwen)
    # Tests all 3 behaviors: Web Search, File Creation, File Modification
    # ---------------------------------------------------------
    vllm_client = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1", 
        api_key="EMPTY", 
    )
    
    await run_scenario_test(
        scenario_name="vLLM Task 1: News Search (Artemis Mission)",
        client=vllm_client,
        model="qwen3.5-9b",
        provider="openai",
        tools=OPENAI_TOOLS,
        goal="Search for the latest news regarding the Artemis lunar mission in 2026 and provide a brief summary.",
        allowlist=[], 
        expected_action_type=ActionToolCall
    )

    await run_scenario_test(
        scenario_name="vLLM Task 2: Generate New Code File",
        client=vllm_client,
        model="qwen3.5-9b",
        provider="openai",
        tools=OPENAI_TOOLS,
        goal="Write a complete Python script to calculate the Fibonacci sequence. It must include a main block.",
        allowlist=["fibonacci.py"], 
        expected_action_type=ActionWriteFile
    )

    mock_existing_code = (
        "def get_fibonacci(n):\n"
        "    if n <= 1: return n\n"
        "    return get_fibonacci(n-1) + get_fibonacci(n-2)\n"
    )
    
    await run_scenario_test(
        scenario_name="vLLM Task 3: Modify Existing Code",
        client=vllm_client,
        model="qwen3.5-9b",
        provider="openai",
        tools=OPENAI_TOOLS,
        goal="Add type hinting (int) to the existing get_fibonacci function and add a docstring.",
        allowlist=["fibonacci.py"],
        mock_content=mock_existing_code,
        expected_action_type=(ActionApplyDiff, ActionWriteFile) # Either diff or rewrite is acceptable
    )

    # ---------------------------------------------------------
    # Test Block 2: Official OpenAI API
    # Testing fundamental tool calling capability
    # ---------------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        openai_client = AsyncOpenAI(api_key=openai_key)
        await run_scenario_test(
            scenario_name="OpenAI Task: Tool Calling (gpt-4o-mini)", 
            client=openai_client, 
            model="gpt-4o-mini", 
            provider="openai", 
            tools=OPENAI_TOOLS,
            goal="Search for the latest news regarding the Artemis lunar mission in 2026 and provide a brief summary.",
            allowlist=[],
            expected_action_type=ActionToolCall
        )
    else:
        console.print("\n[dim]Skipping OpenAI tests (OPENAI_API_KEY not set).[/dim]")

    # ---------------------------------------------------------
    # Test Block 3: Official Anthropic API
    # Testing fundamental tool calling capability
    # ---------------------------------------------------------
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key and AsyncAnthropic:
        anthropic_client = AsyncAnthropic(api_key=anthropic_key)
        await run_scenario_test(
            scenario_name="Anthropic Task: Tool Calling (Claude 3.5 Sonnet)", 
            client=anthropic_client, 
            model="claude-3-5-sonnet-20241022", 
            provider="anthropic", 
            tools=ANTHROPIC_TOOLS,
            goal="Search for the latest news regarding the Artemis lunar mission in 2026 and provide a brief summary.",
            allowlist=[],
            expected_action_type=ActionToolCall
        )
    else:
        console.print("\n[dim]Skipping Anthropic tests (ANTHROPIC_API_KEY not set or library missing).[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
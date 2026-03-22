import asyncio
import nest_asyncio
nest_asyncio.apply()

from rich.console import Console
from rich.table import Table
from openai import AsyncOpenAI
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BatchAgent.prompt_registry_v2 import PromptRegistry
from BatchAgent.tools.tools_registry import get_base_tools, compile_tools_for_provider
from BatchAgent.tools.tool_registry_runtime import configure_global_tool_registry
from BatchAgent.llm_wrapper_v2 import complete_with_continuation_async
from BatchAgent.mini_batch_agent_base import ActionToolCall, ActionWriteFile, ActionReplaceText, ActionApplyDiff
from BatchAgent.tools.universal_tool_handler import UniversalToolHandler, ToolExecutionContext

import os
import tempfile
os.environ["EMBEDDING_BASE_URL"] = "http://localhost:8081/v1"
os.environ["EMBEDDING_API_KEY"] = "EMPTY"
os.environ["EMBEDDING_MODEL"] = "text-embeddings-inference"

PDF_PATH = "data/sote/SP24_CMPE131_01.pdf"
OCR_SERVER = "http://localhost:8002/v1"
OCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
OCR_WORKSPACE = "./output_old/tmp_ocr"

class MockConfig:
    def __init__(self, strategy, working_memory=None):
        self.tool_strategy = strategy
        self.ocr_server = OCR_SERVER
        self.ocr_model = OCR_MODEL
        self.ocr_workspace = OCR_WORKSPACE
        self.tool_domain = "general"
        self.auto_approve = True
        self.session_dir = None
        self.working_memory = working_memory
        self.task_profile = None
        self.document_mode = False
        self.research_mode = False

console = Console()

API_BASE = "http://100.110.236.127:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "qwen3.5-9b"

STRATEGIES = ["native_all", "hybrid", "text_only"]

# Per-scenario registry flags (merged over the defaults in the eval loop).
# Only keys that differ from the defaults need to be listed.
_DEFAULT_REGISTRY_FLAGS = dict(
    enable_memory=False,
    enable_web=True,
    enable_document=True,
    enable_code_tools=True,
    enable_mutation=True,
    enable_bash=True,
    enable_parallel=False,
    enable_meta=True,
)

SCENARIOS = [
    # ── Observation / search ──────────────────────────────────────────
    {
        "name": "search web",
        "mode": "research",
        "goal": (
            "Search the web for the latest news about Python 3.14 and give me a "
            "2 sentence summary. Make sure you use the web_search tool to fetch real data."
        ),
        "expected_actions": ["web_search"],
    },
    # ── Document tools ────────────────────────────────────────────────
    {
        "name": "get document overview",
        "mode": "document",
        "goal": (
            f"Load the document at '{PDF_PATH}' and return its structural overview "
            "using get_document_overview."
        ),
        "expected_actions": ["get_document_overview"],
    },
    {
        "name": "read document section",
        "mode": "document",
        "goal": (
            f"Read page 1 of the document '{PDF_PATH}'. "
            "Call read_document_section with filepath='{PDF_PATH}' and page_number=1. "
            "Do NOT call get_document_overview first."
        ).format(PDF_PATH=PDF_PATH),
        "expected_actions": ["read_document_section"],
    },
    {
        "name": "search document",
        "mode": "document",
        "goal": (
            f"Call search_document directly with query='software engineering' and "
            f"filepath='{PDF_PATH}'. Do NOT call get_document_overview first — "
            "the tool will load the document automatically if needed."
        ),
        "expected_actions": ["search_document"],
    },
    # ── Memory tools ──────────────────────────────────────────────────
    {
        "name": "update memory",
        "mode": "general",
        "goal": (
            "Record the fact that 'Python 3.14 introduces new typing improvements' "
            "in your working memory. Use update_memory with a patch that adds this "
            "to knowledge.key_facts."
        ),
        "expected_actions": ["update_memory"],
        "enable_memory": True,
    },
    {
        "name": "get memory",
        "mode": "general",
        "goal": (
            "Read your full working memory using the get_memory tool. "
            "Do not pass any section argument — retrieve the entire snapshot."
        ),
        "expected_actions": ["get_memory"],
        "enable_memory": True,
    },
    # ── Coding / mutation tools ───────────────────────────────────────
    {
        "name": "write code",
        "mode": "coding",
        "goal": (
            "Write a python script called 'factorial.py' that contains a function "
            "to calculate the factorial of a number, and prints the factorial of 5. "
            "You MUST actually create the file on disk."
        ),
        "expected_actions": ["write_file"],
    },
    {
        "name": "modify code",
        "mode": "coding",
        "goal": (
            "The file 'factorial.py' already exists and contains exactly this line: "
            "print('Hello World')\n"
            "Use search_and_replace directly to change 'Hello' to 'Hola' in "
            "'factorial.py'. Use old_text='Hello' and new_text='Hola'. "
            "Do NOT read the file first."
        ),
        "expected_actions": ["search_and_replace"],
    },
    # ── Local tools ───────────────────────────────────────────────────
    {
        "name": "list dir",
        "mode": "general",
        "goal": "List the contents of the 'BatchAgent' directory.",
        "expected_actions": ["list_directory"],
    },
    {
        "name": "find file",
        "mode": "general",
        "goal": (
            "Find the file named 'test_universal_tool_handler.py' "
            "in the 'BatchAgent' directory."
        ),
        "expected_actions": ["find_file"],
    },
    {
        "name": "read file chunk",
        "mode": "coding",
        "goal": "Read lines 1 to 20 of 'BatchAgent/tools/tool_router.py'.",
        "expected_actions": ["read_file_chunk"],
    },
    {
        "name": "search code",
        "mode": "coding",
        "goal": (
            "Search the codebase in the 'BatchAgent' directory for the string "
            "'class UniversalToolHandler'."
        ),
        "expected_actions": ["search_code"],
    },
    {
        "name": "run bash",
        "mode": "general",
        "goal": "Run a bash command to echo 'hello from bash'.",
        "expected_actions": ["run_bash_command"],
    },
]

def map_action_to_name(action):
    """Maps the AgentAction object instances back to their tool string primitive labels."""
    if isinstance(action, ActionToolCall):
        return action.name
    elif isinstance(action, ActionWriteFile):
        return "write_file"
    elif isinstance(action, ActionReplaceText):
        return "search_and_replace"
    elif isinstance(action, ActionApplyDiff):
        return "apply_diff"
    return "unknown"

def _check_executed(scenario_name: str, execution_results: str, has_mutation: bool) -> bool:
    """Determine whether the scenario's tool actually produced a valid result."""
    r = execution_results
    checks = {
        "search web":           lambda: "Result for web_search" in r and len(r) > 100,
        "get document overview": lambda: "Result for get_document_overview" in r or "Document Overview:" in r,
        "read document section": lambda: "Result for read_document_section" in r,
        "search document":       lambda: "Result for search_document" in r,
        "update memory":         lambda: "Result for update_memory" in r,
        "get memory":            lambda: "Result for get_memory" in r,
        "write code":            lambda: has_mutation,
        "modify code":           lambda: has_mutation,
        "list dir":              lambda: "Result for list_directory" in r,
        "find file":             lambda: "Result for find_file" in r,
        "read file chunk":       lambda: "Result for read_file_chunk" in r,
        "search code":           lambda: "Result for search_code" in r,
        "run bash":              lambda: "Result for run_bash_command" in r,
    }
    fn = checks.get(scenario_name)
    return fn() if fn else False


async def evaluate():
    from BatchAgent.working_memory import WorkingMemory

    # Make sure target file exists for search_and_replace
    with open("factorial.py", "w") as f:
        f.write("def factorial(n):\n    return 1\nprint('Hello World')\n")

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
    console.print(f"\n[bold green]Testing Strategies on vLLM ({API_BASE} with {MODEL_NAME})[/bold green]\n")

    results = []

    # Allowlist covers the working directory so any file written/modified there is permitted
    allowlist = [str(Path(".").resolve())]

    import builtins
    original_input = builtins.input
    builtins.input = lambda prompt="": "y"

    try:
        for strategy in STRATEGIES:
            for scenario in SCENARIOS:
                console.print(f"\n[bold cyan]--- Strategy: {strategy} | Scenario: {scenario['name']} ---[/bold cyan]")

                # Reset factorial.py before modify code so old_text='Hello' is always present.
                if scenario["name"] == "modify code":
                    with open("factorial.py", "w") as f:
                        f.write("def factorial(n):\n    return 1\nprint('Hello World')\n")

                # 1) Build per-scenario registry flags (start from defaults, apply overrides)
                reg_flags = {**_DEFAULT_REGISTRY_FLAGS}
                if scenario.get("enable_memory"):
                    reg_flags["enable_memory"] = True

                configure_global_tool_registry(
                    strategy=strategy,
                    profile=scenario["mode"],
                    domain="general",
                    **reg_flags,
                )
                active_tools = get_base_tools(strategy=strategy, domain=scenario["mode"])

                # 2) Build a per-scenario WorkingMemory when needed
                working_memory = (
                    WorkingMemory(session_dir=Path(tempfile.mkdtemp()))
                    if scenario.get("enable_memory")
                    else None
                )

                # 3) Assemble the prompt
                sys_prompt = PromptRegistry.get_system_prompt(
                    strategy=strategy,
                    base_tools_list=active_tools,
                    domain=scenario["mode"],
                )

                user_prompt = f"Task: {scenario['goal']}\nExecute the task using exactly the tools provided."

                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # 4) Compile tools into OpenAI API format
                compiled_tools = compile_tools_for_provider(active_tools, "openai", strategy)

                # 5) LLM call
                full_content, parsed_actions = await complete_with_continuation_async(
                    client=client,
                    model=MODEL_NAME,
                    messages=messages,
                    tool_strategy=strategy,
                    tools=compiled_tools,
                    verbose=True,
                    allowlist=allowlist,
                    dynamic_tools_registry={},
                    backend="vllm",
                )

                # 6) Map parsed actions back to tool name strings for display
                actual_tool_names = [map_action_to_name(a) for a in parsed_actions]

                # 7) Execute via UniversalToolHandler
                config = MockConfig(strategy, working_memory=working_memory)
                handler = UniversalToolHandler(
                    config=config,
                    turn_dir=Path(tempfile.mkdtemp()),
                    allowlist=[str(Path(".").resolve())],
                    dynamic_tools_registry={},
                )

                has_mutation, execution_results = handler.execute(parsed_actions, full_content)

                console.print("\n[bold yellow]--- Execution Verification ---[/bold yellow]")
                console.print(execution_results)

                executed = _check_executed(scenario["name"], execution_results, has_mutation)
                success = len(parsed_actions) > 0 and executed

                console.print(f"\n-> Parsed Actions: {actual_tool_names}")
                console.print(f"-> Execution Success: {'[green]Yes[/green]' if success else '[red]No[/red]'}")

                results.append({
                    "strategy": strategy,
                    "scenario": scenario["name"],
                    "success": success,
                    "actions": actual_tool_names,
                })

    finally:
        builtins.input = original_input

    # Summary Generation
    console.print("\n[bold magenta]=== Strategy Evaluation Summary ===[/bold magenta]")
    table = Table(title="Tool Parsing Strategy Success Rates")
    table.add_column("Strategy", justify="left", style="cyan")
    table.add_column("Scenario", justify="left")
    table.add_column("Success", justify="center")
    table.add_column("Parsed Actions", justify="left", style="dim")

    strategy_stats = {s: {"total": 0, "success": 0} for s in STRATEGIES}

    for r in results:
        strat = r["strategy"]
        strategy_stats[strat]["total"] += 1
        if r["success"]:
            strategy_stats[strat]["success"] += 1
        success_icon = "✅" if r["success"] else "❌"
        table.add_row(strat, r["scenario"], success_icon, str(r["actions"]))

    console.print(table)

    console.print("\n[bold magenta]Aggregate Success Rates:[/bold magenta]")
    for strat, stat in strategy_stats.items():
        rate = (stat["success"] / stat["total"]) * 100 if stat["total"] > 0 else 0
        console.print(f"- {strat}: {rate:.1f}% ({stat['success']}/{stat['total']})")


if __name__ == "__main__":
    try:
        asyncio.run(evaluate())
    except KeyboardInterrupt:
        console.print("[yellow]\nEvaluation interrupted.[/yellow]")

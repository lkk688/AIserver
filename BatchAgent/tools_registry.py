from typing import List, Dict, Any
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BatchAgent.domain_tools import DOMAIN_REGISTRY

# ==========================================
# all tools are defined here in a provider-agnostic way, then compiled to specific formats in get_active_tools() based on strategy and provider.
# ==========================================

# 1. observation tools, short parameters, used for information gathering and verification. Safe to expose in all strategies.
OBSERVATION_TOOLS = [
    {
        "name": "search_code",
        "description": "Search for a string or regex pattern in the codebase.",
        "properties": {"query": {"type": "string", "description": "The text pattern to search for"}},
        "required": ["query"]
    },
    {
        "name": "read_file_chunk",
        "description": "Read the contents of a file along with line numbers.",
        "properties": {
            "filepath": {"type": "string", "description": "Relative path to the file"},
            "start_line": {"type": "integer", "description": "Start line number"},
            "end_line": {"type": "integer", "description": "End line number"}
        },
        "required": ["filepath"]
    },
    {
        "name": "run_bash_command",
        "description": "Execute a terminal command. Use for verification, testing, or reading environment states.",
        "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
        "required": ["command"]
    },
    {
        "name": "web_search",
        "description": (
            "Search the internet for real-time information. Use the 'category' field to route "
            "the query to authoritative domain-specific sources for best results."
        ),
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "category": {
                "type": "string",
                "enum": [
                    "general",       # Default: no site filter
                    "news",          # Reuters, AP, BBC, Bloomberg, Guardian
                    "code",          # GitHub, StackOverflow, PyPI, docs.python.org
                    "academic",      # arXiv, Google Scholar, SemanticScholar, PubMed
                    "medical",       # PubMed, NIH, MayoClinic, WebMD
                    "software_eng",  # SO, GitHub, dev.to, PyPI
                    "math",          # AoPS, Math.SE, MathWorld, KhanAcademy, Brilliant
                    "science",       # Nature, ScienceDirect, Phys.org, WolframAlpha
                    "language",      # Wiktionary, LanguageGuide, Italki
                    "business",      # SEC, Yahoo Finance, Bloomberg, Investopedia
                    "assistant",     # SuperUser, AskUbuntu, ServerFault, Apple.SE
                    "sales_support", # Zendesk, HubSpot, Salesforce, Freshdesk
                    "research",      # arXiv, SemanticScholar, Scholar, JSTOR
                    "finance",       # Alias for business
                    "health",        # Alias for medical
                    "programming",   # Alias for code
                ],
                "description": (
                    "Domain category to focus the search. Pick the most specific match: "
                    "use 'math' for math problems (AoPS, MathWorld), 'academic' for papers, "
                    "'medical' for health queries, 'news' for current events, 'code'/'software_eng' "
                    "for programming. Default to 'general' when unsure."
                )
            }
        },
        "required": ["query"]
    },
    {
        "name": "read_url",
        "description": "Fetch and read the full text content of a specific webpage URL.",
        "properties": {"url": {"type": "string"}},
        "required": ["url"]
    },
    {
        "name": "finish_task",
        "description": "Call this tool ONLY when you have fully completed the user's goal and verified the results.",
        "properties": {"summary": {"type": "string", "description": "A brief summary of what was accomplished."}},
        "required": ["summary"]
    },
    {
        "name": "execute_parallel_branches",
        "description": "Execute multiple independent exploration, research, or coding paths simultaneously. Each branch runs in an isolated workspace and reports back a summary. Use this for brainstorming, testing different algorithms, or parallel web searching.",
        "properties": {
            "branches": {
                "type": "array",
                "description": "List of parallel branches to execute.",
                "items": {
                    "type": "object",
                    "properties": {
                        "branch_id": {"type": "string", "description": "Unique alphanumeric ID for this branch, e.g., 'branch_a_quick_sort'"},
                        "instruction": {"type": "string", "description": "Highly specific instruction for what this branch should do (e.g., 'Write and test a QuickSort algorithm', or 'Search for Python 3.13 typing features')."}
                    },
                    "required": ["branch_id", "instruction"]
                }
            }
        },
        "required": ["branches"]
    },
    {
        "name": "inspect_branch_details",
        "description": "Retrieve the full details, generated code, and test results of a specific branch executed previously.",
        "properties": {
            "branch_id": {"type": "string", "description": "The ID of the branch you want to inspect."}
        },
        "required": ["branch_id"]
    }
]

# 2. file mutation tools, longer parameters, potentially destructive. Only for native_all strategy, never exposed in hybrid or text_only.
MUTATION_TOOLS = [
    {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing file. WARNING: For large existing files, use search_and_replace instead.",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "content": {"type": "string", "description": "The complete file content to write."}
        },
        "required": ["path", "content"]
    },
    {
        "name": "search_and_replace",
        "description": "Precisely replace a block of text in an existing file. Much safer for large files.",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "old_text": {"type": "string", "description": "The exact substring to replace, including exact whitespace and indentation."},
            "new_text": {"type": "string", "description": "The new replacement text."}
        },
        "required": ["path", "old_text", "new_text"]
    }
]

# 动态提取 domain_tools.py 中的领域名称，作为 Enum 给大模型参考
DOMAIN_NAMES = list(DOMAIN_REGISTRY.keys())

META_TOOLS = [
    {
        "name": "load_domain_tools",
        "description": "If your task requires specialized knowledge (e.g., medical, academic, finance), use this tool to load the specific domain plugin into your context. This unlocks advanced tools.",
        "properties": {
            "domain": {
                "type": "string",
                "enum": DOMAIN_NAMES,
                "description": "The specific domain plugin to load."
            }
        },
        "required": ["domain"]
    },
    {
        "name": "register_custom_tool",
        "description": "Create and register a custom tool mid-flight! FIRST, write a Python script that takes a JSON string from sys.argv[1], processes it, and prints the result. THEN, use this tool to register it.",
        "properties": {
            "tool_name": {"type": "string", "description": "Name of the new tool (lowercase, underscores)."},
            "description": {"type": "string", "description": "What the tool does. Be descriptive so you know when to use it later."},
            "schema_properties": {"type": "object", "description": "The JSON schema 'properties' object for the tool's arguments."},
            "required_args": {"type": "array", "items": {"type": "string"}, "description": "List of required argument names."},
            "script_path": {"type": "string", "description": "The relative path to the Python script you wrote (e.g., 'custom_tools/math_solver.py')."}
        },
        "required": ["tool_name", "description", "schema_properties", "script_path"]
    }
]

def get_base_tools(strategy: str, enable_parallel: bool = False, domain: str = "general") -> List[Dict[str, Any]]:
    """
    Step 1: Get the raw tool schemas based on the execution strategy and features.
    This list is used to dynamically generate the System Prompt.
    """
    # 基础观察工具 (所有模式都可用)
    active_tools = OBSERVATION_TOOLS.copy()
    
    # 动态注入并行思考工具
    if enable_parallel:
        active_tools.append({
            "name": "brainstorm_solutions",
            "description": "Trigger parallel LLM thinking. Use this when facing a complex problem to brainstorm multiple distinct approaches simultaneously before writing code.",
            "properties": {
                "problem_statement": {"type": "string"},
                "n_variations": {"type": "integer", "description": "Number of parallel approaches to generate (max 4)."}
            },
            "required": ["problem_statement", "n_variations"]
        })

    # [NEW] 动态注入领域专属工具
    # 动态注入领域专属工具
    if domain != "auto" and domain != "general" and domain in DOMAIN_REGISTRY:
        active_tools.extend(DOMAIN_REGISTRY[domain])
        
    # Mutation tools (write_file, search_and_replace):
    # - native_all  : exposed as JSON schema tools to the API
    # - text_only   : NOT exposed to API (compile_tools_for_provider returns [])
    #                 but IS included here so prompt_registry can document them as XML tags
    # - hybrid      : same as text_only — JSON for observation, XML for mutations
    if strategy in ("native_all", "text_only", "hybrid"):
        active_tools.extend(MUTATION_TOOLS)
        
    return active_tools

def compile_tools_for_provider(base_tools: List[Dict[str, Any]], provider: str, strategy: str) -> List[Dict[str, Any]]:
    """
    Step 2: Compile raw schemas into the specific format expected by the LLM Provider API.
    """
    # [FIX] 如果是纯文本模式，向 API 注册空工具列表，强制关闭底层 Function Calling
    if strategy == "text_only":
        return []
        
    compiled = []
    for tool in base_tools:
        if provider == "anthropic":
            compiled.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool.get("properties", {}),
                    "required": tool.get("required", [])
                }
            })
        else: # OpenAI / vLLM Format
            compiled.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool.get("properties", {}),
                        "required": tool.get("required", [])
                    }
                }
            })
    return compiled

# [DEPRECATED] 
def get_active_tools(strategy: str, provider: str) -> List[Dict[str, Any]]:
    """
    Compile the active tool list based on the chosen strategy and provider format.
    """
    active_base_tools = []
    
    if strategy == "native_all":
        # Anthropic/OpenAI: all JSON Tool Call
        active_base_tools = OBSERVATION_TOOLS + MUTATION_TOOLS
    elif strategy == "hybrid":
        # hybrid mode: only short parameter observation tools are registered as JSON tools, mutation tools can still be used but must be called via text commands that the agent parses and executes internally.
        active_base_tools = OBSERVATION_TOOLS
    elif strategy == "text_only":
        # pure text mode: no tools are registered in the provider format, but the agent can still use all tools via text commands that it parses and executes internally. This allows maximum flexibility but requires the agent to have strong parsing and execution capabilities.
        return []
    
    # compile the active tools into the provider-specific format
    compiled = []
    for tool in active_base_tools:
        if provider == "anthropic":
            compiled.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["properties"],
                    "required": tool.get("required", [])
                }
            })
        else: # OpenAI / vLLM Format
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
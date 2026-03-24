from __future__ import annotations

"""
tools_registry.py

Compatibility layer for:
- prompt construction
- provider-specific tool schema compilation
- backward-compatible external API

This module does NOT own the runtime selection logic.
That belongs to tool_registry_runtime.py.

Responsibilities
----------------
1. Read currently active tools from GLOBAL_TOOL_REGISTRY
2. Expose lightweight schema dicts for prompt generation
3. Compile active tools to OpenAI / Anthropic provider formats
4. Preserve the old external calling style:
   - get_base_tools(...)
   - compile_tools_for_provider(...)
   - get_active_tools(...)
"""

from typing import Any, Dict, List

from BatchAgent.tools.domain_tools import DOMAIN_REGISTRY

# 1. observation tools, short parameters, used for information gathering and verification. Safe to expose in all strategies.
OBSERVATION_TOOLS = [
    {
        "name": "search_code",
        "description": "Search for a string or regex pattern in the codebase.",
        "properties": {"query": {"type": "string", "description": "The text pattern to search for"}},
        "required": ["query"]
    },
    {
        "name": "find_file",
        "description": "Find files by name pattern in the workspace. Returns relative paths.",
        "properties": {
            "filename_pattern": {"type": "string", "description": "Glob pattern (e.g. *.py)"},
            "root_dir": {"type": "string", "description": "Directory to search in (default is current workspace)"}
        },
        "required": ["filename_pattern"]
    },
    {
        "name": "list_directory",
        "description": "List contents of a directory (files and subdirectories) to understand the project structure.",
        "properties": {
            "dir_path": {"type": "string", "description": "Directory path to list (default is current folder)"}
        },
        "required": []
    },
    {
        "name": "read_file_chunk",
        "description": "Read a source code or text file by line numbers. Use this for code files (.py, .js, .ts, etc.). For PDF/document content use read_document_section instead.",
        "properties": {
            "filepath": {"type": "string", "description": "Relative path to the file"},
            "start_line": {"type": "integer", "description": "Start line number (1-based)"},
            "end_line": {"type": "integer", "description": "End line number (inclusive)"}
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
        "description": "Fetch and read a webpage, file, or GitHub URL. For missing URLs it returns nearby reachable links. For GitHub folders it returns a paged folder listing. Use offset to continue and name_contains to narrow by filename or folder name.",
        "properties": {
            "url": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
            "name_contains": {"type": "string"},
        },
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
        "description": (
            "Execute multiple independent instruction branches concurrently. "
            "USE THIS when the document is LARGE (🚨 flag in get_document_overview) or when you need to "
            "brainstorm multiple distinct approaches simultaneously. "
            "Each branch gets one LLM call + tool access and runs in parallel. "
            "If the combined results exceed the context window, they are stored in memory automatically — "
            "call get_memory('knowledge') to retrieve branch summaries, "
            "or call read_document_section / other tools directly for full content."
        ),
        "properties": {
            "branches": {
                "type": "array",
                "description": "List of parallel branches to execute.",
                "items": {
                    "type": "object",
                    "properties": {
                        "branch_id": {"type": "string", "description": "Unique snake_case ID, e.g. 'branch_intro' or 'branch_quicksort'"},
                        "instruction": {"type": "string", "description": "Specific instruction for this branch (e.g. 'Read sections sec_4 and sec_5', or 'Search for Python 3.13 typing features')."}
                    },
                    "required": ["branch_id", "instruction"]
                }
            }
        },
        "required": ["branches"]
    },
    {
        "name": "get_document_overview",
        "description": (
            "Load a PDF/document and return its structural overview (table of contents with section IDs). "
            "If the document is already loaded, just returns the overview. "
            "Always provide the filepath when the user mentions a PDF path. "
            "After getting the overview, use read_document_section with a section_id to read any section in full."
        ),
        "properties": {
            "filepath": {
                "type": "string",
                "description": (
                    "Path to the PDF or document file to load (relative to the working directory). "
                    "Required on the first call. Omit if the document is already loaded."
                )
            }
        },
        "required": []
    },
    {
        "name": "read_document_section",
        "description": (
            "Read the full content of a specific section in the currently loaded PDF/document. "
            "Unlike read_file_chunk (which is for code files with line numbers), this tool works with "
            "document section IDs and page numbers. "
            "Workflow: call get_document_overview first to get section IDs, then call this with section_id. "
            "Alternatively provide a page number to retrieve all sections on that page."
        ),
        "properties": {
            "section_id": {
                "type": "string",
                "description": (
                    "The section ID returned by get_document_overview (e.g. 'sec_2_1'). "
                    "Takes priority over page if both are provided."
                )
            },
            "page": {
                "type": "integer",
                "description": "Page number (1-based). Returns all sections found on that page."
            }
        },
        "required": []
    },
    {
        "name": "search_document",
        "description": (
            "Search within the currently loaded PDF/document using hybrid retrieval. "
            "Returns matched snippets, page numbers, and section IDs/titles."
        ),
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "description": "Maximum number of results to return"}
        },
        "required": ["query"]
    }
]

# 2. memory tools — always safe, always available regardless of strategy
MEMORY_TOOLS = [
    {
        "name": "get_memory",
        "description": (
            "Read your working memory to recall task state, completed steps, key facts, "
            "section summaries, pending items, and produced artifacts. "
            "Call this when you need to remember what you have already done or decide what to do next. "
            "Available sections: control, task, knowledge, progress, actions, artifacts. "
            "Omit 'section' to retrieve the full memory."
        ),
        "properties": {
            "section": {
                "type": "string",
                "enum": ["control", "task", "knowledge", "progress", "actions", "artifacts"],
                "description": (
                    "Optional. Which section to retrieve. "
                    "Omit (or pass null) for the full memory snapshot."
                ),
            }
        },
        "required": [],
    },
    {
        "name": "update_memory",
        "description": (
            "Update your working memory with new knowledge or progress (PATCH-style: only "
            "provide fields you want to change; existing values are preserved). "
            "Lists are APPENDED to, not replaced. "
            "Use this at key checkpoints to record: "
            "what you just learned (knowledge.key_facts, knowledge.summaries), "
            "which step was completed (progress.completed_steps), "
            "what comes next (progress.pending_steps), "
            "and what you are doing right now (control.current_step, control.stage). "
            "Do NOT call this for every tool use — it is for meaningful state checkpoints only."
        ),
        "properties": {
            "patch": {
                "type": "object",
                "description": (
                    "JSON object whose top-level keys are section names "
                    "(control, task, knowledge, progress, actions, artifacts). "
                    'Example: {"progress": {"completed_steps": ["Read intro section"]}, '
                    '"knowledge": {"key_facts": ["Model uses 9B parameters"]}, '
                    '"control": {"current_step": "reading section 2", "stage": "gathering"}}'
                ),
            }
        },
        "required": ["patch"],
    },
]

# 3. file mutation tools, longer parameters, potentially destructive. Only for native_all strategy, never exposed in hybrid or text_only.
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



def _strip_runtime_fields(tool_obj) -> Dict[str, Any]:
    """
    Convert RuntimeToolSpec -> plain schema dict used by prompts/provider compile.
    """
    return {
        "name": tool_obj.name,
        "description": tool_obj.description,
        "properties": tool_obj.properties,
        "required": list(tool_obj.required),
        "category": tool_obj.category,
        "handler_name": tool_obj.handler_name,
    }


def get_base_tools(strategy: str, enable_parallel: bool = False, domain: str = "general") -> List[Dict[str, Any]]:
    """
    Return raw active tool schemas for prompt construction.

    Notes
    -----
    - strategy/domain are kept for backward compatibility in the signature
      but runtime activation should already have been configured upstream
      through GLOBAL_TOOL_REGISTRY.configure(...).
    - This function simply reads the currently active tools.
    """
    from BatchAgent.tools.tool_registry_runtime import GLOBAL_TOOL_REGISTRY
    return [_strip_runtime_fields(t) for t in GLOBAL_TOOL_REGISTRY.active_tools()]


def compile_tools_for_provider(base_tools: List[Dict[str, Any]], provider: str, strategy: str) -> List[Dict[str, Any]]:
    """
    Compile plain schema dicts into provider-specific tool format.

    Rules
    -----
    - text_only: returns []
    - hybrid: hides mutation tools so the model must use textual XML fallback
    - native_all: includes mutation tools too if caller passed them in
    """
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
                    "required": tool.get("required", []),
                },
            })
        else:
            compiled.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool.get("properties", {}),
                        "required": tool.get("required", []),
                    },
                },
            })

    return compiled


def get_active_tools(strategy: str, provider: str) -> List[Dict[str, Any]]:
    """
    Backward-compatible helper.

    Reads the currently active runtime tools, then compiles them for the provider.
    """
    base_tools = get_base_tools(strategy=strategy, enable_parallel=False, domain="general")
    return compile_tools_for_provider(base_tools, provider, strategy)

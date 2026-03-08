from typing import List, Dict, Any

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
        "description": "Search the internet for real-time documentation or solutions.",
        "properties": {"query": {"type": "string"}, "category": {"type": "string", "enum": ["general", "code"]}},
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
from typing import List, Dict, Any
import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BatchAgent.tools.domain_tools import DOMAIN_REGISTRY

# Import canonical tool schema definitions from tools/tools_registry.py.
# This file keeps the backward-compatible get_base_tools / compile_tools_for_provider
# API while delegating schema ownership to tools/tools_registry.py.
from BatchAgent.tools.tools_registry import (
    OBSERVATION_TOOLS,
    MEMORY_TOOLS,
    MUTATION_TOOLS,
)

# ==========================================
# Tool schemas are now owned by BatchAgent/tools/tools_registry.py.
# OBSERVATION_TOOLS, MEMORY_TOOLS, MUTATION_TOOLS are imported above.
# META_TOOLS (load_domain_tools, register_custom_tool) are intentionally excluded
# from the default tool list — domain routing is handled by the web_search
# category parameter via online_search_toolkit.
# ==========================================

DOMAIN_NAMES = list(DOMAIN_REGISTRY.keys())

def get_base_tools(strategy: str, enable_parallel: bool = False, domain: str = "general") -> List[Dict[str, Any]]:
    """
    Step 1: Get the raw tool schemas based on the execution strategy and features.
    This list is used to dynamically generate the System Prompt.
    """
    # 基础观察工具 (所有模式都可用)
    active_tools = OBSERVATION_TOOLS.copy()

    # Memory tools are always available (safe, stateful, internal)
    active_tools.extend(MEMORY_TOOLS)
    
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
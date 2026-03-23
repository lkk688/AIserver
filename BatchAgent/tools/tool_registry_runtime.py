from __future__ import annotations

"""
tool_registry_runtime.py

Runtime tool registry with:
1. Unified registration from static tool specs
2. Runtime activation by:
   - profile
   - strategy
   - domain
   - capability flags
3. Fast lookup for ToolRouter / orchestration layer

Design goals
------------
- Keep the static tool definitions in tools_registry.py
- Keep runtime selection logic here
- Allow the agent to expose only the right subset of tools
- Avoid giving document tools to coding-only tasks
- Avoid giving code tools to pure document QA tasks
- Avoid exposing mutation tools through ToolRouter
- Keep domain tool support pluggable for later

Expected flow
-------------
At startup / session init:
    runtime_registry.configure(...)

Later:
    ToolRouter.dispatch(name, args)
        -> GLOBAL_TOOL_REGISTRY.get(name)

This file does NOT execute tools.
It only defines which tools are currently active.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

from BatchAgent.tools.tools_registry import (
    OBSERVATION_TOOLS,
    MEMORY_TOOLS,
    MUTATION_TOOLS,
    META_TOOLS,
)


# =============================================================================
# 1. Normalized runtime model
# =============================================================================

@dataclass
class RuntimeToolSpec:
    """
    Normalized runtime representation of one tool.

    Fields
    ------
    name:
        Unique tool name.

    description:
        Natural-language description shown to the model / provider layer.

    properties:
        JSON-schema properties for arguments.

    required:
        Required argument names.

    category:
        One of:
        - observation
        - mutation
        - memory
        - meta
        - domain

    handler_name:
        Name used by ToolRouter to dispatch to a handler method.

    profiles:
        Task profiles under which this tool is relevant.
        Examples: coding, document, research, general

    strategies:
        Allowed tool strategies:
        - native_all
        - hybrid
        - text_only

    domains:
        Logical domains where this tool is relevant.
        For now we keep "general" as the default and do not activate custom domain tools.

    requires:
        Runtime capabilities required for activation.
        Examples:
        - working_memory
        - document_manager
        - web_access
        - local_shell
        - mutation
    """
    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    category: str = "observation"
    handler_name: Optional[str] = None
    profiles: Set[str] = field(default_factory=set)
    strategies: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    requires: Set[str] = field(default_factory=set)
    enabled_by_default: bool = True

    def to_openai_function(self) -> Dict[str, Any]:
        """
        Compile this tool to the OpenAI / vLLM function-calling schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """
        Compile this tool to Anthropic's tool schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.properties,
                "required": self.required,
            },
        }


# =============================================================================
# 2. Static normalization rules
# =============================================================================

def _tool_category_for_name(name: str, source_group: str) -> str:
    """
    Infer category from the source list and tool name.
    """
    if source_group == "memory":
        return "memory"
    if source_group == "mutation":
        return "mutation"
    if source_group == "meta":
        return "meta"
    return "observation"


def _default_profiles_for_tool(name: str, category: str) -> Set[str]:
    """
    Assign sensible default task profiles.

    Profiles
    --------
    - coding
    - document
    - research
    - general
    """
    if category == "memory":
        return {"coding", "document", "research", "general"}

    if category == "mutation":
        return {"coding", "general"}

    if name in {"search_code", "find_file", "read_file_chunk", "list_directory", "run_bash_command"}:
        return {"coding", "general"}

    if name in {"get_document_overview", "read_document_section", "search_document"}:
        return {"document", "research", "general"}

    if name in {"web_search", "read_url"}:
        return {"research", "document", "general"}

    if name in {"finish_task"}:
        return {"coding", "document", "research", "general"}

    if name in {"load_domain_tools", "register_custom_tool"}:
        return {"coding", "research", "general"}

    if name == "execute_parallel_branches":
        return {"coding", "research", "general", "document"}

    return {"general"}


def _default_strategies_for_tool(name: str, category: str) -> Set[str]:
    """
    Assign allowed tool strategies.

    Notes
    -----
    - observation / memory / meta tools can exist in all strategies
    - mutation tools are only provider-native in native_all
      but may still be documented in hybrid/text_only externally
    """
    if category == "mutation":
        return {"native_all", "hybrid", "text_only"}

    return {"native_all", "hybrid", "text_only"}


def _default_domains_for_tool(name: str, category: str) -> Set[str]:
    """
    For now, most tools are general-purpose.
    Domain-specific expansion can be added later.
    """
    return {"general"}


def _default_requires_for_tool(name: str, category: str) -> Set[str]:
    """
    Capability requirements for runtime activation.
    """
    if category == "memory":
        return {"working_memory"}

    if category == "mutation":
        return {"mutation"}

    if name in {"get_document_overview", "read_document_section", "search_document"}:
        return {"document_manager"}

    if name in {"web_search", "read_url"}:
        return {"web_access"}

    if name == "run_bash_command":
        return {"local_shell"}

    return set()


def _default_handler_name_for_tool(name: str, category: str) -> Optional[str]:
    """
    In your current architecture, handler_name normally matches tool name.
    Mutation tools are handled by MutationManager, so handler_name stays None.
    """
    if category == "mutation":
        return None
    return name


def _normalize_tool_dict(tool: Dict[str, Any], source_group: str) -> RuntimeToolSpec:
    """
    Convert one raw tool dict from tools_registry.py into RuntimeToolSpec.
    """
    name = tool["name"]
    category = tool.get("category") or _tool_category_for_name(name, source_group)

    return RuntimeToolSpec(
        name=name,
        description=tool.get("description", ""),
        properties=tool.get("properties", {}),
        required=list(tool.get("required", [])),
        category=category,
        handler_name=tool.get("handler_name", _default_handler_name_for_tool(name, category)),
        profiles=set(tool.get("profiles", _default_profiles_for_tool(name, category))),
        strategies=set(tool.get("strategies", _default_strategies_for_tool(name, category))),
        domains=set(tool.get("domains", _default_domains_for_tool(name, category))),
        requires=set(tool.get("requires", _default_requires_for_tool(name, category))),
        enabled_by_default=bool(tool.get("enabled_by_default", True)),
    )


def _load_all_static_tools() -> List[RuntimeToolSpec]:
    """
    Load all static tools from tools_registry.py and normalize them.

    Important
    ---------
    domain_tools.py is intentionally NOT loaded yet.
    """
    specs: List[RuntimeToolSpec] = []

    for item in OBSERVATION_TOOLS:
        specs.append(_normalize_tool_dict(item, source_group="observation"))

    for item in MEMORY_TOOLS:
        specs.append(_normalize_tool_dict(item, source_group="memory"))

    for item in MUTATION_TOOLS:
        specs.append(_normalize_tool_dict(item, source_group="mutation"))

    for item in META_TOOLS:
        specs.append(_normalize_tool_dict(item, source_group="meta"))

    return specs


# =============================================================================
# 3. Activation config
# =============================================================================

@dataclass
class ToolActivationConfig:
    """
    Runtime activation configuration.

    strategy:
        native_all | hybrid | text_only

    profile:
        coding | document | research | general | mixed

    domain:
        currently general / auto / custom string
        domain-specific activation is reserved for later

    Capability flags
    ----------------
    enable_memory:
        Whether working_memory tools should be active.

    enable_web:
        Whether web tools should be active.

    enable_document:
        Whether document tools should be active.

    enable_code_tools:
        Whether local code/file observation tools should be active.

    enable_mutation:
        Whether mutation tools should be active in the runtime toolset.

    enable_bash:
        Whether run_bash_command should be active.

    enable_parallel:
        Whether parallel/branch meta tools should be active.

    enable_meta:
        Whether meta tools like register_custom_tool should be active.
    """
    strategy: str = "hybrid"
    profile: str = "general"
    domain: str = "general"

    enable_memory: bool = True
    enable_web: bool = True
    enable_document: bool = False
    enable_code_tools: bool = True
    enable_mutation: bool = True
    enable_bash: bool = False
    enable_parallel: bool = False
    enable_meta: bool = True

    def normalized_profile_set(self) -> Set[str]:
        p = (self.profile or "general").strip().lower()
        if p == "mixed":
            return {"coding", "document", "research", "general"}
        return {p, "general"}

    def normalized_domain_set(self) -> Set[str]:
        d = (self.domain or "general").strip().lower()
        if d in {"auto", ""}:
            return {"general"}
        return {d, "general"}

    def capability_set(self) -> Set[str]:
        caps: Set[str] = set()

        if self.enable_memory:
            caps.add("working_memory")
        if self.enable_document:
            caps.add("document_manager")
        if self.enable_web:
            caps.add("web_access")
        if self.enable_mutation:
            caps.add("mutation")
        if self.enable_bash:
            caps.add("local_shell")

        return caps


# =============================================================================
# 4. Runtime registry
# =============================================================================

class ToolRegistryRuntime:
    """
    Runtime registry of currently active tools.

    This object holds:
    - all known tools
    - currently active subset
    - fast name lookup
    - provider compilation helpers
    """

    def __init__(self):
        self._all_tools: List[RuntimeToolSpec] = _load_all_static_tools()
        self._active_tools: Dict[str, RuntimeToolSpec] = {}
        self._last_config: Optional[ToolActivationConfig] = None

    # ------------------------------------------------------------------
    # Configuration / activation
    # ------------------------------------------------------------------

    def configure(self, config: ToolActivationConfig) -> None:
        """
        Activate a subset of tools based on runtime config.
        """
        self._last_config = config
        active: Dict[str, RuntimeToolSpec] = {}

        wanted_profiles = config.normalized_profile_set()
        wanted_domains = config.normalized_domain_set()
        capabilities = config.capability_set()

        for tool in self._all_tools:
            if not tool.enabled_by_default:
                continue

            # strategy filter
            if config.strategy not in tool.strategies:
                continue

            # profile filter
            if tool.profiles and wanted_profiles.isdisjoint(tool.profiles):
                continue

            # domain filter
            if tool.domains and wanted_domains.isdisjoint(tool.domains):
                continue

            # capability filter
            if tool.requires and not tool.requires.issubset(capabilities):
                continue

            # extra runtime feature gates
            if not self._passes_runtime_feature_gate(tool, config):
                continue

            active[tool.name] = tool

        self._active_tools = active

    def _passes_runtime_feature_gate(self, tool: RuntimeToolSpec, config: ToolActivationConfig) -> bool:
        """
        Additional explicit feature gating for tools that should be controlled
        by high-level runtime flags, even if profile/capability technically match.
        """
        name = tool.name
        category = tool.category

        if category == "memory" and not config.enable_memory:
            return False

        if category == "mutation" and not config.enable_mutation:
            return False

        if name in {"web_search", "read_url"} and not config.enable_web:
            return False

        if name in {"get_document_overview", "read_document_section", "search_document"} and not config.enable_document:
            return False

        if name in {"search_code", "find_file", "read_file_chunk", "list_directory"} and not config.enable_code_tools:
            return False

        if name == "run_bash_command" and not config.enable_bash:
            return False

        if name == "execute_parallel_branches" and not config.enable_parallel:
            return False

        if category == "meta" and not config.enable_meta:
            return False

        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[RuntimeToolSpec]:
        """
        Get one currently active tool by name.
        """
        return self._active_tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._active_tools

    def all_tools(self) -> List[RuntimeToolSpec]:
        """
        Return all statically known tools.
        """
        return list(self._all_tools)

    def active_tools(self) -> List[RuntimeToolSpec]:
        """
        Return the currently active runtime tool subset.
        """
        return list(self._active_tools.values())

    def active_tool_names(self) -> List[str]:
        return sorted(self._active_tools.keys())

    # ------------------------------------------------------------------
    # Filtered views
    # ------------------------------------------------------------------

    def active_observation_tools(self) -> List[RuntimeToolSpec]:
        return [t for t in self._active_tools.values() if t.category in {"observation", "memory", "meta"}]

    def active_mutation_tools(self) -> List[RuntimeToolSpec]:
        return [t for t in self._active_tools.values() if t.category == "mutation"]

    # ------------------------------------------------------------------
    # Provider compilation
    # ------------------------------------------------------------------

    def compile_for_provider(self, provider: str, include_mutation: bool = False) -> List[Dict[str, Any]]:
        """
        Compile currently active tools to provider-specific schema.

        include_mutation
        ----------------
        - native_all: typically True
        - hybrid / text_only: typically False for provider-native tool exposure
        """
        compiled: List[Dict[str, Any]] = []

        for tool in self.active_tools():
            if tool.category == "mutation" and not include_mutation:
                continue

            if provider == "anthropic":
                compiled.append(tool.to_anthropic_tool())
            else:
                compiled.append(tool.to_openai_function())

        return compiled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary_dict(self) -> Dict[str, Any]:
        """
        Small diagnostic summary for logs / tests.
        """
        return {
            "configured": self._last_config is not None,
            "last_config": None if self._last_config is None else {
                "strategy": self._last_config.strategy,
                "profile": self._last_config.profile,
                "domain": self._last_config.domain,
                "enable_memory": self._last_config.enable_memory,
                "enable_web": self._last_config.enable_web,
                "enable_document": self._last_config.enable_document,
                "enable_code_tools": self._last_config.enable_code_tools,
                "enable_mutation": self._last_config.enable_mutation,
                "enable_bash": self._last_config.enable_bash,
                "enable_parallel": self._last_config.enable_parallel,
                "enable_meta": self._last_config.enable_meta,
            },
            "all_tool_count": len(self._all_tools),
            "active_tool_count": len(self._active_tools),
            "active_tools": self.active_tool_names(),
        }


# =============================================================================
# 5. Global singleton
# =============================================================================

GLOBAL_TOOL_REGISTRY = ToolRegistryRuntime()


# =============================================================================
# 6. Convenience helpers
# =============================================================================

def configure_global_tool_registry(
    strategy: str = "hybrid",
    profile: str = "general",
    domain: str = "general",
    enable_memory: bool = True,
    enable_web: bool = True,
    enable_document: bool = False,
    enable_code_tools: bool = True,
    enable_mutation: bool = True,
    enable_bash: bool = False,
    enable_parallel: bool = False,
    enable_meta: bool = True,
) -> ToolRegistryRuntime:
    """
    Configure the global runtime registry in one call.
    """
    cfg = ToolActivationConfig(
        strategy=strategy,
        profile=profile,
        domain=domain,
        enable_memory=enable_memory,
        enable_web=enable_web,
        enable_document=enable_document,
        enable_code_tools=enable_code_tools,
        enable_mutation=enable_mutation,
        enable_bash=enable_bash,
        enable_parallel=enable_parallel,
        enable_meta=enable_meta,
    )
    GLOBAL_TOOL_REGISTRY.configure(cfg)
    return GLOBAL_TOOL_REGISTRY
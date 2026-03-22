from __future__ import annotations

"""
test_tool_registry_runtime.py

Tests for the runtime tool registry.

What this validates
-------------------
1. Different profiles activate different tool subsets
2. Capability flags correctly enable/disable tool families
3. Strategy affects provider compilation behavior
4. Active tool lookup works as expected
5. Mutation tools are present/absent correctly
6. Memory/document/code/web tool gating works correctly
"""

from typing import List

from BatchAgent.tools.tool_registry_runtime import (
    ToolActivationConfig,
    ToolRegistryRuntime,
    configure_global_tool_registry,
    GLOBAL_TOOL_REGISTRY,
)


# =============================================================================
# Helpers
# =============================================================================

def _names(tools) -> List[str]:
    return sorted([t.name for t in tools])


def _assert_contains(names: List[str], expected: List[str]):
    missing = [x for x in expected if x not in names]
    assert not missing, f"Missing expected tools: {missing}\nActual: {names}"


def _assert_not_contains(names: List[str], forbidden: List[str]):
    present = [x for x in forbidden if x in names]
    assert not present, f"Unexpected tools present: {present}\nActual: {names}"


def _print_summary(registry: ToolRegistryRuntime, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    summary = registry.summary_dict()
    for k, v in summary.items():
        print(f"{k}: {v}")


# =============================================================================
# Tests
# =============================================================================

def test_coding_profile_activation():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="coding",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=False,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=True,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_coding_profile_activation")

    _assert_contains(names, [
        "search_code",
        "find_file",
        "read_file_chunk",
        "list_directory",
        "run_bash_command",
        "get_memory",
        "update_memory",
        "write_file",
        "search_and_replace",
        "web_search",
        "read_url",
        "finish_task",
    ])

    _assert_not_contains(names, [
        "get_document_overview",
        "read_document_section",
        "search_document",
    ])


def test_document_profile_activation():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="document",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=True,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_document_profile_activation")

    _assert_contains(names, [
        "get_document_overview",
        "read_document_section",
        "get_memory",
        "update_memory",
        "web_search",
        "read_url",
        "finish_task",
    ])

    # search_document only appears if you added it to tools_registry static definitions
    # so check softly:
    if "search_document" in names:
        print("search_document is active (good).")
    else:
        print("search_document not active; check if it exists in static registry.")

    _assert_not_contains(names, [
        "search_code",
        "find_file",
        "read_file_chunk",
        "list_directory",
        "run_bash_command",
        "write_file",
        "search_and_replace",
    ])


def test_research_profile_activation():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="native_all",
            profile="research",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=True,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=True,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_research_profile_activation")

    _assert_contains(names, [
        "web_search",
        "read_url",
        "get_memory",
        "update_memory",
        "finish_task",
    ])

    # Optional / placeholder tools may or may not exist in your static registry
    for maybe_name in ["execute_parallel_branches", "inspect_branch_details"]:
        if maybe_name in names:
            print(f"{maybe_name} is active.")

    _assert_not_contains(names, [
        "run_bash_command",
        "write_file",
        "search_and_replace",
    ])


def test_disable_memory_tools():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="coding",
            domain="general",
            enable_memory=False,
            enable_web=False,
            enable_document=False,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_disable_memory_tools")

    _assert_not_contains(names, [
        "get_memory",
        "update_memory",
    ])


def test_disable_mutation_tools():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="coding",
            domain="general",
            enable_memory=True,
            enable_web=False,
            enable_document=False,
            enable_code_tools=True,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_disable_mutation_tools")

    _assert_not_contains(names, [
        "write_file",
        "search_and_replace",
    ])


def test_disable_code_tools():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="general",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=False,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_disable_code_tools")

    _assert_not_contains(names, [
        "search_code",
        "find_file",
        "read_file_chunk",
        "list_directory",
        "run_bash_command",
    ])


def test_disable_web_tools():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="research",
            domain="general",
            enable_memory=True,
            enable_web=False,
            enable_document=True,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    names = registry.active_tool_names()
    _print_summary(registry, "test_disable_web_tools")

    _assert_not_contains(names, [
        "web_search",
        "read_url",
    ])


def test_compile_for_openai_hybrid():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="coding",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=False,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    compiled = registry.compile_for_provider(
        provider="openai",
        include_mutation=False,
    )
    print("\nCompiled openai tools (hybrid, no mutation):")
    for tool in compiled[:5]:
        print(tool)

    names = [t["function"]["name"] for t in compiled]
    _assert_not_contains(names, ["write_file", "search_and_replace"])


def test_compile_for_openai_native_all_with_mutation():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="native_all",
            profile="coding",
            domain="general",
            enable_memory=True,
            enable_web=False,
            enable_document=False,
            enable_code_tools=True,
            enable_mutation=True,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    compiled = registry.compile_for_provider(
        provider="openai",
        include_mutation=True,
    )
    names = [t["function"]["name"] for t in compiled]

    print("\nCompiled openai tools (native_all, with mutation):")
    print(names)

    _assert_contains(names, ["write_file", "search_and_replace"])


def test_compile_for_anthropic():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="document",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=True,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    compiled = registry.compile_for_provider(
        provider="anthropic",
        include_mutation=False,
    )

    print("\nCompiled anthropic tools:")
    for tool in compiled[:5]:
        print(tool)

    assert all("name" in t and "input_schema" in t for t in compiled)


def test_global_registry_helper():
    registry = configure_global_tool_registry(
        strategy="hybrid",
        profile="document",
        domain="general",
        enable_memory=True,
        enable_web=True,
        enable_document=True,
        enable_code_tools=False,
        enable_mutation=False,
        enable_bash=False,
        enable_parallel=False,
        enable_meta=True,
    )

    assert registry is GLOBAL_TOOL_REGISTRY

    names = registry.active_tool_names()
    _print_summary(registry, "test_global_registry_helper")

    _assert_contains(names, [
        "get_document_overview",
        "read_document_section",
    ])


def test_lookup_active_and_inactive_tool():
    registry = ToolRegistryRuntime()
    registry.configure(
        ToolActivationConfig(
            strategy="hybrid",
            profile="document",
            domain="general",
            enable_memory=True,
            enable_web=True,
            enable_document=True,
            enable_code_tools=False,
            enable_mutation=False,
            enable_bash=False,
            enable_parallel=False,
            enable_meta=True,
        )
    )

    assert registry.has("get_document_overview") is True
    assert registry.get("get_document_overview") is not None

    # code tools disabled in this profile/config
    assert registry.has("search_code") is False
    assert registry.get("search_code") is None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    test_coding_profile_activation()
    test_document_profile_activation()
    test_research_profile_activation()
    test_disable_memory_tools()
    test_disable_mutation_tools()
    test_disable_code_tools()
    test_disable_web_tools()
    test_compile_for_openai_hybrid()
    test_compile_for_openai_native_all_with_mutation()
    test_compile_for_anthropic()
    test_global_registry_helper()
    test_lookup_active_and_inactive_tool()
    print("\nAll tool_registry_runtime tests passed.")

"""
python -m BatchAgent.tools.test_tool_registry_runtime
"""
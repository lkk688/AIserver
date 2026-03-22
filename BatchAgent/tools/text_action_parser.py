from __future__ import annotations

"""
text_action_parser.py

Fallback parser for non-native tool calling modes.

Purpose
-------
Convert plain-text / XML-like LLM outputs into AgentAction objects.

This parser is used when:
- provider tool calling is unavailable
- strategy == "text_only"
- strategy == "hybrid" but file mutations are still emitted as XML/text
- the model output is unstable and we need robust recovery

Design goals
------------
1. Parse ONLY tools that are currently active in the runtime registry.
2. Support dynamically registered custom tools.
3. Keep mutation parsing separate and higher-priority.
4. Ignore <think>...</think> reasoning blocks.
5. Stay compatible with PromptRegistry's XML tool format.

Supported formats
-----------------
A) Canonical XML tool call:
   <tool_call><web_search><query>...</query></web_search></tool_call>

B) Canonical XML mutation:
   <tool_call><write_file><path>...</path><content>...</content></write_file></tool_call>

C) Legacy WRITE_FILE:
   WRITE_FILE: ./foo.py
   ```python
   ...
   ```

D) Legacy WRITE_FILE content block:
WRITE_FILE: ./foo.py
<<<CONTENT
...
CONTENT>>>

E) Unified diff output
"""

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from BatchAgent.mini_batch_agent_base import (
    AgentAction,
    ActionToolCall,
    ActionWriteFile,
    ActionApplyDiff,
    ActionReplaceText,
)
from BatchAgent.tools.local_tools import (
    resolve_path,
    extract_write_file_actions_v2,
    extract_all_diffs,
)
from BatchAgent.tools.tool_registry_runtime import GLOBAL_TOOL_REGISTRY

# =============================================================================
# Runtime schema helpers
# =============================================================================

def _tool_spec_to_schema_dict(tool: Any) -> Dict[str, Any]:
    """
    Normalize a runtime tool spec object into a simple schema dict.

    Supports:
    - dataclass-like specs
    - dict-like specs
    - plain objects with attributes
    """
    if isinstance(tool, dict):
        return {
            "name": tool.get("name", ""),
            "properties": dict(tool.get("properties", {}) or {}),
            "required": list(tool.get("required", []) or []),
            "category": tool.get("category", "observation"),
        }

    if is_dataclass(tool):
        obj = asdict(tool)
        return {
            "name": obj.get("name", ""),
            "properties": dict(obj.get("properties", {}) or {}),
            "required": list(obj.get("required", []) or []),
            "category": obj.get("category", "observation"),
        }

    return {
        "name": getattr(tool, "name", ""),
        "properties": dict(getattr(tool, "properties", {}) or {}),
        "required": list(getattr(tool, "required", []) or []),
        "category": getattr(tool, "category", "observation"),
    }

def build_runtime_tool_schema_map(
    dynamic_tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build the tool schema map from CURRENT ACTIVE runtime tools, then merge
    dynamic custom tool schemas if provided.

    Parameters
    ----------
    dynamic_tools_registry:
        Example:
        {
            "my_custom_tool": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "category": "meta",
            }
        }
    """
    schema_map: Dict[str, Dict[str, Any]] = {}

    for tool in GLOBAL_TOOL_REGISTRY.active_tools():
        spec = _tool_spec_to_schema_dict(tool)
        name = spec["name"]
        if not name:
            continue
        schema_map[name] = spec

    if dynamic_tools_registry:
        for tool_name, spec in dynamic_tools_registry.items():
            schema_map[tool_name] = {
                "name": tool_name,
                "properties": dict(spec.get("properties", {}) or {}),
                "required": list(spec.get("required", []) or []),
                "category": spec.get("category", "meta"),
            }

    return schema_map

def _coerce_tool_args(
    tool_name: str,
    args_dict: Dict[str, Any],
    schema_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Apply lightweight type coercion based on schema.
    """
    schema = schema_map.get(tool_name, {})
    props = schema.get("properties", {}) or {}

    out = dict(args_dict)

    for key, prop in props.items():
        if key not in out:
            continue

        raw = out[key]
        expected_type = prop.get("type")

        if expected_type == "integer":
            try:
                out[key] = int(str(raw).strip())
            except Exception:
                pass
        elif expected_type == "number":
            try:
                out[key] = float(str(raw).strip())
            except Exception:
                pass
        elif expected_type == "boolean":
            sval = str(raw).strip().lower()
            if sval in {"true", "1", "yes"}:
                out[key] = True
            elif sval in {"false", "0", "no"}:
                out[key] = False
        elif expected_type == "array":
            # Minimal convenience: comma-separated fallback
            if isinstance(raw, str) and "," in raw:
                out[key] = [x.strip() for x in raw.split(",") if x.strip()]

    return out

# =============================================================================
# XML/text parsing helpers
# =============================================================================

def _strip_think_blocks(content: str) -> str:
    """
    Remove <think>...</think> blocks so internal draft tool calls are ignored.
    """
    stripped = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return stripped if stripped.strip() else content

def _parse_child_tags(inner_content: str) -> Dict[str, str]:
    """
    Parse child XML tags inside a tool body.

    Example:
        <query>abc</query><category>news</category>
        -> {"query": "abc", "category": "news"}
    """
    args: Dict[str, str] = {}

    for arg_match in re.finditer(
        r"<([a-zA-Z0-9_]+)>(.*?)</\1>",
        inner_content,
        re.DOTALL,
    ):
        arg_name = arg_match.group(1).strip()
        arg_val = arg_match.group(2).strip()
        args[arg_name] = arg_val

    return args

def _parse_single_arg_fallback(
    tool_name: str,
    inner_content: str,
    schema_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Support:
    <web_search>latest qwen3.5 news</web_search>

    If the tool has at least one parameter, map the content to the first parameter.
    """
    if not inner_content or "<" in inner_content:
        return {}

    schema = schema_map.get(tool_name, {})
    props = schema.get("properties", {}) or {}
    if not props:
        return {}

    first_arg_name = list(props.keys())[0]
    return {first_arg_name: inner_content.strip()}

def _default_args_for_tool(tool_name: str, args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tool-specific defaults for omitted arguments.
    """
    if tool_name == "list_directory" and not args_dict:
        return {"dir_path": "."}
    return args_dict

def _dedupe_actions(actions: List[AgentAction]) -> List[AgentAction]:
    """
    Deduplicate actions conservatively using a stable signature.
    """
    seen = set()
    out: List[AgentAction] = []

    for a in actions:
        try:
            if isinstance(a, ActionToolCall):
                sig = ("tool", a.name, json.dumps(a.args, sort_keys=True, ensure_ascii=False))
            elif isinstance(a, ActionWriteFile):
                sig = ("write", a.path, a.content)
            elif isinstance(a, ActionReplaceText):
                sig = ("replace", a.path, a.old_text, a.new_text)
            elif isinstance(a, ActionApplyDiff):
                sig = ("diff", a.diff_text)
            else:
                sig = ("other", repr(a))
        except Exception:
            sig = ("fallback", repr(a))

        if sig in seen:
            continue
        seen.add(sig)
        out.append(a)

    return out

# =============================================================================
# Mutation parsers
# =============================================================================

def _parse_write_file_xml(parse_content: str, allowlist: List[str]) -> List[AgentAction]:
    actions: List[AgentAction] = []

    for match in re.finditer(
        r"(?:<tool_call>\s*)?<write_file>\s*<path>\s*(.*?)\s*</path>\s*<content>(.*?)</content>\s*</write_file>(?:\s*</tool_call>)?",
        parse_content,
        re.DOTALL | re.IGNORECASE,
    ):
        fpath = match.group(1).strip()
        file_content = match.group(2)
        target_path = resolve_path(fpath, allowlist)
        if target_path and len(file_content.strip()) >= 5:
            actions.append(
                ActionWriteFile(
                    path=str(target_path),
                    content=file_content.rstrip() + "\n",
                )
            )

    return actions

def _parse_replace_text_xml(parse_content: str, allowlist: List[str]) -> List[AgentAction]:
    actions: List[AgentAction] = []

    for match in re.finditer(
        r"(?:<tool_call>\s*)?<search_and_replace>\s*<path>\s*(.*?)\s*</path>\s*<old_text>(.*?)</old_text>\s*<new_text>(.*?)</new_text>\s*</search_and_replace>(?:\s*</tool_call>)?",
        parse_content,
        re.DOTALL | re.IGNORECASE,
    ):
        fpath = match.group(1).strip()
        old_txt = match.group(2)
        new_txt = match.group(3)
        target_path = resolve_path(fpath, allowlist)
        if target_path:
            actions.append(
                ActionReplaceText(
                    path=str(target_path),
                    old_text=old_txt,
                    new_text=new_txt,
                )
            )

    return actions

def _parse_write_file_old_xml(parse_content: str, allowlist: List[str]) -> List[AgentAction]:
    actions: List[AgentAction] = []

    for match in re.finditer(
        r"(?:<tool_call>\s*)?<write_file>\s*path\s*=\s*(\S+)\s*\ncontent=(.*?)</write_file>(?:\s*</tool_call>)?",
        parse_content,
        re.DOTALL | re.IGNORECASE,
    ):
        fpath = match.group(1).strip()
        file_content = match.group(2).rstrip()
        target_path = resolve_path(fpath, allowlist)
        if target_path and len(file_content.strip()) >= 5:
            actions.append(
                ActionWriteFile(
                    path=str(target_path),
                    content=file_content + "\n",
                )
            )

    return actions

def _parse_write_file_fenced(parse_content: str, allowlist: List[str]) -> List[AgentAction]:
    actions: List[AgentAction] = []

    for match in re.finditer(
        r"WRITE_FILE:\s*(\S+)\s*\n```(?:\w+)?\n(.*?)```",
        parse_content,
        re.DOTALL,
    ):
        fpath = match.group(1).strip()
        file_content = match.group(2).rstrip()
        target_path = resolve_path(fpath, allowlist)
        if target_path and len(file_content.strip()) >= 5:
            actions.append(
                ActionWriteFile(
                    path=str(target_path),
                    content=file_content + "\n",
                )
            )

    return actions


def _parse_write_file_legacy_content_blocks(content: str, allowlist: List[str]) -> List[AgentAction]:
    actions: List[AgentAction] = []

    write_actions = extract_write_file_actions_v2(content)
    for path, text in write_actions:
        target_path = resolve_path(path, allowlist)
        if target_path:
            actions.append(ActionWriteFile(path=str(target_path), content=text))

    return actions

def _parse_diff(content: str) -> List[AgentAction]:
    diff = extract_all_diffs(content)
    if diff:
        return [ActionApplyDiff(diff_text=diff)]
    return []

# =============================================================================
# Generic XML tool parser
# =============================================================================

def _parse_generic_xml_tool_calls(
    parse_content: str,
    schema_map: Dict[str, Dict[str, Any]],
) -> List[AgentAction]:
    actions: List[AgentAction] = []
    active_tool_names = set(schema_map.keys())

    for match in re.finditer(
        r"(?:<tool_call>\s*)?<([a-zA-Z0-9_]+)(?:\s*/>|>(.*?)</\1>)(?:\s*</tool_call>)?",
        parse_content,
        re.DOTALL,
    ):
        tool_name = match.group(1).strip()
        inner_content = (match.group(2) or "").strip()

        # Specialized mutation tools are handled separately
        if tool_name in {"write_file", "search_and_replace"}:
            continue

        # Only parse tools currently available in runtime registry
        if tool_name not in active_tool_names:
            continue

        schema = schema_map.get(tool_name, {})
        if schema.get("category") == "mutation":
            continue

        args_dict = _parse_child_tags(inner_content)

        if not args_dict:
            args_dict = _parse_single_arg_fallback(
                tool_name=tool_name,
                inner_content=inner_content,
                schema_map=schema_map,
            )

        args_dict = _default_args_for_tool(tool_name, args_dict)
        args_dict = _coerce_tool_args(tool_name, args_dict, schema_map)

        actions.append(ActionToolCall(name=tool_name, args=args_dict))

    return actions

def _parse_markdown_json_tool_calls(
    parse_content: str,
    schema_map: Dict[str, Dict[str, Any]],
) -> List[AgentAction]:
    actions: List[AgentAction] = []
    active_tool_names = set(schema_map.keys())

    for match in re.finditer(r"```(?:json)\s*(.*?)\s*```", parse_content, re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        try:
            parsed = json.loads(block)
            if not isinstance(parsed, dict):
                continue
                
            tool_name = parsed.get("tool") or parsed.get("name") or parsed.get("action")
            if not tool_name or tool_name not in active_tool_names:
                continue
                
            args_dict = parsed.get("arguments") or parsed.get("parameters") or parsed.get("args")
            
            if args_dict is None:
                args_dict = {k: v for k, v in parsed.items() if k not in ["tool", "name", "action"]}
                
            if not isinstance(args_dict, dict):
                if isinstance(args_dict, str):
                    try:
                        args_dict = json.loads(args_dict)
                    except Exception:
                        args_dict = {}
                else:
                    args_dict = {}
                    
            args_dict = _default_args_for_tool(tool_name, args_dict)
            args_dict = _coerce_tool_args(tool_name, args_dict, schema_map)

            if tool_name == "write_file":
                content = args_dict.get("content", "")
                path = args_dict.get("path") or args_dict.get("file_path")
                if path and len(content.strip()) >= 5:
                    actions.append(ActionWriteFile(path=path, content=content + "\n"))
                continue
            elif tool_name == "search_and_replace":
                path = args_dict.get("path") or args_dict.get("file_path")
                if path:
                    actions.append(ActionReplaceText(
                        path=path, 
                        old_text=args_dict.get("old_text", ""),
                        new_text=args_dict.get("new_text", "")
                    ))
                continue

            actions.append(ActionToolCall(name=tool_name, args=args_dict))
        except json.JSONDecodeError:
            continue
            
    return actions

# =============================================================================
# Public API
# =============================================================================

def parse_text_actions(
    content: str,
    allowlist: List[str],
    dynamic_tools_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[AgentAction]:
    """
    Parse text/XML fallback tool calls into AgentAction objects.

    Parameters
    ----------
    content:
        Raw model output
    allowlist:
        Allowed file paths for file mutations
    dynamic_tools_registry:
        Optional runtime custom tool schema registry, for example:
        {
            "my_custom_tool": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "category": "meta",
            }
        }

    Returns
    -------
    List[AgentAction]
    """
    actions: List[AgentAction] = []
    parse_content = _strip_think_blocks(content)
    schema_map = build_runtime_tool_schema_map(
        dynamic_tools_registry=dynamic_tools_registry
    )

    # ------------------------------------------------------------------
    # 1. Mutation parsing first
    # ------------------------------------------------------------------
    actions.extend(_parse_write_file_xml(parse_content, allowlist))
    actions.extend(_parse_replace_text_xml(parse_content, allowlist))

    if not any(isinstance(a, ActionWriteFile) for a in actions):
        actions.extend(_parse_write_file_old_xml(parse_content, allowlist))

    if not any(isinstance(a, ActionWriteFile) for a in actions):
        actions.extend(_parse_write_file_fenced(parse_content, allowlist))

    actions.extend(_parse_write_file_legacy_content_blocks(content, allowlist))
    actions.extend(_parse_diff(content))

    # ------------------------------------------------------------------
    # 2. Generic active-tool XML and JSON Markdown parsing
    # ------------------------------------------------------------------
    actions.extend(_parse_generic_xml_tool_calls(parse_content, schema_map))
    actions.extend(_parse_markdown_json_tool_calls(parse_content, schema_map))

    # ------------------------------------------------------------------
    # 3. Deduplicate
    # ------------------------------------------------------------------
    return _dedupe_actions(actions)

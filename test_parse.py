import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Developer/AIserver").resolve()))
from BatchAgent.tools.text_action_parser import _parse_markdown_json_tool_calls
text = """
```json
{
  "name": "web_search",
  "arguments": {
    "query": "Python 3.13 latest news 2026"
  }
}
```
"""
schema_map = {"web_search": {"name": "web_search", "properties": {"query": {}}}}
actions = _parse_markdown_json_tool_calls(text, schema_map)
print(actions)

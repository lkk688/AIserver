import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Developer/AIserver").resolve()))
from CodeAgent.mini_code_agent_v2 import parse_text_actions, ANTHROPIC_TOOLS, ActionWriteFile

qwen_resp = """
## Reasoning
I need to update `hello.py`.
## Action
```diff
diff --git a/hello.py b/hello.py
--- a/hello.py
+++ b/hello.py
@@ -1,2 +1,3 @@
 def main():
-    pass
+    print("Hello")
```
"""
actions = parse_text_actions(qwen_resp, ["hello.py"])
print("Qwen actions:", actions)

qwen_write = """
WRITE_FILE: hello.py
<<<CONTENT
def new_main():
    pass
CONTENT>>>
"""
actions2 = parse_text_actions(qwen_write, ["hello.py"])
print("Qwen Write Actions:", actions2)
print("Anthropic Tools length:", len(ANTHROPIC_TOOLS))

import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Developer/AIserver").resolve()))
from CodeAgent.mini_code_agent_v2 import parse_text_actions, ActionToolCall

resp = """
Here are my commands.
<run_bash_command>
<command>ls -l</command>
</run_bash_command>

Also view:
<read_file_chunk>
<filepath>test.py</filepath>
<start_line>10</start_line>
<end_line>20</end_line>
</read_file_chunk>

And list:
<list_directory>
<dir_path>CodeAgent</dir_path>
</list_directory>
"""
actions = parse_text_actions(resp, ["test.py"])
for a in actions:
    print(a)

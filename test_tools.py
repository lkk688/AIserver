import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Developer/AIserver").resolve()))
from BatchAgent.tools.tools_registry import get_base_tools
print([t["name"] for t in get_base_tools("native_all", domain="research")])

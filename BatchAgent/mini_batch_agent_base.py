from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys

# ---------------------------
# Agent Action Protocol
# ---------------------------

@dataclass
class AgentAction:
    pass

@dataclass
class ActionWriteFile(AgentAction):
    path: str
    content: str

@dataclass
class ActionReplaceText(AgentAction):
    path: str
    old_text: str
    new_text: str

@dataclass
class ActionApplyDiff(AgentAction):
    diff_text: str

@dataclass
class ActionToolCall(AgentAction):
    name: str
    args: Dict[str, Any]


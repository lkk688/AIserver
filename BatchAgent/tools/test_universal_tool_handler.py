from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from BatchAgent.tools.universal_tool_handler import UniversalToolHandler
from BatchAgent.mini_batch_agent_base import (
    ActionToolCall,
    ActionWriteFile,
    ActionReplaceText,
)
from BatchAgent.tools.tool_registry_runtime import configure_global_tool_registry
from BatchAgent.working_memory import WorkingMemory


@dataclass
class DummyConfig:
    session_dir: Path
    workspace_dir: Path
    tool_strategy: str = "hybrid"
    task_profile: str = "coding"
    tool_domain: str = "general"
    auto_approve: bool = False
    current_time: str = "2026-03-21 10:00:00"
    serper_api_key: str = ""
    tavily_api_key: str = ""
    enable_youtube: bool = False
    sandbox_container: str | None = None
    working_memory: object | None = None


def _configure_registry_for_test(profile: str, memory_enabled: bool = True):
    configure_global_tool_registry(
        strategy="hybrid",
        profile=profile,
        domain="general",
        enable_memory=memory_enabled,
        enable_web=False,
        enable_document=(profile in {"document", "research"}),
        enable_code_tools=(profile in {"coding", "general"}),
        enable_mutation=True,
        enable_bash=False,
        enable_parallel=False,
        enable_meta=True,
    )


def _make_handler(tmpdir: Path, profile: str = "coding") -> UniversalToolHandler:
    session_dir = tmpdir / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    wm = WorkingMemory(
        session_dir=session_dir,
        goal="test task",
        inputs=[],
    )

    config = DummyConfig(
        session_dir=session_dir,
        workspace_dir=tmpdir,
        task_profile=profile,
        working_memory=wm,
    )

    _configure_registry_for_test(profile=profile, memory_enabled=True)

    allowlist = [
        str(tmpdir / "a.py"),
        str(tmpdir / "b.txt"),
        str(session_dir / "output.py"),
        str(session_dir / "notes.md"),
    ]

    turn_dir = tmpdir / "turn"
    turn_dir.mkdir(parents=True, exist_ok=True)

    return UniversalToolHandler(
        config=config,
        turn_dir=turn_dir,
        allowlist=allowlist,
        dynamic_tools_mapping={},
    )


def test_write_file_basic():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        actions = [
            ActionWriteFile(
                path="output.py",
                content="print('hello world')\n",
            )
        ]

        has_mutation, observation = handler.execute(actions, full_llm_content="")
        print("has_mutation:", has_mutation)
        print(observation)

        out_file = tmpdir / "session" / "output.py"
        assert has_mutation is True
        assert out_file.exists()
        assert "hello world" in out_file.read_text(encoding="utf-8")


def test_replace_text_basic():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        target = tmpdir / "b.txt"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        actions = [
            ActionReplaceText(
                path=str(target),
                old_text="beta",
                new_text="BETA",
            )
        ]

        has_mutation, observation = handler.execute(actions, full_llm_content="")
        print("has_mutation:", has_mutation)
        print(observation)

        assert has_mutation is True
        assert "BETA" in target.read_text(encoding="utf-8")


def test_finish_task():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        actions = [
            ActionToolCall(
                name="finish_task",
                args={"summary": "done successfully"},
            )
        ]

        has_mutation, observation = handler.execute(actions, full_llm_content="")
        print("has_mutation:", has_mutation)
        print(observation)

        assert has_mutation is False
        assert handler.task_finished is True
        assert handler.finish_summary == "done successfully"


def test_list_directory_tool():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        (tmpdir / "x.txt").write_text("x", encoding="utf-8")
        (tmpdir / "y.txt").write_text("y", encoding="utf-8")

        actions = [
            ActionToolCall(
                name="list_directory",
                args={"dir_path": str(tmpdir)},
            )
        ]

        has_mutation, observation = handler.execute(actions, full_llm_content="")
        print("has_mutation:", has_mutation)
        print(observation)

        assert has_mutation is False
        assert "x.txt" in observation or "y.txt" in observation


def test_read_file_chunk_tool():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        target = tmpdir / "a.py"
        target.write_text(
            "line1\nline2\nline3\nline4\n",
            encoding="utf-8",
        )

        actions = [
            ActionToolCall(
                name="read_file_chunk",
                args={
                    "filepath": str(target),
                    "start_line": 2,
                    "end_line": 3,
                },
            )
        ]

        has_mutation, observation = handler.execute(actions, full_llm_content="")
        print("has_mutation:", has_mutation)
        print(observation)

        assert has_mutation is False
        assert "line2" in observation
        assert "line3" in observation


def test_get_and_update_memory():
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        handler = _make_handler(tmpdir, profile="coding")

        actions1 = [
            ActionToolCall(name="get_memory", args={"section": "control"})
        ]
        has_mutation1, observation1 = handler.execute(actions1, full_llm_content="")
        print(observation1)
        assert has_mutation1 is False
        assert "planning" in observation1 or "control" in observation1

        actions2 = [
            ActionToolCall(
                name="update_memory",
                args={"patch": {"control": {"current_step": "testing tools"}}},
            )
        ]
        has_mutation2, observation2 = handler.execute(actions2, full_llm_content="")
        print(observation2)
        assert has_mutation2 is False
        assert "Updated" in observation2 or "updated" in observation2

        actions3 = [
            ActionToolCall(name="get_memory", args={"section": "control"})
        ]
        has_mutation3, observation3 = handler.execute(actions3, full_llm_content="")
        print(observation3)
        assert has_mutation3 is False
        assert "testing tools" in observation3


def test_parsing_text_only():
    from BatchAgent.tools.text_action_parser import parse_text_actions
    from BatchAgent.mini_batch_agent_base import ActionToolCall
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _configure_registry_for_test(profile="coding")
        content = "<tool_call><list_directory><dir_path>.</dir_path></list_directory></tool_call>"
        actions = parse_text_actions(content, allowlist=[])
        assert len(actions) == 1
        assert isinstance(actions[0], ActionToolCall)
        assert actions[0].name == "list_directory"
        assert actions[0].args == {"dir_path": "."}

def test_parsing_hybrid():
    from BatchAgent.tools.text_action_parser import parse_text_actions
    from BatchAgent.mini_batch_agent_base import ActionWriteFile
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _configure_registry_for_test(profile="coding")
        content = "<write_file><path>test.py</path><content>print('ok')</content></write_file>"
        actions = parse_text_actions(content, allowlist=["test.py"])
        assert len(actions) == 1
        assert isinstance(actions[0], ActionWriteFile)
        assert actions[0].path.endswith("test.py")
        assert "print('ok')" in actions[0].content

def test_parsing_native_all():
    from BatchAgent.tools.text_action_parser import parse_text_actions
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        _configure_registry_for_test(profile="coding")
        # In native_all, content usually lacks tool tags (except maybe think)
        content = "Here is my response without any XML tool calls since I use native JSON."
        actions = parse_text_actions(content, allowlist=[])
        assert len(actions) == 0


if __name__ == "__main__":
    test_write_file_basic()
    test_replace_text_basic()
    test_finish_task()
    test_list_directory_tool()
    test_read_file_chunk_tool()
    test_get_and_update_memory()
    test_parsing_text_only()
    test_parsing_hybrid()
    test_parsing_native_all()
    print("All universal_tool_handler tests passed.")

"""
python -m BatchAgent.tools.test_universal_tool_handler
"""
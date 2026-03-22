from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path


# ----------------------------------------------------------------------
# Dummy collaborators
# ----------------------------------------------------------------------

class DummyMemory:
    def __init__(self):
        self.data = {
            "control": {"stage": "idle"},
            "progress": {"done": []},
        }

    def get_memory(self, section):
        return self.data.get(section, {})

    def dump(self):
        return self.data

    def update_memory(self, patch_dict):
        for k, v in patch_dict.items():
            if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                self.data[k].update(v)
            else:
                self.data[k] = v


class DummyDocumentTools:
    def __init__(self, config=None):
        self.active_path = None

    def get_overview(self, filepath=None):
        if filepath:
            self.active_path = filepath
            return f"Loaded document: {filepath}\nSections:\n- [sec_1] Intro"
        if self.active_path:
            return f"Active document: {self.active_path}\nSections:\n- [sec_1] Intro"
        return "No active document."

    def read_section(self, section_id):
        return f"Section {section_id}: Lorem ipsum."

    def read_page(self, page):
        return f"Page {page}: page content"

    def search_document(self, query, top_k=5):
        return f"Document search for '{query}' (top_k={top_k})"


@dataclass
class DummyConfig:
    workspace_dir: Path
    current_time: str = "2026-03-21 10:00:00"
    serper_api_key: str = ""
    tavily_api_key: str = ""
    enable_youtube: bool = False
    sandbox_container: str | None = None
    working_memory: object | None = None


# ----------------------------------------------------------------------
# Build router helper
# ----------------------------------------------------------------------

def make_router():
    from BatchAgent.tools.tool_router import ToolRouter
    with tempfile.TemporaryDirectory() as td:
        cfg = DummyConfig(
            workspace_dir=Path(td),
            working_memory=DummyMemory(),
        )
        router = ToolRouter(
            config=cfg,
            dynamic_tools_mapping={},
            document_tools=DummyDocumentTools(cfg),
        )
        return router, cfg


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_memory_get_and_update():
    router, _ = make_router()

    out1 = router._handle_get_memory({"section": "control"})
    print(out1)
    assert "idle" in out1

    out2 = router._handle_update_memory({"patch": {"control": {"stage": "running"}}})
    print(out2)
    assert "updated" in out2.lower()

    out3 = router._handle_get_memory({"section": "control"})
    print(out3)
    assert "running" in out3


def test_document_overview_and_read():
    router, _ = make_router()

    out1 = router._handle_get_document_overview({"filepath": "data/paper.pdf"})
    print(out1)
    assert "Loaded document" in out1

    out2 = router._handle_read_document_section({"section_id": "sec_1"})
    print(out2)
    assert "sec_1" in out2

    out3 = router._handle_read_document_section({"page": 3})
    print(out3)
    assert "Page 3" in out3


def test_document_search():
    router, _ = make_router()
    out = router._handle_search_document({"query": "attention", "top_k": 3})
    print(out)
    assert "attention" in out
    assert "top_k=3" in out


def test_register_custom_tool():
    router, _ = make_router()

    out = router._handle_register_custom_tool({
        "tool_name": "my_tool",
        "description": "does something",
        "schema_properties": {"x": {"type": "string"}},
        "required_args": ["x"],
        "script_path": "tools/my_tool.py",
    })
    print(out)
    assert "Registered custom tool 'my_tool'" in out
    assert "my_tool" in router.ctx.dynamic_tools_mapping


def test_run_bash_guardrail():
    router, _ = make_router()

    long_cmd = "echo " + ("x" * 400)
    out = router._handle_run_bash_command({"command": long_cmd})
    print(out)
    assert "Guardrail" in out or "forbidden" in out

    short_cmd = "pwd"
    out2 = router._handle_run_bash_command({"command": short_cmd})
    print(out2)
    assert isinstance(out2, str)


def test_finish_task_handler():
    router, _ = make_router()
    out = router._handle_finish_task({"summary": "completed"})
    print(out)
    assert "completed" in out


def test_unknown_dispatch_returns_error():
    router, _ = make_router()
    out = router.dispatch("definitely_unknown_tool_name", {})
    print(out)
    assert "Unknown or inactive tool" in out


if __name__ == "__main__":
    test_memory_get_and_update()
    test_document_overview_and_read()
    test_document_search()
    test_register_custom_tool()
    test_run_bash_guardrail()
    test_finish_task_handler()
    test_unknown_dispatch_returns_error()
    print("All tool_router tests passed.")

"""
python -m BatchAgent.tools.test_tool_router
"""
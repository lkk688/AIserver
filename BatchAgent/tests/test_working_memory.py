"""
Tests for BatchAgent/working_memory.py

Run with:
    pytest BatchAgent/tests/test_working_memory.py -v
"""

import json
import tempfile
import warnings
from pathlib import Path

import pytest

from BatchAgent.working_memory import (
    WorkingMemory,
    _deep_merge,
    _make_default_memory,
    _validate_memory,
    handle_get_memory,
    handle_update_memory,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_session(tmp_path):
    """Yield a temporary session directory."""
    return tmp_path


@pytest.fixture
def mem(tmp_session):
    """Yield a fresh WorkingMemory instance."""
    return WorkingMemory(tmp_session, goal="Test goal", inputs=["input.txt"])


# ──────────────────────────────────────────────────────────────
# _deep_merge
# ──────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_scalar_overwrite(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_nested_dict_merge(self):
        base = {"control": {"stage": "planning", "status": "active"}}
        patch = {"control": {"stage": "gathering"}}
        result = _deep_merge(base, patch)
        assert result["control"]["stage"] == "gathering"
        assert result["control"]["status"] == "active"  # preserved

    def test_list_append_dedup(self):
        base = {"facts": ["a", "b"]}
        patch = {"facts": ["b", "c"]}
        result = _deep_merge(base, patch)
        assert result["facts"] == ["a", "b", "c"]

    def test_list_dedup_dicts(self):
        base = {"items": [{"id": 1}]}
        patch = {"items": [{"id": 1}, {"id": 2}]}
        result = _deep_merge(base, patch)
        assert len(result["items"]) == 2

    def test_new_key_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_base_not_mutated(self):
        base = {"control": {"stage": "planning"}}
        _ = _deep_merge(base, {"control": {"stage": "done"}})
        assert base["control"]["stage"] == "planning"


# ──────────────────────────────────────────────────────────────
# _validate_memory
# ──────────────────────────────────────────────────────────────

class TestValidateMemory:
    def test_empty_dict_gets_all_sections(self):
        result = _validate_memory({})
        assert "control" in result
        assert "knowledge" in result
        assert "artifacts" in result

    def test_missing_sub_key_filled(self):
        partial = {"control": {"stage": "planning"}}
        result = _validate_memory(partial)
        assert "status" in result["control"]

    def test_extra_keys_preserved(self):
        extra = _make_default_memory()
        extra["my_custom_key"] = "hello"
        result = _validate_memory(extra)
        assert result["my_custom_key"] == "hello"

    def test_corrupted_section_replaced(self):
        bad = _make_default_memory()
        bad["knowledge"] = "not a dict"  # type: ignore[assignment]
        result = _validate_memory(bad)
        assert isinstance(result["knowledge"], dict)


# ──────────────────────────────────────────────────────────────
# WorkingMemory.get_memory
# ──────────────────────────────────────────────────────────────

class TestGetMemory:
    def test_full_memory_returns_all_sections(self, mem):
        full = mem.get_memory()
        for section in ("control", "task", "knowledge", "progress", "actions", "artifacts"):
            assert section in full

    def test_section_returns_correct_data(self, mem):
        task = mem.get_memory("task")
        assert task["goal"] == "Test goal"
        assert "input.txt" in task["inputs"]

    def test_returns_deep_copy(self, mem):
        full = mem.get_memory()
        full["control"]["stage"] = "MUTATED"
        assert mem.get_memory("control")["stage"] != "MUTATED"

    def test_unknown_section_raises(self, mem):
        with pytest.raises(ValueError, match="Unknown section"):
            mem.get_memory("nonexistent")


# ──────────────────────────────────────────────────────────────
# WorkingMemory.update_memory
# ──────────────────────────────────────────────────────────────

class TestUpdateMemory:
    def test_scalar_update(self, mem):
        mem.update_memory({"control": {"stage": "gathering"}})
        assert mem.get_memory("control")["stage"] == "gathering"

    def test_list_append(self, mem):
        mem.update_memory({"progress": {"completed_steps": ["Step 1"]}})
        mem.update_memory({"progress": {"completed_steps": ["Step 2"]}})
        steps = mem.get_memory("progress")["completed_steps"]
        assert "Step 1" in steps
        assert "Step 2" in steps

    def test_list_no_duplicates(self, mem):
        mem.update_memory({"knowledge": {"key_facts": ["fact A"]}})
        mem.update_memory({"knowledge": {"key_facts": ["fact A"]}})
        facts = mem.get_memory("knowledge")["key_facts"]
        assert facts.count("fact A") == 1

    def test_unknown_section_ignored(self, mem):
        mem.update_memory({"evil_key": {"bad": "data"}})
        full = mem.get_memory()
        assert "evil_key" not in full

    def test_timestamp_refreshed(self, mem):
        before = mem.get_memory("artifacts")["timestamps"]["last_updated"]
        import time; time.sleep(0.01)
        mem.update_memory({"control": {"stage": "done"}})
        after = mem.get_memory("artifacts")["timestamps"]["last_updated"]
        assert after >= before

    def test_non_dict_patch_raises(self, mem):
        with pytest.raises(TypeError):
            mem.update_memory("not a dict")  # type: ignore[arg-type]

    def test_summary_dict_merge(self, mem):
        mem.update_memory({"knowledge": {"summaries": {"sec_1": "Intro"}}})
        mem.update_memory({"knowledge": {"summaries": {"sec_2": "Method"}}})
        summaries = mem.get_memory("knowledge")["summaries"]
        assert summaries["sec_1"] == "Intro"
        assert summaries["sec_2"] == "Method"


# ──────────────────────────────────────────────────────────────
# WorkingMemory.record_action / record_artifact / record_failure
# ──────────────────────────────────────────────────────────────

class TestRecordHelpers:
    def test_record_action_appends(self, mem):
        mem.record_action("Called read_document_section(sec_1)")
        assert "Called read_document_section(sec_1)" in mem.get_memory("actions")["recent_actions"]

    def test_record_action_capped(self, mem):
        from BatchAgent.working_memory import MAX_RECENT_ACTIONS
        for i in range(MAX_RECENT_ACTIONS + 5):
            mem.record_action(f"action {i}")
        assert len(mem.get_memory("actions")["recent_actions"]) == MAX_RECENT_ACTIONS

    def test_record_action_truncated(self, mem):
        from BatchAgent.working_memory import MAX_RECENT_ACTION_CHARS
        long_desc = "x" * (MAX_RECENT_ACTION_CHARS + 50)
        mem.record_action(long_desc)
        stored = mem.get_memory("actions")["recent_actions"][-1]
        assert len(stored) == MAX_RECENT_ACTION_CHARS

    def test_record_artifact_output(self, mem):
        mem.record_artifact("report.md", is_output=True)
        assert "report.md" in mem.get_memory("artifacts")["output_files"]

    def test_record_artifact_scratch(self, mem):
        mem.record_artifact("tmp_notes.txt", is_output=False)
        assert "tmp_notes.txt" in mem.get_memory("artifacts")["scratch_files"]

    def test_record_artifact_no_dup(self, mem):
        mem.record_artifact("output.md")
        mem.record_artifact("output.md")
        assert mem.get_memory("artifacts")["output_files"].count("output.md") == 1

    def test_record_failure(self, mem):
        mem.record_failure("read_url → timeout")
        assert any("read_url" in f for f in mem.get_memory("actions")["failed_attempts"])


# ──────────────────────────────────────────────────────────────
# WorkingMemory.compact_memory
# ──────────────────────────────────────────────────────────────

class TestCompactMemory:
    def test_trims_key_facts(self, mem):
        for i in range(40):
            mem.update_memory({"knowledge": {"key_facts": [f"fact {i}"]}})
        mem.compact_memory(max_facts=10)
        assert len(mem.get_memory("knowledge")["key_facts"]) == 10

    def test_trims_recent_actions(self, mem):
        for i in range(20):
            mem.record_action(f"action {i}")
        mem.compact_memory(max_actions=5)
        assert len(mem.get_memory("actions")["recent_actions"]) == 5

    def test_compress_fn_called_for_excess(self, mem):
        for i in range(15):
            mem.update_memory({"knowledge": {"key_facts": [f"fact {i}"]}})
        called_with = []

        def fake_compress(text: str) -> str:
            called_with.append(text)
            return "COMPRESSED"

        mem.compact_memory(max_facts=5, compress_fn=fake_compress)
        assert called_with, "compress_fn was not called"
        facts = mem.get_memory("knowledge")["key_facts"]
        assert facts[0].startswith("[compressed]")

    def test_preserves_summaries(self, mem):
        mem.update_memory({"knowledge": {"summaries": {"sec_1": "Important summary"}}})
        for i in range(40):
            mem.update_memory({"knowledge": {"key_facts": [f"fact {i}"]}})
        mem.compact_memory(max_facts=5)
        assert mem.get_memory("knowledge")["summaries"]["sec_1"] == "Important summary"

    def test_preserves_open_loops(self, mem):
        mem.update_memory({"progress": {"open_loops": ["Is sec 3 relevant?"]}})
        mem.compact_memory()
        assert "Is sec 3 relevant?" in mem.get_memory("progress")["open_loops"]


# ──────────────────────────────────────────────────────────────
# WorkingMemory.save_memory / load_memory
# ──────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_file(self, mem, tmp_session):
        path = mem.save_memory()
        assert path.exists()

    def test_round_trip(self, tmp_session):
        m1 = WorkingMemory(tmp_session, goal="Round-trip test")
        m1.update_memory({
            "knowledge": {"key_facts": ["fact A"]},
            "control": {"stage": "gathering"},
        })
        m1.save_memory()

        m2 = WorkingMemory(tmp_session, goal="Round-trip test")
        assert "fact A" in m2.get_memory("knowledge")["key_facts"]
        assert m2.get_memory("control")["stage"] == "gathering"

    def test_corrupt_file_falls_back(self, tmp_session):
        save_path = tmp_session / "working_memory.json"
        save_path.write_text("{not valid json!!}", encoding="utf-8")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = WorkingMemory(tmp_session, goal="Fallback test")
            assert any("corrupt" in str(warning.message).lower() for warning in w)

        assert m.get_memory("control")["status"] == "active"

    def test_empty_file_falls_back(self, tmp_session):
        save_path = tmp_session / "working_memory.json"
        save_path.write_text("", encoding="utf-8")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = WorkingMemory(tmp_session, goal="Empty file test")
        assert isinstance(m.get_memory(), dict)

    def test_save_to_custom_path(self, mem, tmp_path):
        custom = tmp_path / "subdir" / "custom.json"
        mem.save_memory(custom)
        assert custom.exists()
        data = json.loads(custom.read_text())
        assert "control" in data


# ──────────────────────────────────────────────────────────────
# WorkingMemory.to_context_string
# ──────────────────────────────────────────────────────────────

class TestContextString:
    def test_returns_string(self, mem):
        s = mem.to_context_string()
        assert isinstance(s, str)

    def test_empty_memory_message(self, tmp_session):
        # Fresh memory should still show status info, not be truly "empty"
        m = WorkingMemory(tmp_session)
        s = m.to_context_string()
        assert len(s) > 0

    def test_section_filter(self, mem):
        mem.update_memory({"knowledge": {"key_facts": ["Fact 1"]}})
        s_knowledge = mem.to_context_string(["knowledge"])
        s_control = mem.to_context_string(["control"])
        assert "Fact 1" in s_knowledge
        assert "Fact 1" not in s_control

    def test_shows_recent_actions(self, mem):
        mem.record_action("Called web_search(['query'])")
        s = mem.to_context_string(["actions"])
        assert "web_search" in s

    def test_shows_output_files(self, mem):
        mem.record_artifact("summary_report.md")
        s = mem.to_context_string(["artifacts"])
        assert "summary_report.md" in s


# ──────────────────────────────────────────────────────────────
# Tool handler functions
# ──────────────────────────────────────────────────────────────

class TestToolHandlers:
    def test_handle_get_memory_full(self, mem):
        result = handle_get_memory(mem, {})
        assert "control" in result
        assert "```json" in result

    def test_handle_get_memory_section(self, mem):
        result = handle_get_memory(mem, {"section": "task"})
        assert '"goal"' in result

    def test_handle_get_memory_invalid_section(self, mem):
        result = handle_get_memory(mem, {"section": "bogus"})
        assert "❌" in result

    def test_handle_update_memory_ok(self, mem):
        result = handle_update_memory(mem, {
            "patch": {"control": {"stage": "synthesizing"}}
        })
        assert "✅" in result
        assert mem.get_memory("control")["stage"] == "synthesizing"

    def test_handle_update_memory_non_dict_patch(self, mem):
        result = handle_update_memory(mem, {"patch": "not a dict"})
        assert "❌" in result

    def test_handle_update_memory_empty_patch(self, mem):
        result = handle_update_memory(mem, {"patch": {}})
        assert "⚠️" in result

"""
memory_usage_example.py — Minimal walkthrough of WorkingMemory.

Demonstrates:
  1. Initialising a WorkingMemory for a new task
  2. Reading memory (get_memory)
  3. Applying a patch update (update_memory)
  4. Orchestration-layer auto-recording (record_action / record_artifact)
  5. Compacting memory after many reads
  6. Saving and recovering state from disk

Run with:
    python -m BatchAgent.examples.memory_usage_example
"""

import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from BatchAgent.working_memory import WorkingMemory, handle_get_memory, handle_update_memory

# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _hr(title: str = "") -> None:
    print("\n" + "─" * 60)
    if title:
        print(f"  {title}")
    print()


# ──────────────────────────────────────────────────────────────
# Main example
# ──────────────────────────────────────────────────────────────

def run():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir)

        # ── Step 1: Initialise ────────────────────────────────
        _hr("Step 1 — Initialise WorkingMemory")
        mem = WorkingMemory(
            session_dir=session_dir,
            goal="Summarise the NVIDIA Nemotron-3 technical report",
            inputs=["data/pdftest/NVIDIA-Nemotron-3-Technical-Report.pdf"],
        )
        print(repr(mem))
        print("\nInitial control section:")
        print(json.dumps(mem.get_memory("control"), indent=2))

        # ── Step 2: Model reads memory at task start ──────────
        _hr("Step 2 — Model reads memory (get_memory tool)")
        result = handle_get_memory(mem, {})   # full snapshot
        print(result[:400], "...")

        # ── Step 3: Model updates after reading doc overview ──
        _hr("Step 3 — Model patches memory after get_document_overview")
        handle_update_memory(mem, {
            "patch": {
                "control": {
                    "stage": "gathering",
                    "current_step": "reading section 1: Introduction",
                },
                "progress": {
                    "completed_steps": ["Called get_document_overview"],
                    "pending_steps": [
                        "Read sec_1 Introduction",
                        "Read sec_2 Architecture",
                        "Read sec_3 Training",
                        "Read sec_4 Results",
                        "Write summary report",
                    ],
                },
                "knowledge": {
                    "key_facts": [
                        "Document has 15 sections and 42 pages",
                        "Paper describes Nemotron-3 model family",
                    ],
                },
            }
        })
        print("Progress after patch:")
        print(json.dumps(mem.get_memory("progress"), indent=2))

        # ── Step 4: Orchestration auto-records tool calls ─────
        _hr("Step 4 — Orchestration layer auto-records tool calls")
        # Simulating what agent_main.py does after each tool execution
        mem.record_action("T2: read_document_section(['section_id'])")
        mem.record_action("T2: read_document_section(['section_id'])")  # duplicate ignored
        mem.record_action("T3: read_document_section(['section_id'])")

        print("Recent actions:", mem.get_memory("actions")["recent_actions"])

        # ── Step 5: Model updates knowledge checkpoint ────────
        _hr("Step 5 — Model records knowledge after reading section 1")
        handle_update_memory(mem, {
            "patch": {
                "control": {"current_step": "reading section 2: Architecture"},
                "progress": {
                    "completed_steps": ["Read sec_1: Introduction"],
                    "pending_steps": ["Read sec_2 Architecture"],   # dedup — won't double-add
                },
                "knowledge": {
                    "summaries": {
                        "sec_1": (
                            "Introduction: Nemotron-3 is a family of LLMs trained on "
                            "6T tokens. Smallest model is 4B params, largest is 22B."
                        )
                    },
                    "key_facts": ["Nemotron-3 family spans 4B–22B parameters"],
                },
            }
        })

        # ── Step 6: Simulate many reads, then compact ─────────
        _hr("Step 6 — Simulate 25 more reads, then compact memory")
        for i in range(2, 28):
            mem.record_action(f"T{i}: read_document_section(sec_{i})")
            mem.update_memory({"knowledge": {"key_facts": [f"Fact from section {i}"]}})

        before = len(mem.get_memory("knowledge")["key_facts"])
        print(f"key_facts before compact: {before}")

        mem.compact_memory(max_facts=15, max_actions=8)

        after = len(mem.get_memory("knowledge")["key_facts"])
        print(f"key_facts after compact : {after}")
        print(f"recent_actions after compact: {len(mem.get_memory('actions')['recent_actions'])}")

        # ── Step 7: Orchestration records the output file ─────
        _hr("Step 7 — Orchestration records written output file")
        mem.record_artifact("summary_report.md", is_output=True)
        mem.update_memory({
            "control": {"stage": "done", "status": "done", "current_step": "finished"},
            "progress": {"completed_steps": ["Wrote summary_report.md"]},
        })

        # ── Step 8: Render context string ─────────────────────
        _hr("Step 8 — Context string injected into user feedback")
        snapshot = mem.to_context_string()
        print(snapshot)

        # ── Step 9: Save and reload ───────────────────────────
        _hr("Step 9 — Save, then load in a new instance (crash recovery)")
        saved_path = mem.save_memory()
        print(f"Saved to: {saved_path}")

        mem2 = WorkingMemory(session_dir, goal="Summarise the NVIDIA Nemotron-3 technical report")
        print(f"Recovered: {repr(mem2)}")
        print("Output files:", mem2.get_memory("artifacts")["output_files"])
        assert mem2.get_memory("control")["status"] == "done"
        assert "summary_report.md" in mem2.get_memory("artifacts")["output_files"]

        print("\n✅  All steps completed successfully.")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Test suite for LLM response extraction and patch application.

Validates extract_all_diffs, extract_write_file_actions, sanitize_diff_text,
apply_fuzzy_patch, and _try_apply_content against real LLM session data 
from .agent/sessions/.

Usage:
    python3 CodeAgent/test_patch_apply.py                    # Run all tests
    python3 CodeAgent/test_patch_apply.py --session-prefix 2026-02-17_001031  # Test specific session
    python3 CodeAgent/test_patch_apply.py --verbose          # Show all details
"""

import os
import re
import sys
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from CodeAgent.codeagent_libs import (
    extract_all_diffs,
    extract_write_file_actions,
    sanitize_diff_text,
    apply_fuzzy_patch,
    extract_files_from_diff,
    resolve_path,
    error_code_extraction,
)
from CodeAgent.mini_code_agent import AgentConfig, _try_apply_content

# ─── Mocks & Constants ───────────────────────────────────────────────────────
KNOWN_BROKEN_DIFFS = {
    # Add any known unpatchable dir names here if they should be skipped/expected to fail
}

# ─── Test Result Tracking ────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.failures: List[str] = []
    
    def record_pass(self, name: str, detail: str = ""):
        self.passed += 1
        if detail:
            print(f"  ✅ {name}: {detail}")
        else:
            print(f"  ✅ {name}")
    
    def record_fail(self, name: str, detail: str):
        self.failed += 1
        msg = f"  ❌ {name}: {detail}"
        self.failures.append(msg)
        print(msg)
    
    def record_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: {reason}")
    
    def summary(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed, {self.skipped} skipped / {total} total")
        if self.failures:
            print(f"\nFAILURES:")
            for f in self.failures:
                print(f)
        print(f"{'='*60}")
        return self.failed == 0





def test_full_pipeline_sessions(results: TestResults, session_prefix: str = "2026-02-18_", verbose: bool = False):
    """
    Full-pipeline E2E test: simulate CodeAgent's generate-then-patch flow.
    
    Iterates through all log folders (0000, 0001, 0002...) for each session:
    1. Parse response.md → try extract_write_file_actions or extract_all_diffs
    2. Write file or apply fuzzy patch
    3. Verify Python code extraction with AST
    
    This tests the REAL extraction + application pipeline with REAL LLM data.
    """
    print("\n── Full-Pipeline Session Tests ──")
    
    base_dir = Path(__file__).resolve().parent.parent
    sessions_dir = base_dir / ".agent" / "sessions"
    if not sessions_dir.exists():
        results.record_skip("FullPipeline", "no sessions dir")
        return
    
    tested = 0
    passed_count = 0
    expected_fail_count = 0
    unexpected_fail = []
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        
        for session_dir in sorted(sessions_dir.iterdir()):
            if not session_dir.name.startswith(session_prefix):
                continue
            
            test_file = tmp_dir / f"session_{session_dir.name}" / "task.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            if test_file.exists():
                test_file.unlink() # reset per session
            
            turns = sorted([d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit()])
            if not turns:
                continue
            
            session_tested = False
            session_failed = False
            initial_line_count = 0
            
            for turn_dir in turns:
                r = turn_dir / "response.md"
                if not r.exists(): continue
                
                response_text = r.read_text(encoding="utf-8", errors="ignore")
                wf_actions = extract_write_file_actions(response_text)
                diff = extract_all_diffs(response_text)
                
                acted = False
                turn_failed = False
                files_to_check = []
                
                if wf_actions:
                    for rel_path, content in wf_actions:
                        target_file = tmp_dir / f"session_{session_dir.name}" / Path(rel_path).name
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_text(content, encoding="utf-8")
                        acted = True
                        files_to_check.append(target_file)
                        initial_line_count = len(content.splitlines())
                        
                elif diff:
                    file_diffs = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
                    file_diffs = [d for d in file_diffs if d.strip().startswith('diff --git')]
                    
                    if file_diffs:
                        for single_diff in file_diffs:
                            fname_match = re.search(r'^diff --git a/\S+ b/(\S+)', single_diff, re.MULTILINE)
                            if fname_match:
                                rel_path = fname_match.group(1)
                                target_file = tmp_dir / f"session_{session_dir.name}" / Path(rel_path).name
                                target_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                if not target_file.exists():
                                    target_file.write_text("\n", encoding="utf-8")
                                    
                                ok = apply_fuzzy_patch(target_file, single_diff)
                                if not ok:
                                    if verbose:
                                        print(f"    ❌ {session_dir.name}/{turn_dir.name}: patch failed for {rel_path}")
                                    turn_failed = True
                                else:
                                    acted = True
                                    files_to_check.append(target_file)
                
                if acted and not turn_failed:
                    session_tested = True
                    # Verify grammar on all touched files
                    for target_file in files_to_check:
                        if target_file.exists() and target_file.suffix == '.py':
                            code = target_file.read_text(encoding="utf-8")
                            error_context = error_code_extraction(code)
                            if "Syntax Error detected" in error_context or "SyntaxError:" in error_context:
                                if verbose:
                                    print(f"    ❌ {session_dir.name}/{turn_dir.name}: Syntax error introduced in {target_file.name}")
                                    print(error_context)
                                turn_failed = True
                                break
                
                # Update the session's overall failure status based on the LAST turn
                # If a turn acted successfully and had no syntax error, we consider the process successful up to now
                if acted:
                    session_failed = turn_failed
            
            if session_tested:
                tested += 1
                is_known_broken = session_dir.name in KNOWN_BROKEN_DIFFS
                
                if not session_failed:
                    passed_count += 1
                    results.record_pass(f"Pipeline: {session_dir.name}")
                    if verbose:
                        result_lines = len(test_file.read_text().splitlines()) if test_file.exists() else 0
                        print(f"    ✅ {session_dir.name}: {initial_line_count}→{result_lines} lines")
                elif is_known_broken:
                    expected_fail_count += 1
                    results.record_skip(f"Pipeline: {session_dir.name}", "known broken LLM diff")
                    if verbose:
                        print(f"    ⏭️  {session_dir.name}: known broken diff (expected)")
                else:
                    unexpected_fail.append(session_dir.name)
                    results.record_fail(f"Pipeline: {session_dir.name}", "Pipeline failed at some turn")
                    if verbose:
                        print(f"    ❌ {session_dir.name}: UNEXPECTED failure")
    
    if tested == 0:
        results.record_skip("FullPipeline", "no testable sessions found")
    else:
        print(f"  Full pipeline: {passed_count}/{tested} passed, "
              f"{expected_fail_count} expected failures, "
              f"{len(unexpected_fail)} unexpected failures")
        if unexpected_fail:
            print(f"  ⚠️  Unexpected failures: {', '.join(unexpected_fail)}")

def test_error_code_extraction(results: TestResults, verbose: bool = False):
    print("\n── Error Code Extraction Tests ──")
    
    # Test 1: Syntax Error
    code_with_syntax_error = '''def my_func():
    print("Hello"
    return 1
'''
    res = error_code_extraction(code_with_syntax_error)
    if "SyntaxError:" in res and (">>    2:" in res or ">>    3:" in res) and "print(\"Hello\"" in res:
        results.record_pass("ErrorExtraction_Syntax", "Found correct syntax error line")
    else:
        results.record_fail("ErrorExtraction_Syntax", f"Failed to find syntax error context. Got:\n{res}")

    # Test 2: Valid Code
    valid_code = '''def my_func():
    print("Hello")
    return 1
'''
    res2 = error_code_extraction(valid_code)
    if "No syntax errors detected" in res2:
        results.record_pass("ErrorExtraction_Valid", "Correctly identified valid code")
    else:
        results.record_fail("ErrorExtraction_Valid", f"Falsely reported an error. Got:\n{res2}")
        
    # Test 3: Traceback parsing
    traceback_code = '''def a():
    b()
def b():
    c()
def c():
    print(1/0)
'''
    tb_msg = '''Traceback (most recent call last):
  File "test.py", line 2, in a
  File "test.py", line 4, in b
  File "test.py", line 6, in c
ZeroDivisionError: division by zero'''
    res3 = error_code_extraction(traceback_code, error_message=tb_msg)
    if ">>    2:" in res3 and ">>    4:" in res3 and ">>    6:" in res3:
        results.record_pass("ErrorExtraction_Traceback", "Found all traceback lines")
    else:
        results.record_fail("ErrorExtraction_Traceback", f"Failed to extract traceback logic. Got:\n{res3}")





def main():
    parser = argparse.ArgumentParser(description="Test LLM response extraction and patch apply")
    parser.add_argument("--session-prefix", default="2026-02-18_",
                        help="Prefix to filter session dirs (default: 2026-02-17_)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for each test")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Max sessions to test (0=all)")
    args = parser.parse_args()
    
    results = TestResults()
    base_dir = Path(__file__).resolve().parent.parent
    
    # Full-pipeline session tests
    test_full_pipeline_sessions(results, session_prefix=args.session_prefix, verbose=args.verbose)
    
    test_error_code_extraction(results, verbose=args.verbose)
    
    # Summary
    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

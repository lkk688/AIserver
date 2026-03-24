from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
except Exception:
    class _DummyConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = _DummyConsole()
    Panel = None  # type: ignore

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import json_repair
except ImportError:
    json_repair = None


# =============================================================================
# 1. Generic utilities
# =============================================================================

def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    target_chars = int(max_tokens * 3.5)
    return text[:target_chars] + "\n...[TRUNCATED]..."


def ensure_dirs(base_dir: Path):
    (base_dir / "sessions").mkdir(parents=True, exist_ok=True)
    (base_dir / "skilldb").mkdir(parents=True, exist_ok=True)
    for p in [
        base_dir / "skilldb/successes.jsonl",
        base_dir / "skilldb/failures.jsonl",
        base_dir / "runs.jsonl",
    ]:
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")


def write_jsonl(path: Path, obj: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_file(path: str, max_chars: int = 64000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[MISSING FILE] {path}"
    data = p.read_text(encoding="utf-8", errors="ignore")
    if len(data) > max_chars:
        return data[:max_chars] + "\n\n[TRUNCATED]\n"
    return data


def top_level_tree(max_items: int = 200) -> str:
    items = []
    try:
        for p in Path(".").iterdir():
            if p.name.startswith(".agent") or p.name.startswith(".git"):
                continue
            items.append(p.name + ("/" if p.is_dir() else ""))
    except Exception:
        pass
    items = sorted(items)[:max_items]
    return "\n".join(items)


# =============================================================================
# 2. Shell / process helpers
# =============================================================================

def run_shell(
    cmd: str,
    cwd: Optional[str] = None,
    cap: int = 20000,
    timeout: Optional[int] = None,
    sandbox_container: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Run a shell command with streaming output and a dynamic timeout.

    Timeout semantics
    -----------------
    * ``timeout=None``  → start with a 60 s soft deadline; every time the
      process emits a line of output the deadline is pushed forward by
      another 60 s (i.e. "alive as long as it is making progress").  A hard
      cap of 1 200 s (20 min) prevents runaway jobs.  If the process produces
      *no output at all* for 30 s straight it is killed (catches GUI windows
      blocking on ``plt.show()`` etc.).
    * ``timeout=N``     → fixed hard limit of N seconds, no auto-extension.

    Environment extras
    ------------------
    ``MPLBACKEND=Agg`` and ``MPLBACKEND`` are injected so that matplotlib
    scripts never try to open a display window.
    """
    import select

    if sandbox_container:
        cmd_to_run = (
            f'docker exec -i -w "{cwd}" {sandbox_container} /bin/bash -c {cmd!r}'
            if cwd
            else f'docker exec -i {sandbox_container} /bin/bash -c {cmd!r}'
        )
        run_cwd = None
    else:
        cmd_to_run = cmd
        run_cwd = cwd if not sandbox_container else None

    # Force non-interactive / headless matplotlib so plt.show() never blocks.
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["MPLBACKEND"] = env.get("MPLBACKEND", "Agg")   # keep if already set to something else
    env["DISPLAY"] = env.get("DISPLAY", "")

    fixed_timeout = timeout is not None
    soft_window   = 60        # extend deadline by this many seconds on each output line
    no_out_kill   = 30        # kill if no output at all for this many seconds
    hard_cap      = 1200      # absolute maximum regardless of activity
    deadline      = time.time() + (timeout if fixed_timeout else soft_window)
    hard_deadline = time.time() + hard_cap

    start_time = time.time()
    lines: List[str] = []
    code = 1

    try:
        proc = subprocess.Popen(
            cmd_to_run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=run_cwd,
            env=env,
            bufsize=1,
        )

        last_output_time = time.time()

        while True:
            now = time.time()

            # Hard cap
            if now >= hard_deadline:
                proc.kill()
                lines.append(f"\n[Error] Hard cap reached after {hard_cap}s — killed.")
                break

            # Fixed or dynamic deadline
            if fixed_timeout and now >= deadline:
                proc.kill()
                lines.append(f"\n[Error] Command timed out after {timeout}s.")
                break

            # Readable?
            r, _, _ = select.select([proc.stdout], [], [], 0.1)
            if r:
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    last_output_time = now
                    if not fixed_timeout:
                        # Push the soft deadline forward — process is alive & working
                        deadline = min(hard_deadline, now + soft_window)
                elif proc.poll() is not None:
                    # EOF — drain any remaining bytes
                    rest = proc.stdout.read()
                    if rest:
                        lines.append(rest)
                    break
            elif proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    lines.append(rest)
                break
            elif not fixed_timeout and (now - last_output_time) >= no_out_kill:
                proc.kill()
                lines.append(
                    f"\n[Error] No output for {no_out_kill}s — process killed "
                    f"(possible GUI block or infinite wait)."
                )
                break

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        code = proc.returncode if proc.returncode is not None else 1

    except Exception as exc:
        code = 1
        lines.append(f"[Error] Failed to execute command: {exc}")

    elapsed = time.time() - start_time
    out = "".join(lines)
    out += f"\n[Execution Time: {elapsed:.2f}s]"

    if len(out) > cap:
        out = out[-cap:]
    return code, out


def run_bash_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = []
        if result.stdout:
            output.append("STDOUT:\n" + result.stdout)
        if result.stderr:
            output.append("STDERR:\n" + result.stderr)
        if result.returncode != 0:
            output.insert(0, f"[Error] Command exited with code {result.returncode}")
        if not output:
            return "[Empty Output - Success]"
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return f"[Error] Command timed out after 60 seconds:\n{command}"
    except Exception as e:
        return f"[Error] Failed to run command: {e}"


def is_git_repo() -> bool:
    code, _ = run_shell("git rev-parse --is-inside-work-tree")
    return code == 0


def git_status() -> str:
    code, out = run_shell("git status -sb")
    return out if code == 0 else ""


def git_diff() -> str:
    code, out = run_shell("git diff")
    return out if code == 0 else ""


# =============================================================================
# 3. JSON robustness helpers
# =============================================================================

def robust_json_loads(json_str: str, tool_name: str = "") -> Optional[Dict[str, Any]]:
    if not json_str or not isinstance(json_str, str):
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    if json_repair is not None:
        try:
            repaired = json_repair.repair_json(json_str, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            if isinstance(repaired, list) and repaired and isinstance(repaired[0], dict):
                return repaired[0]
        except Exception:
            pass

    if tool_name == "write_file":
        logging.warning("JSON parsing failed. Attempting violent regex extraction for write_file.")
        try:
            def _extract_string_field(payload: str, field_name: str) -> Optional[str]:
                key = f'"{field_name}"'
                key_idx = payload.find(key)
                if key_idx < 0:
                    return None
                colon_idx = payload.find(":", key_idx + len(key))
                if colon_idx < 0:
                    return None
                start_quote_idx = payload.find('"', colon_idx + 1)
                if start_quote_idx < 0:
                    return None

                value_chars: List[str] = []
                escaped = False
                i = start_quote_idx + 1
                while i < len(payload):
                    ch = payload[i]
                    if escaped:
                        value_chars.append(ch)
                        escaped = False
                    elif ch == "\\":
                        value_chars.append(ch)
                        escaped = True
                    elif ch == '"':
                        return "".join(value_chars)
                    else:
                        value_chars.append(ch)
                    i += 1
                return "".join(value_chars) if value_chars else None

            path_raw = _extract_string_field(json_str, "path")
            content_raw = _extract_string_field(json_str, "content")
            if path_raw is not None and content_raw is not None:
                path = bytes(path_raw, "utf-8").decode("unicode_escape")
                try:
                    clean_content = bytes(content_raw, "utf-8").decode("unicode_escape")
                except Exception:
                    clean_content = (
                        content_raw.replace("\\n", "\n")
                        .replace('\\"', '"')
                        .replace("\\\\", "\\")
                    )
                return {"path": path, "content": clean_content}
        except Exception as e:
            logging.error(f"Violent regex extraction failed: {e}")

    try:
        clean_str = json_str.replace("\n", "\\n").replace("\r", "")
        return json.loads(clean_str)
    except Exception:
        return None


def extract_json_robust(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break

    if start >= 0:
        candidate = text[start:]
        for suffix in ["]}", "]", "}", '"]}', '"]}']:
            try:
                return json.loads(candidate + suffix)
            except Exception:
                pass

        last_quote = candidate.rfind('"')
        if last_quote > 0:
            trimmed = candidate[:last_quote + 1]
            for suffix in ["]}", "]}", "]}\n"]:
                try:
                    return json.loads(trimmed + suffix)
                except Exception:
                    pass

    m = re.search(r"\{[^{}]+\}", text)
    if m:
        candidate = m.group(0)
        fixed = re.sub(r"(\w+)\s*:", r'"\1":', candidate)
        try:
            return json.loads(fixed)
        except Exception:
            pass

    return None


# =============================================================================
# 4. File path resolution
# =============================================================================

def resolve_path(raw_path: str, allowlist: List[str], root_dir: Path = Path(".")) -> Optional[Path]:
    original = raw_path.strip().strip("'").strip('"')

    if os.path.isabs(original):
        abs_candidate = Path(original).resolve()
        if abs_candidate.exists() or abs_candidate.parent.exists():
            return abs_candidate

    clean = original.replace("\\", "/")
    clean = re.sub(r"^\./+", "", clean)
    clean = clean.lstrip("/")
    if not clean:
        return None

    root = root_dir.resolve()
    clean_parts = [part for part in Path(clean).parts if part not in ("", ".")]
    clean_parts_lower = [part.lower() for part in clean_parts]
    allowed_paths = [Path(p).expanduser().resolve() for p in allowlist if str(p).strip()]

    for allowed in allowed_paths:
        if allowed.as_posix() == Path(clean).as_posix() or str(allowed) == clean:
            return allowed

    suffix_matches: List[Path] = []
    for allowed in allowed_paths:
        allowed_parts_lower = [part.lower() for part in allowed.parts]
        if (
            len(allowed_parts_lower) >= len(clean_parts_lower)
            and allowed_parts_lower[-len(clean_parts_lower):] == clean_parts_lower
        ):
            suffix_matches.append(allowed)
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    basename = Path(clean).name.lower()
    basename_matches = [p for p in allowed_paths if p.name.lower() == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]

    candidate = (root / Path(clean)).resolve()
    if candidate.exists() or candidate.parent.exists():
        return candidate

    return None


# =============================================================================
# 5. Local observation tools
# =============================================================================

def _is_binary(file_path: Path) -> bool:
    if not file_path.exists():
        return True
    try:
        with open(file_path, "tr") as f:
            f.read(1024)
            return False
    except UnicodeDecodeError:
        return True


_SEARCH_CONTEXT_LINES = 3        # surrounding lines shown in full-context mode
_SEARCH_COMPACT_MATCHES = 8      # switch to compact when total matches exceed this
_SEARCH_COMPACT_GROUPS = 5       # or when total context windows (groups) exceed this
_SEARCH_MAX_MATCHES = 100        # hard cap on total matches returned
_SEARCH_SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env", ".venv"}


def _walk_source_files(root: Path):
    """Yield all readable, non-binary, <1 MB files under root, skipping hidden/junk dirs."""
    for root_path_str, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SEARCH_SKIP_DIRS]
        root_path = Path(root_path_str)
        for file in files:
            if file.startswith("."):
                continue
            path = root_path / file
            try:
                if path.stat().st_size > 1024 * 1024:
                    continue
            except Exception:
                continue
            if _is_binary(path):
                continue
            yield path


def _build_ast_scopes(path: Path) -> list[tuple[int, int, str]]:
    """
    For Python files: return (start_line, end_line, label) for every function/class.
    Uses an iterative walk to avoid recursion-limit issues.
    Returns [] for non-.py files or parse errors.
    """
    if path.suffix != ".py":
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return []

    scopes: list[tuple[int, int, str]] = []
    # stack items: (ast_node, qualified_name_prefix)
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qname = f"{prefix}{node.name}" if prefix else node.name
            end = getattr(node, "end_lineno", node.lineno)
            scopes.append((node.lineno, end, f"def {qname}"))
            for child in ast.iter_child_nodes(node):
                stack.append((child, f"{node.name}."))
        elif isinstance(node, ast.ClassDef):
            qname = f"{prefix}{node.name}" if prefix else node.name
            end = getattr(node, "end_lineno", node.lineno)
            scopes.append((node.lineno, end, f"class {qname}"))
            for child in ast.iter_child_nodes(node):
                stack.append((child, f"{node.name}."))
        else:
            for child in ast.iter_child_nodes(node):
                stack.append((child, prefix))
    return scopes


def _scope_at_line(scopes: list[tuple[int, int, str]], line_no: int) -> str:
    """Return the innermost scope label containing line_no, or 'module level'."""
    best: str | None = None
    best_size = float("inf")
    for start, end, name in scopes:
        if start <= line_no <= end:
            size = end - start
            if size < best_size:
                best_size = size
                best = name
    return best if best is not None else "module level"


def _merge_context_windows(
    match_lines: list[int], total_lines: int, ctx: int
) -> list[tuple[int, int]]:
    """Merge overlapping ±ctx windows around each match into sorted non-overlapping spans."""
    windows: list[tuple[int, int]] = []
    for ln in sorted(match_lines):
        lo = max(1, ln - ctx)
        hi = min(total_lines, ln + ctx)
        if windows and lo <= windows[-1][1] + 1:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    return windows


def _format_file_exact_matches(
    path: Path,
    root: Path,
    match_nos: list[int],
    file_lines: list[str],
    scopes: list[tuple[int, int, str]],
    compact: bool,
    ctx: int = _SEARCH_CONTEXT_LINES,
) -> list[str]:
    """Format matching lines for one file in either compact or full-context style."""
    rel = str(path.relative_to(root))
    n = len(match_nos)
    label = f"{'es' if n != 1 else ''}"
    output: list[str] = [f"\n=== {rel} ({n} match{label}) ==="]

    if compact:
        for ln in match_nos:
            scope = _scope_at_line(scopes, ln)
            output.append(f"  {ln}: {file_lines[ln - 1].rstrip()}  [in: {scope}]")
        return output

    # Full-context mode: show ±ctx lines, merge overlapping windows
    total = len(file_lines)
    pad = len(str(total))
    match_set = set(match_nos)
    windows = _merge_context_windows(match_nos, total, ctx)

    prev_scope: str | None = None
    for wi, (lo, hi) in enumerate(windows):
        if wi > 0:
            output.append("  --")
        # Scope banner for the first match inside this window
        first_match = next(ln for ln in match_nos if lo <= ln <= hi)
        scope = _scope_at_line(scopes, first_match)
        if scope != prev_scope:
            output.append(f"  [in: {scope}]")
            prev_scope = scope
        for ln in range(lo, hi + 1):
            sep = ":" if ln in match_set else "-"
            output.append(f"  {ln:{pad}}{sep} {file_lines[ln - 1].rstrip()}")

    return output


def _search_code_for_keyword(keyword: str, root: Path, limit: int = 30) -> list[str]:
    """Return up to `limit` compact matching lines for a single keyword (used by fallback)."""
    results: list[str] = []
    for path in _walk_source_files(root):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, start=1):
                    if keyword in line:
                        results.append(f"{path.relative_to(root)}:{i}: {line.rstrip()}")
                        if len(results) >= limit:
                            return results
        except Exception:
            pass
    return results


def search_code(query: str, root_dir: str = ".") -> str:
    root = Path(root_dir).resolve()

    # ── Phase 1: collect exact matches per file ────────────────────────────
    file_results: list[tuple[Path, list[int], list[str]]] = []
    total_matches = 0
    truncated = False

    for path in _walk_source_files(root):
        try:
            file_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        match_nos = [i + 1 for i, line in enumerate(file_lines) if query in line]
        if not match_nos:
            continue
        # Per-file cap: never show more than _SEARCH_MAX_MATCHES total
        remaining = _SEARCH_MAX_MATCHES - total_matches
        if remaining <= 0:
            truncated = True
            break
        if len(match_nos) > remaining:
            match_nos = match_nos[:remaining]
            truncated = True
        file_results.append((path, match_nos, file_lines))
        total_matches += len(match_nos)

    # ── No exact match → keyword fallback ─────────────────────────────────
    if not file_results:
        keywords = [kw for kw in query.split() if kw]
        if len(keywords) <= 1:
            return "No matches found."
        sections = [f'No exact matches found for "{query}". Showing per-keyword results:\n']
        any_hit = False
        for kw in keywords:
            kw_results = _search_code_for_keyword(kw, root, limit=30)
            if kw_results:
                any_hit = True
                cap = len(kw_results) >= 30
                sections.append(
                    f'--- keyword: "{kw}" ({len(kw_results)}{"+" if cap else ""} matches) ---'
                )
                sections.extend(kw_results)
                if cap:
                    sections.append(f'  (first 30 shown for "{kw}"; refine query to see more)')
            else:
                sections.append(f'--- keyword: "{kw}" (0 matches) ---')
        return "\n".join(sections) if any_hit else "No matches found."

    # ── Phase 2: decide compact vs full-context ────────────────────────────
    total_groups = sum(
        len(_merge_context_windows(mnos, len(lines), _SEARCH_CONTEXT_LINES))
        for _, mnos, lines in file_results
    )
    compact = total_matches > _SEARCH_COMPACT_MATCHES or total_groups > _SEARCH_COMPACT_GROUPS

    # ── Phase 3: render ────────────────────────────────────────────────────
    view = "compact" if compact else "full context"
    file_count = len(file_results)
    header = f'Found {total_matches} match(es) in {file_count} file(s) [{view}]:'
    if truncated:
        header += f'  [truncated at {_SEARCH_MAX_MATCHES} matches — refine your query]'
    sections = [header]

    for path, match_nos, file_lines in file_results:
        scopes = _build_ast_scopes(path)
        sections.extend(
            _format_file_exact_matches(path, root, match_nos, file_lines, scopes, compact)
        )

    return "\n".join(sections)


def find_file(filename_pattern: str, root_dir: str = ".") -> str:
    results = []
    root = Path(root_dir).resolve()
    pattern = filename_pattern.lower()

    for root_path_str, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".")
            and d not in ("__pycache__", "node_modules", "site-packages", "venv", "env", ".venv")
        ]

        root_path = Path(root_path_str)
        for file in files:
            if file.startswith("."):
                continue

            if fnmatch.fnmatch(file.lower(), f"*{pattern}*"):
                path = root_path / file
                results.append(str(path.relative_to(root)))

    if not results:
        return "No matching files found."
    return "\n".join(results)


def read_file_chunk(filepath: str, start_line: int = 1, end_line: int = 200) -> str:
    try:
        raw_path = Path(filepath.strip()).expanduser()
        cwd = Path.cwd()

        candidate_paths: List[Path] = []
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append(cwd / raw_path)
            candidate_paths.append(raw_path)

        file_path: Optional[Path] = None
        for candidate in candidate_paths:
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_file():
                file_path = resolved
                break

        if file_path is None and raw_path.name:
            suffix_parts = [part.lower() for part in raw_path.parts if part not in (".", "")]
            matches: List[Path] = []
            for match in cwd.rglob(raw_path.name):
                if not match.is_file():
                    continue
                rel_parts = [part.lower() for part in match.relative_to(cwd).parts]
                if len(rel_parts) >= len(suffix_parts) and rel_parts[-len(suffix_parts):] == suffix_parts:
                    matches.append(match.resolve())

            if not matches:
                for match in cwd.rglob("*"):
                    if not match.is_file():
                        continue
                    if match.name.lower() != raw_path.name.lower():
                        continue
                    rel_parts = [part.lower() for part in match.relative_to(cwd).parts]
                    if len(rel_parts) >= len(suffix_parts) and rel_parts[-len(suffix_parts):] == suffix_parts:
                        matches.append(match.resolve())

            unique_matches = list(dict.fromkeys(matches))
            if len(unique_matches) == 1:
                file_path = unique_matches[0]

        if file_path is None:
            return f"[Error] File not found: {filepath}"

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if start_line < 1:
            start_line = 1
        if end_line > len(lines):
            end_line = len(lines)

        if start_line > end_line:
            return f"[Error] Invalid range: start_line ({start_line}) > end_line ({end_line})"

        result = [f"--- {file_path} (Lines {start_line}-{end_line}) ---"]
        for i in range(start_line - 1, end_line):
            result.append(f"{i + 1:4d} | {lines[i].rstrip()}")

        return "\n".join(result)

    except Exception as e:
        return f"[Error] Could not read file {filepath}: {e}"


view_file_content = read_file_chunk


def list_directory(dir_path: str = ".") -> str:
    """
    Return a compact recursive tree of *dir_path* so the model can see the
    full workspace layout in a single call.

    Design choices
    --------------
    * Depth up to 3 levels (enough to see sub-packages without drowning in
      deep build artefacts).
    * At most 10 files shown per directory; a "… N more" line summarises the
      rest so large output directories stay readable.
    * Noisy system directories (.git, __pycache__, node_modules, .venv, …)
      are skipped entirely.
    * File sizes are shown for quick assessment (KB / MB).
    """
    _MAX_DEPTH       = 3
    _MAX_FILES_DIR   = 10
    _SKIP_DIRS: Set[str] = {
        ".git", "__pycache__", ".mypy_cache", ".pytest_cache",
        "node_modules", ".venv", "venv", ".tox", "dist", "build",
        ".eggs", "*.egg-info",
    }

    try:
        target_dir = Path(dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return f"[Error] Directory not found or not a directory: {dir_path}"

        lines: List[str] = [str(target_dir) + "/"]

        def _size_str(p: Path) -> str:
            try:
                sz = p.stat().st_size
                if sz >= 1_048_576:
                    return f"  ({sz / 1_048_576:.1f} MB)"
                if sz >= 1_024:
                    return f"  ({sz / 1_024:.0f} KB)"
                return f"  ({sz} B)"
            except OSError:
                return ""

        def _render(path: Path, prefix: str, depth: int) -> None:
            if depth > _MAX_DEPTH:
                return
            try:
                entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            except PermissionError:
                return

            dirs  = [e for e in entries if e.is_dir()  and e.name not in _SKIP_DIRS
                     and not any(e.name.endswith(s.lstrip("*")) for s in _SKIP_DIRS if "*" in s)]
            files = [e for e in entries if e.is_file()]

            shown_files   = files[:_MAX_FILES_DIR]
            hidden_count  = len(files) - len(shown_files)
            all_shown     = dirs + shown_files

            for i, entry in enumerate(all_shown):
                is_last = (i == len(all_shown) - 1) and (hidden_count == 0)
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")

                if entry.is_dir():
                    lines.append(f"{prefix}{connector}📁 {entry.name}/")
                    _render(entry, child_prefix, depth + 1)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}{_size_str(entry)}")

            if hidden_count > 0:
                lines.append(f"{prefix}└── … {hidden_count} more file(s)")

        _render(target_dir, "", 0)
        return "\n".join(lines)

    except Exception as exc:
        return f"[Error] Failed to list directory {dir_path}: {exc}"


# =============================================================================
# 6. Missing module auto-fix
# =============================================================================

def _handle_missing_modules(error_output: str) -> Optional[str]:
    match = re.search(r"ModuleNotFoundError: No module named '(.+?)'", error_output)
    if not match:
        match = re.search(r"ImportError: No module named '(.+?)'", error_output)
    if not match:
        return None

    module_name = match.group(1)
    package_map = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "cv2": "opencv-python",
        "yaml": "PyYAML",
        "bs4": "beautifulsoup4",
        "dotenv": "python-dotenv",
        "dateutil": "python-dateutil",
    }
    package_name = package_map.get(module_name, module_name)

    console.print(f"[yellow]Detected missing module: '{module_name}'. Attempting auto-install of '{package_name}'...[/yellow]")
    cmd = f"pip install --no-input {package_name}"
    code, out = run_shell(cmd)
    log = f"\n[Auto-Install: {cmd}]\nExit Code: {code}\nOutput:\n{out}\n"

    if code == 0:
        console.print(f"[green]Successfully installed '{package_name}'.[/green]")
    else:
        console.print(f"[red]Failed to install '{package_name}'.[/red]")

    return log


# =============================================================================
# 7. Context / model helpers
# =============================================================================

def query_model_context_length(client, model_name: str) -> int:
    try:
        models = client.models.list()
        for m in models.data:
            if m.id == model_name:
                ctx = getattr(m, "max_model_len", 0)
                if ctx and ctx > 0:
                    console.print(f"[green]Auto-detected model context length: {ctx}[/green]")
                    return int(ctx)
        console.print(f"[yellow]Model '{model_name}' not found in /v1/models. Using fallback.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not query model context length: {e}. Using fallback.[/yellow]")
    return 0


def compute_safe_max_tokens(
    input_tokens: int,
    model_max_context: int,
    desired_max_output: int,
    safety_margin: int = 1000,
    min_output: int = 1024,
) -> int:
    adjusted_input = int(input_tokens * 1.1)
    available = model_max_context - adjusted_input - safety_margin
    if available < min_output:
        console.print(
            f"[red]Context budget very tight: {available} tokens available "
            f"(est_input={input_tokens} -> {adjusted_input}, limit={model_max_context}). "
            f"Returning available tokens to avoid exceeding context window.[/red]"
        )
        return max(1, available)
    return min(desired_max_output, available)


def compress_messages(messages: List[Dict[str, str]], max_allowed_tokens: int) -> List[Dict[str, str]]:
    import copy
    msgs = copy.deepcopy(messages)

    while True:
        current_tokens = sum(estimate_tokens(m.get("content", "")) for m in msgs)
        if current_tokens <= max_allowed_tokens:
            break

        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(msgs):
            if i in (0, 1):
                continue
            content_len = len(m.get("content", ""))
            if content_len > longest_len:
                longest_len = content_len
                longest_idx = i

        if longest_idx == -1:
            for i, m in enumerate(msgs):
                if i == 0:
                    continue
                content_len = len(m.get("content", ""))
                if content_len > longest_len:
                    longest_len = content_len
                    longest_idx = i

        if longest_idx == -1 or longest_len < 400:
            break

        content = msgs[longest_idx]["content"]
        keep_chars = int(longest_len * 0.45)
        if keep_chars * 2 + 35 >= longest_len:
            break

        msgs[longest_idx]["content"] = (
            content[:keep_chars]
            + "\n...[TRUNCATED TO FIT CONTEXT]...\n"
            + content[-keep_chars:]
        )

    return msgs


# =============================================================================
# 8. AST / debug helpers
# =============================================================================

def _generate_ast_map_from_string(source: str) -> str:
    try:
        tree = ast.parse(source)
        lines = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        args = [arg.arg for arg in sub_node.args.args]
                        lines.append(f"    def {sub_node.name}({', '.join(args)}): ...")
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                lines.append(f"def {node.name}({', '.join(args)}): ...")
        if not lines:
            return "# No top-level classes or functions found."
        return "\n".join(lines)
    except SyntaxError as e:
        return f"# [Warning] SyntaxError in file, cannot generate AST: {e}"
    except Exception as e:
        return f"# [Warning] Could not generate file map: {e}"


def _generate_ast_map_from_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        return _generate_ast_map_from_string(source)
    except Exception as e:
        return f"# [Warning] Could not read file for AST: {e}"


def _extract_snippets_from_string(source: str, error_lines: Set[int], window: int) -> str:
    try:
        source_lines = source.splitlines()
        total_lines = len(source_lines)
        windows = []

        for eline in sorted(error_lines):
            start = max(1, eline - window)
            end = min(total_lines, eline + window)
            windows.append([start, end, {eline}])

        merged_windows = []
        for w in sorted(windows, key=lambda x: x[0]):
            if not merged_windows:
                merged_windows.append(w)
            else:
                prev = merged_windows[-1]
                if w[0] <= prev[1] + 1:
                    prev[1] = max(prev[1], w[1])
                    prev[2].update(w[2])
                else:
                    merged_windows.append(w)

        snippets = []
        for start, end, elines in merged_windows:
            snippet_lines = [f"# --- Snippet from line {start} to {end} ---"]
            for i in range(start, end + 1):
                idx = i - 1
                if 0 <= idx < len(source_lines):
                    line_content = source_lines[idx]
                    marker = ">> " if i in elines else "   "
                    snippet_lines.append(f"{marker}{i:4d}: {line_content}")
            snippets.append("\n".join(snippet_lines))

        return "\n\n".join(snippets)
    except Exception as e:
        return f"# [Warning] Could not extract snippets: {e}"


def error_code_extraction(source_code: str, error_message: str = "") -> str:
    error_lines = set()

    if error_message:
        matches = re.finditer(r"line (\d+)", error_message)
        for match in matches:
            error_lines.add(int(match.group(1)))

    try:
        ast.parse(source_code)
    except SyntaxError as e:
        if e.lineno:
            error_lines.add(e.lineno)
            if not error_message:
                error_message = f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception:
        pass

    if not error_lines:
        if error_message:
            return f"Error: {error_message}\n(No specific line numbers found in traceback)"
        return "No syntax errors detected."

    result = []
    if error_message:
        result.append(f"Error Information:\n{error_message}\n")
    else:
        result.append("Error Information:\nSyntax Error detected.\n")
    result.append("Code Context:")
    result.append(_extract_snippets_from_string(source_code, error_lines, window=5))
    return "\n".join(result)


def build_debug_prompt(traceback_str: str, window_size: int = 15, root_dir: str = ".") -> str:
    tb_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+)')
    error_locations: Dict[str, Set[int]] = {}

    for match in tb_pattern.finditer(traceback_str):
        filepath = match.group(1)
        line_num = int(match.group(2))

        abs_path = os.path.abspath(filepath)
        abs_root = os.path.abspath(root_dir)
        if not abs_path.startswith(abs_root) or "site-packages" in abs_path:
            continue

        if os.path.exists(filepath) and os.path.isfile(filepath):
            error_locations.setdefault(filepath, set()).add(line_num)

    if not error_locations:
        return f"## Error Output\n```text\n{traceback_str[-2000:]}\n```\n"

    prompt_parts = []
    prompt_parts.append("## Error Traceback\n```text\n" + traceback_str[-2000:].strip() + "\n```\n")

    for filepath, lines in error_locations.items():
        rel_path = os.path.relpath(filepath, root_dir)
        prompt_parts.append(f"## Context for `{rel_path}`\n")
        try:
            ast_map = _generate_ast_map_from_file(filepath)
            if ast_map:
                prompt_parts.append("### File Map (Structure)\n```python\n" + ast_map + "\n```\n")

            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            snippets = _extract_snippets_from_string(source, lines, window_size)
            prompt_parts.append("### Error Context Snippets\n```python\n" + snippets + "\n```\n")
        except Exception as e:
            prompt_parts.append(f"> [Warning] Could not extract detailed AST/snippets: {e}\n")

    return "\n".join(prompt_parts)


# =============================================================================
# 9. Verification / lint helpers
# =============================================================================

def run_linter(files: List[str]) -> Optional[str]:
    py_files = [str(f) for f in files if str(f).endswith(".py")]
    if not py_files:
        return None

    cmd = f"ruff check --select=E9,F821,F823 --output-format=text {' '.join(py_files)}"
    code, out = run_shell(cmd)

    if code != 0:
        return f"STATIC ANALYSIS FAILED (Ruff):\n{out}\n(Fix these syntax/name errors first!)"
    return None


def _determine_verify_cmd(
    allowlist: List[str],
    modified_files: List[str],
    auto_verify_cmd: Optional[str],
    config: Any,
) -> str:
    candidate = auto_verify_cmd

    if not candidate:
        py_files = [str(f) for f in modified_files if str(f).endswith(".py")]
        if py_files:
            candidate = f"python3 {py_files[0]}"

    if not candidate:
        py_files = [str(f) for f in allowlist if str(f).endswith(".py")]
        if py_files:
            candidate = f"python3 {py_files[0]}"

    if hasattr(config, "require_approval"):
        return candidate or ""

    auto_approve = getattr(config, "auto_approve", True)
    if not auto_approve:
        try:
            from rich.prompt import Prompt, Confirm
            if Confirm.ask("Run verification?", default=True):
                return Prompt.ask("Command", default=candidate or "").strip()
            return ""
        except Exception:
            return candidate or ""

    return candidate or ""


# =============================================================================
# 10. Skill DB helpers
# =============================================================================

@dataclass
class Skill:
    category: str
    pattern: str
    insight: str
    evidence: str
    count: int = 1
    created_at: str = ""


def load_skills(skill_dir: Path) -> List[Skill]:
    skills: List[Skill] = []
    if not skill_dir.exists():
        return []

    for path in skill_dir.glob("*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "insight" not in obj:
                        skills.append(
                            Skill(
                                category="Legacy",
                                pattern=obj.get("pattern", "general"),
                                insight=f"Legacy {obj.get('kind', 'unknown')}: {obj.get('text', '')[:100]}...",
                                evidence=obj.get("evidence", "")[:200],
                                created_at=obj.get("created_at", ""),
                            )
                        )
                    else:
                        skills.append(
                            Skill(
                                category=obj.get("category", "Uncategorized"),
                                pattern=obj.get("pattern", ""),
                                insight=obj.get("insight", ""),
                                evidence=obj.get("evidence", ""),
                                count=obj.get("count", 1),
                                created_at=obj.get("created_at", ""),
                            )
                        )
                except Exception:
                    continue
        except Exception:
            pass
    return skills


def score_skill(skill: Skill, query: str) -> int:
    q = query.lower()
    s = 0

    patt = (skill.pattern or "").lower().strip()
    if patt and patt in q:
        s += 5

    words = re.findall(r"[a-zA-Z0-9_]{3,}", (skill.insight or "").lower())
    hits = 0
    for w in set(words):
        if w in q:
            hits += 1
    s += min(hits, 5)

    return s


def select_relevant_skills(goal_and_notes: str, skill_dir: Path, topk: int = 6) -> List[Skill]:
    skills = load_skills(skill_dir)
    scored = [(score_skill(sk, goal_and_notes), sk) for sk in skills]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sk for sc, sk in scored if sc >= 2][:topk]


def format_skill_injection(skills: List[Skill]) -> str:
    if not skills:
        return ""

    by_cat: Dict[str, List[Skill]] = {}
    for sk in skills:
        by_cat.setdefault(sk.category, []).append(sk)

    lines = ["## Teacher Guidelines (From Experience)"]
    for cat, sk_list in by_cat.items():
        if cat == "Legacy":
            continue
        lines.append(f"### {cat}")
        for sk in sk_list:
            lines.append(f"- [{sk.pattern}] {sk.insight}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines).strip() + "\n"


def detect_tech_stack(goal: str, allowlist: List[str], skill_teacher_path: Path = None) -> str:
    if not skill_teacher_path or not skill_teacher_path.exists():
        return ""

    goal_lower = goal.lower()
    combined_text = goal_lower + " " + " ".join(str(x).lower() for x in allowlist)
    guidelines = []

    try:
        with open(skill_teacher_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    triggers = entry.get("triggers", [])
                    if any(t.lower() in combined_text for t in triggers):
                        header = entry.get("header")
                        if header:
                            guidelines.append(header)
                        guidelines.extend(entry.get("guidelines", []))
                        guidelines.append("")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        console.print(f"[yellow]Failed to load teacher guidelines: {e}[/yellow]")
        return ""

    return "\n".join(guidelines).strip() if guidelines else ""


# =============================================================================
# 11. WRITE_FILE extraction
# =============================================================================

def extract_write_file_actions_v2(text: str) -> List[Tuple[str, str]]:
    results = []
    pattern = re.compile(
        r'(?:^|\n)(?!\-).*?WRITE_FILE:\s*(\S+).*?\n'
        r'\s*<<<CONTENT\n'
        r'(.*?)'
        r'(?:CONTENT>{2,3}|<<<CONTENT\s*$|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|(?=\n\#\#\s)|(?=\n```)|$)',
        re.DOTALL,
    )

    for m in pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)

        for strip_tag in ["CONTENT>>>", "<<<CONTENT"]:
            if strip_tag in content:
                content = content.replace(strip_tag, "")

        xml_artifact_pattern = r'(?:\s*</[a-zA-Z0-9_]+>)+\s*$'
        content = re.sub(xml_artifact_pattern, "", content)

        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue

        if len(content.strip()) < 15:
            continue

        results.append((filepath, content.rstrip() + "\n"))

    return results


def extract_write_file_actions(text: str) -> List[Tuple[str, str]]:
    results = []
    pattern = re.compile(
        r'(?:^|\n)(?!\-).*?WRITE_FILE:\s*(\S+).*?\n'
        r'\s*<<<CONTENT\n'
        r'(.*?)'
        r'(?:CONTENT>{2,3}|<<<CONTENT\s*$|(?=\n.*?WRITE_FILE:)|(?=\ndiff --git)|(?=\n\#\#\s)|(?=\n```)|$)',
        re.DOTALL,
    )

    for m in pattern.finditer(text):
        filepath = m.group(1).strip()
        content = m.group(2)

        for strip_tag in ["CONTENT>>>", "<<<CONTENT"]:
            if strip_tag in content:
                content = content.replace(strip_tag, "")

        if filepath.startswith("a/") or filepath.startswith("b/") or filepath == "/dev/null":
            continue

        if len(content.strip()) < 15:
            continue

        results.append((filepath, content))

    return results


# =============================================================================
# 12. Diff sanitization / extraction
# =============================================================================

def sanitize_diff_text(diff_text: str) -> str:
    lines = diff_text.split("\n")
    cleaned = []

    current_file_a = None
    current_file_b = None
    seen_header_a = False
    seen_header_b = False

    for line in lines:
        stripped = line.strip()

        if re.match(r"^```", stripped):
            if not (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                continue

        if re.match(r"^</?(?:details|summary|br|hr)", stripped, re.IGNORECASE):
            if not (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                continue

        if line.startswith("diff --git"):
            m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
            if m:
                current_file_a = m.group(1)
                current_file_b = m.group(2)
            else:
                current_file_a = None
                current_file_b = None
            seen_header_a = False
            seen_header_b = False
            cleaned.append(line)
            continue

        if line.startswith("index "):
            continue

        if line.startswith("--- "):
            seen_header_a = True
            cleaned.append(line)
            continue

        if line.startswith("+++ "):
            if not seen_header_a and current_file_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
            seen_header_b = True
            cleaned.append(line)
            continue

        if line.startswith("@@ ") and current_file_a and current_file_b:
            if not seen_header_a:
                cleaned.append(f"--- a/{current_file_a}")
                seen_header_a = True
            if not seen_header_b:
                cleaned.append(f"+++ b/{current_file_b}")
                seen_header_b = True

        cleaned.append(line)

    result = "\n".join(cleaned)
    if not result.endswith("\n"):
        result += "\n"
    return result


def extract_all_diffs(text: str) -> Optional[str]:
    t = text.strip()

    t = re.sub(
        r"^(diff --git [^\n]+)\n\s*```(?:diff|python|python3)?\s*\n",
        r"\1\n",
        t,
        flags=re.MULTILINE,
    )

    fenced_diffs = []
    fence_pattern = re.compile(r"```(?:diff)?\s*\n(.*?)^```\s*$", re.DOTALL | re.MULTILINE)
    for m in fence_pattern.finditer(t):
        block = m.group(1).strip()
        if "diff --git" in block:
            fenced_diffs.append(block)

    if fenced_diffs:
        return sanitize_diff_text(fenced_diffs[-1])

    parts = re.split(r"(?=^diff --git )", t, flags=re.MULTILINE)
    raw_diffs = []
    for part in parts:
        part = part.strip()
        if part.startswith("diff --git"):
            diff_lines = []
            for line in part.split("\n"):
                if (
                    line.startswith("diff --git")
                    or line.startswith("---")
                    or line.startswith("+++")
                    or line.startswith("@@")
                    or line.startswith("+")
                    or line.startswith("-")
                    or line.startswith(" ")
                    or line.startswith("\\")
                    or line.startswith("index ")
                    or line.startswith("new file")
                    or line.startswith("old mode")
                    or line.startswith("new mode")
                    or line.startswith("deleted file")
                    or line.startswith("similarity")
                    or line.startswith("rename")
                    or line == ""
                ):
                    diff_lines.append(line)
                else:
                    break
            if diff_lines:
                raw_diffs.append("\n".join(diff_lines))

    if raw_diffs:
        return sanitize_diff_text(raw_diffs[-1])

    return None


# =============================================================================
# 13. Patch / write application
# =============================================================================

def apply_patch_guarded(diff_text: str, turn_dir: Path, auto_approve: bool = False) -> bool:
    patch_path = turn_dir / "patch.diff"
    diff_text = sanitize_diff_text(diff_text)
    patch_path.write_text(diff_text, encoding="utf-8")

    for m in re.finditer(r"^\+\+\+ b/(.+)$", diff_text, re.MULTILINE):
        fpath = Path(m.group(1))
        fpath.parent.mkdir(parents=True, exist_ok=True)

    apply_log_parts = []

    def try_apply(patch_file: Path, label: str) -> bool:
        strategies = [
            f"git apply --check --recount {patch_file.as_posix()}",
            f"git apply --check {patch_file.as_posix()}",
        ]
        for cmd_check in strategies:
            check_code, check_out = run_shell(cmd_check)
            apply_log_parts.append(f"[{cmd_check}] exit={check_code}\n{check_out}\n")

            if check_code == 0:
                cmd_apply = cmd_check.replace("--check ", "")
                app_code, app_out = run_shell(cmd_apply)
                apply_log_parts.append(f"[{cmd_apply}] exit={app_code}\n{app_out}\n")

                if app_code == 0:
                    console.print(f"[green]Patch applied ({label}).[/green]")
                    return True
                console.print(f"[yellow]Apply failed after check passed ({label}): {app_out[:200]}[/yellow]")
        return False

    success = try_apply(patch_path, "combined")

    if not success:
        individual_diffs = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
        individual_diffs = [d for d in individual_diffs if d.strip().startswith("diff --git")]

        if len(individual_diffs) > 1:
            console.print(f"[yellow]Combined patch failed. Trying {len(individual_diffs)} individual patches...[/yellow]")
            all_ok = True
            for idx, single_diff in enumerate(individual_diffs):
                single_path = turn_dir / f"patch_part{idx}.diff"
                single_path.write_text(sanitize_diff_text(single_diff), encoding="utf-8")
                if not try_apply(single_path, f"part {idx + 1}/{len(individual_diffs)}"):
                    all_ok = False
                    fname_m = re.search(r"diff --git a/(\S+)", single_diff)
                    fname = fname_m.group(1) if fname_m else f"part {idx + 1}"
                    console.print(f"[red]Individual patch for {fname} also failed.[/red]")
            success = all_ok

    (turn_dir / "apply.log").write_text("\n".join(apply_log_parts), encoding="utf-8")

    if not success and apply_log_parts and Panel is not None:
        console.print(Panel(
            apply_log_parts[-1][:500],
            title="Patch check failed",
            style="red",
        ))

    return success


def apply_fuzzy_patch(file_path: Path, diff_content: str, log_buffer: list = None) -> bool:
    def log(msg: str):
        if log_buffer is not None:
            log_buffer.append(msg)

    if "new file mode" in diff_content or "--- /dev/null" in diff_content:
        new_content = []
        for line in diff_content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                new_content.append(line[1:])
        if new_content:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("\n".join(new_content) + "\n", encoding="utf-8")
            msg = f"[green]Created new file from diff: {file_path}[/green]"
            console.print(msg)
            log(msg)
            return True
        log(f"New file creation failed: content empty for {file_path}")
        return False

    if not file_path.exists():
        msg = f"[red]Target file {file_path} not found for diff.[/red]"
        console.print(msg)
        log(msg)
        return False

    original_text = file_path.read_text(encoding="utf-8")
    had_trailing_newline = original_text.endswith("\n")
    original_lines = original_text.splitlines()
    modified_lines = list(original_lines)

    hunks = re.split(r"^@@\s.*?\s@@.*$", diff_content, flags=re.MULTILINE)
    hunks = hunks[1:]
    if not hunks:
        msg = "[yellow]No hunks found in diff.[/yellow]"
        console.print(msg)
        log(msg)
        return False

    applied_hunks = 0

    for hunk in hunks:
        hunk_lines = hunk.splitlines()
        if hunk_lines and hunk_lines[0] == "":
            hunk_lines = hunk_lines[1:]
        if not hunk_lines:
            continue

        search_block = []
        replace_block = []

        for line in hunk_lines:
            if line.startswith(" "):
                search_block.append(line[1:])
                replace_block.append(line[1:])
            elif line.startswith("-"):
                search_block.append(line[1:])
            elif line.startswith("+"):
                replace_block.append(line[1:])
            elif line.startswith("\\"):
                pass
            elif line == "":
                search_block.append("")
                replace_block.append("")
            else:
                search_block.append(line)
                replace_block.append(line)

        if not search_block:
            if replace_block:
                for i, rl in enumerate(replace_block):
                    modified_lines.insert(i, rl)
                msg = f"[green]Applied pure-addition hunk ({len(replace_block)} lines)[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
            continue

        if replace_block:
            replace_stripped = [l.strip() for l in replace_block]
            n_replace = len(replace_block)
            found_already = -1
            for i in range(len(modified_lines) - n_replace + 1):
                file_subset = modified_lines[i:i + n_replace]
                if [l.strip() for l in file_subset] == replace_stripped:
                    found_already = i
                    break
            if found_already != -1:
                msg = f"[green]Hunk already applied (found replacement at line {found_already + 1})[/green]"
                console.print(msg)
                log(msg)
                applied_hunks += 1
                continue

        match_index = -1
        n_search = len(search_block)
        n_delete = n_search

        for i in range(len(modified_lines) - n_search + 1):
            if modified_lines[i:i + n_search] == search_block:
                match_index = i
                log(f"Strategy A (Exact) match at line {i + 1}")
                break

        if match_index == -1:
            search_stripped = [l.strip() for l in search_block]
            for i in range(len(modified_lines) - n_search + 1):
                file_subset = modified_lines[i:i + n_search]
                file_stripped = [l.strip() for l in file_subset]
                if file_stripped == search_stripped:
                    match_index = i
                    msg = f"[green]Fuzzy-matched hunk at line {match_index + 1} (whitespace-insensitive)[/green]"
                    console.print(msg)
                    log(msg)
                    break

        if match_index == -1 and n_search >= 2:
            anchors = [(idx, l) for idx, l in enumerate(search_block) if l.strip()]
            if len(anchors) >= 2:
                first_idx, first_line = anchors[0]
                last_idx, last_line = anchors[-1]
                expected_span = last_idx - first_idx
                first_stripped = first_line.strip()
                last_stripped = last_line.strip()

                for i in range(len(modified_lines)):
                    if modified_lines[i].strip() != first_stripped:
                        continue

                    max_end = min(len(modified_lines), i + expected_span * 2 + 2)
                    found_last = -1
                    for j in range(i + 1, max_end):
                        if modified_lines[j].strip() == last_stripped:
                            found_last = j
                            break

                    if found_last != -1:
                        match_index = i - first_idx
                        lines_after_last_anchor = len(search_block) - last_idx - 1
                        actual_end = found_last + lines_after_last_anchor + 1
                        n_delete = actual_end - match_index
                        msg = (
                            f"[cyan]Anchor-matched hunk at line {match_index + 1} "
                            f"(anchors at {match_index + 1}..{actual_end}, deleting {n_delete} lines)[/cyan]"
                        )
                        console.print(msg)
                        log(msg)
                        break

        if match_index == -1 and n_search >= 4:
            search_stripped = [l.strip() for l in search_block]
            best_ratio = 0
            best_pos = -1
            best_wsize = 0
            min_window = max(3, int(n_search * 0.5))

            for wsize in range(n_search, min_window - 1, -1):
                if best_ratio > 0.8:
                    break
                window_stripped = search_stripped[:wsize]
                for i in range(len(modified_lines) - wsize + 1):
                    file_subset = [l.strip() for l in modified_lines[i:i + wsize]]
                    matches = sum(1 for a, b in zip(file_subset, window_stripped) if a == b)
                    ratio = matches / wsize
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pos = i
                        best_wsize = wsize

            if best_ratio >= 0.5 and best_pos >= 0:
                match_index = best_pos
                n_delete = best_wsize
                msg = f"[cyan]Partial-matched hunk at line {best_pos + 1} ({best_ratio:.0%} match, window={best_wsize})[/cyan]"
                console.print(msg)
                log(msg)

        if match_index != -1:
            del modified_lines[match_index:match_index + n_delete]
            for i, line in enumerate(replace_block):
                modified_lines.insert(match_index + i, line)

            msg = f"[green]Applied hunk at line {match_index + 1}[/green]"
            console.print(msg)
            log(msg)
            applied_hunks += 1
        else:
            msg = "[red]Failed to find matching context for hunk:[/red]"
            console.print(msg)
            log(msg)
            ctx_head = search_block[:5]
            if Panel is not None:
                console.print(Panel(
                    "\n".join(ctx_head) + ("\n..." if len(search_block) > 5 else ""),
                    title="Expected Context (First 5 lines)",
                ))
            log("Expected Context snippet:")
            for l in ctx_head:
                log(f"| {l}")

    if applied_hunks == len(hunks):
        new_text = "\n".join(modified_lines)
        if had_trailing_newline and not new_text.endswith("\n"):
            new_text += "\n"
        elif not had_trailing_newline and new_text.endswith("\n"):
            new_text = new_text[:-1]
        file_path.write_text(new_text, encoding="utf-8")
        return True

    return False


def extract_files_from_diff(diff_text: str) -> List[Tuple[str, str]]:
    results = []

    file_diffs = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    file_diffs = [d for d in file_diffs if d.strip().startswith("diff --git")]

    for single_diff in file_diffs:
        fname_match = re.search(r"diff --git a/\S+ b/(\S+)", single_diff)
        if not fname_match:
            continue
        filepath = fname_match.group(1)

        is_new_file = ("new file mode" in single_diff or "--- /dev/null" in single_diff)
        if not is_new_file:
            console.print(f"[yellow]Skipping diff extraction for '{filepath}' (edit diff — would destroy existing file)[/yellow]")
            continue

        lines = single_diff.split("\n")
        content_lines = []
        in_hunk = False

        for line in lines:
            if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
                continue
            if line.startswith("@@"):
                in_hunk = True
                continue
            if line.startswith("\\ No newline"):
                continue

            if in_hunk:
                if line.startswith("+"):
                    content_lines.append(line[1:])
                elif line.startswith(" "):
                    content_lines.append(line[1:])
                elif line == "":
                    content_lines.append("")

        if not content_lines:
            continue

        content = "\n".join(content_lines)
        if not content.endswith("\n"):
            content += "\n"
        results.append((filepath, content))
        console.print(f"[cyan]Extracted NEW file '{filepath}' from diff ({len(content)} bytes)[/cyan]")

    return results


def apply_write_files(
    actions: List[Tuple[str, str]],
    allowlist: List[str],
    turn_dir: Path,
) -> bool:
    written = 0
    log_parts = []

    norm_allowlist = set()
    for p in allowlist:
        s = str(p)
        norm_allowlist.add(s)
        norm_allowlist.add(str(Path(s)))
        norm_allowlist.add(os.path.basename(s))
        parts = Path(s).parts
        for i in range(len(parts)):
            norm_allowlist.add(str(Path(*parts[i:])))

    for filepath, content in actions:
        raw = filepath.strip()
        if os.path.isabs(raw):
            clean_path = raw
        else:
            clean_path = raw.lstrip("/")

        allowed = False
        for ap in norm_allowlist:
            ap_str = str(ap)
            if (
                clean_path == ap_str
                or clean_path.endswith(ap_str)
                or ap_str.endswith(clean_path)
                or os.path.basename(clean_path) == ap_str
            ):
                allowed = True
                break

        if not norm_allowlist or not allowlist:
            allowed = True

        if not allowed:
            log_parts.append(f"SKIPPED (not in allowlist): {filepath} (allowlist: {[str(a) for a in allowlist]})")
            console.print(f"[yellow]Skipping {filepath} — not in allowlist ({[str(a) for a in allowlist]})[/yellow]")
            continue

        try:
            target = Path(clean_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            log_parts.append(f"WROTE: {filepath} ({len(content)} bytes)")
            console.print(f"[green]Wrote file: {filepath}[/green]")
            written += 1

            if is_git_repo():
                run_shell(f"git add {target.as_posix()}")
        except Exception as e:
            log_parts.append(f"FAILED: {filepath} — {e}")
            console.print(f"[red]Failed to write {filepath}: {e}[/red]")

    (turn_dir / "write_files.log").write_text("\n".join(log_parts), encoding="utf-8")
    return written > 0
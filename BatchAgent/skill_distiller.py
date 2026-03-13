"""
Building a "Sleep-Learning" (Offline Consolidation) and "Just-In-Time (JIT) Compilation" mechanism for your Agent.

By having the system autonomously crawl, distill, test, and index skills during idle time, you build a massive capability library without polluting the small model's limited context window. Furthermore, by hijacking existing tools (read_url, search_code, load_domain_tools) instead of adding new ones, you keep the model's action space small and stable.

Why this Architecture is Revolutionary
Zero-Pollution Context: The small model (e.g., Qwen 7B) never sees the 10,000-token prompt from the original Claude skill. It only sees the intercept_search_code output, which is a highly dense, 200-token JSON schema.

"Self-Healing" Execution: The verify_skill_with_agent function uses your own agent framework to debug the code before saving it. If the distilled Python code has a syntax error, the sub-agent fixes it, ensuring that only proven, working code enters the index.

OOD (Out-of-Distribution) Generalization: By intercepting read_url, if a user asks your agent to "Use the technique described in this URL: github.com/...", your agent doesn't crash from context overflow. It seamlessly delegates the reading to the Distiller, which translates it into native tools on the fly.


Real-world skills (like those from the OpenClaw or Claude ecosystems) are not flat text files; they are repositories containing README.md/SKILL.md, Python scripts, reference prompts, and sometimes testing environments.

If we feed a whole directory into a small model, the context window will explode, and the model will suffer from severe hallucination.

To solve this, we will upgrade the skill_distiller into a "Distiller Mini-Agent".

Role: A strict, low-temperature librarian. It does not invent code; it only reads, organizes, checks safety, and packages.

Tools: It will have exclusive access to list_remote_repo and read_remote_file.

Flow: It reads SKILL.md first, figures out which .py files actually matter, reads only those specific files, and then calls a final tool submit_distilled_skill to finish its job.


Integrating the Idle Skill Discovery Daemon into an existing arq async worker is an excellent architectural pattern. It allows your worker to effectively utilize its idle compute cycles to self-improve, while still remaining highly responsive to priority tasks via Redis.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Your wrappers
from BatchAgent.llm_wrapper import complete_with_async, complete_with_continuation_async
from BatchAgent.mini_batch_agent_libs import run_shell

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
DEFAULT_GITHUB_TIMEOUT_S = 20.0
DEFAULT_MAX_FILE_CHARS = 12_000     # keep small-model friendly
DEFAULT_MAX_TURNS = 6
DEFAULT_DISTILL_TEMPERATURE = 0.1

SKILL_INDEX_PATH = Path("agent_workspace/.agent/distilled_skills_index.json")
SKILL_CODE_DIR = Path("agent_workspace/.agent/distilled_skills_code")  # store .py files
SKILL_SANDBOX_DIR = Path("agent_workspace/.agent/skill_sandbox")       # verification workdir

# If you want a “hard allowlist” of deps for safety, put them here.
# None means allow any pip install command but still scan.
ALLOWED_PIP_PACKAGES: Optional[set[str]] = None

# -----------------------------
# Distiller tool schema
# -----------------------------
DISTILLER_TOOLS_SCHEMA = [
    {
        "name": "list_remote_repo",
        "description": "List all files and directories in a given GitHub URL (repo or subdir).",
        "properties": {"url": {"type": "string", "description": "The GitHub directory URL."}},
        "required": ["url"],
    },
    {
        "name": "read_remote_file",
        "description": "Read the text content of a specific file from GitHub. Always read SKILL.md or README.md first.",
        "properties": {"file_url": {"type": "string", "description": "The specific file URL returned from list_remote_repo."}},
        "required": ["file_url"],
    },
    {
        "name": "submit_distilled_skill",
        "description": "Submit a distilled, safe, standalone python tool.",
        "properties": {
            "is_safe": {"type": "boolean"},
            "domain_category": {"type": "string"},
            "skill_name": {"type": "string", "description": "Short, lowercase name for the tool."},
            "summary": {"type": "string", "description": "What the tool does."},
            "env_setup_bash": {"type": "string", "description": "e.g., pip install pandas"},
            "schema_properties": {"type": "object", "description": "JSON schema for arguments."},
            "required_args": {"type": "array", "items": {"type": "string"}},
            "standalone_python_code": {"type": "string", "description": "Single unified python script."},
            "test_payload_json": {"type": "string", "description": "JSON string to test the code."},
        },
        "required": ["is_safe", "domain_category", "skill_name", "summary", "standalone_python_code"],
    },
]

import os
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ========== Tool Action adapters ==========
def _action_name_args(action: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Robustly extract tool action name/args from your LLM wrapper.
    Supports common shapes:
      - action.name + action.args
      - {"name": "...", "arguments": {...}}
      - {"tool": "...", "args": {...}}
    """
    if action is None:
        return None, {}
    if hasattr(action, "name"):
        return getattr(action, "name"), getattr(action, "args", {}) or {}
    if isinstance(action, dict):
        name = action.get("name") or action.get("tool")
        args = action.get("args") or action.get("arguments") or {}
        # Sometimes "arguments" is a JSON string:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        return name, args
    return None, {}

def _safe_join(base: Path, rel: str) -> Path:
    """
    Prevent path traversal: only allow writes inside base directory.
    """
    rel = rel.lstrip("/").replace("\\", "/")
    candidate = (base / rel).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise ValueError(f"Unsafe path (traversal): {rel}")
    return candidate

def write_text_file(session_dir: Path, relpath: str, content: str) -> str:
    session_dir.mkdir(parents=True, exist_ok=True)
    p = _safe_join(session_dir, relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {relpath} ({len(content)} chars)"

def run_bash(session_dir: Path, command: str, timeout_s: int = 120) -> Tuple[int, str]:
    """
    Uses your project's run_shell wrapper, runs within session_dir.
    Expects run_shell(cmd, cwd=..., timeout_s=...) -> (rc, output)
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    rc, out = run_shell(command, cwd=str(session_dir), timeout_s=timeout_s)
    return rc, out

# ========== QA harness tool schema (for verify_skill_with_agent) ==========
QA_TOOLS_SCHEMA = [
    {
        "name": "write_file",
        "description": "Write a UTF-8 text file relative to the session_dir. Use this to create the tool script.",
        "properties": {
            "path": {"type": "string", "description": "Relative file path inside session_dir"},
            "content": {"type": "string", "description": "Text content to write"},
        },
        "required": ["path", "content"],
    },
    {
        "name": "bash",
        "description": "Run a shell command in session_dir and return stdout/stderr.",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "timeout_s": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    },
    {
        "name": "finish_task",
        "description": "Call this when verification is complete and successful.",
        "properties": {
            "ok": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": ["ok"],
    },
]

async def run_qa_agent_loop(
    client: Any,
    model: str,
    goal: str,
    session_dir: Path,
    max_turns: int = 8,
) -> Tuple[bool, str]:
    """
    A small “mini-agent runtime” that routes write_file/bash/finish_task tools.
    This makes verify_skill_with_agent actually runnable regardless of your broader agent framework.
    """
    messages = [{"role": "system", "content": "You are a strict QA agent. Use tools to accomplish the goal."},
                {"role": "user", "content": goal}]
    last_note = ""

    for t in range(max_turns):
        content, actions = await complete_with_continuation_async(
            client=client,
            model=model,
            messages=messages,
            temperature=0.2,
            max_output_tokens=1200,
            tools=QA_TOOLS_SCHEMA,
            tool_strategy="native",
            session_dir=session_dir,   # keep your wrapper behavior
            enable_thinking=True,
        )
        messages.append({"role": "assistant", "content": content})
        tool_feedback: List[str] = []
        finished = None

        for a in actions or []:
            name, args = _action_name_args(a)
            if not name:
                continue

            if name == "write_file":
                try:
                    res = write_text_file(session_dir, args.get("path", ""), args.get("content", ""))
                    tool_feedback.append(f"### write_file result\n{res}")
                except Exception as e:
                    tool_feedback.append(f"### write_file error\n{e!r}")

            elif name == "bash":
                cmd = args.get("command", "")
                timeout_s = int(args.get("timeout_s", 120) or 120)
                rc, out = run_bash(session_dir, cmd, timeout_s=timeout_s)
                tool_feedback.append(f"### bash rc={rc}\n```\n{out}\n```")

            elif name == "finish_task":
                finished = args
                break

        if finished is not None:
            ok = bool(finished.get("ok", False))
            notes = finished.get("notes", "") or ""
            return ok, notes or content or ""

        if not tool_feedback:
            tool_feedback.append("System: No tools were called. You must use write_file/bash and then finish_task(ok=true/false).")
        last_note = "\n\n".join(tool_feedback)
        messages.append({"role": "user", "content": last_note})

    return False, last_note or "QA agent max turns exceeded."


from typing import Iterable

try:
    import jsonschema
except Exception:
    jsonschema = None

SUBMIT_DISTILLED_SKILL_JSONSCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["is_safe", "domain_category", "skill_name", "summary", "standalone_python_code"],
    "properties": {
        "is_safe": {"type": "boolean"},
        "domain_category": {"type": "string"},
        "skill_name": {"type": "string"},
        "summary": {"type": "string"},
        "env_setup_bash": {"type": "string"},
        "schema_properties": {"type": "object"},
        "required_args": {"type": "array", "items": {"type": "string"}},
        "standalone_python_code": {"type": "string"},
        "test_payload_json": {"type": "string"},
        "source_url": {"type": "string"},
    },
    "additionalProperties": True,
}

def _simple_type_check(distilled: Dict[str, Any]) -> List[str]:
    """
    Fallback if jsonschema is not installed.
    """
    errs = []
    req = ["is_safe", "domain_category", "skill_name", "summary", "standalone_python_code"]
    for k in req:
        if k not in distilled:
            errs.append(f"missing_required_field:{k}")

    if "is_safe" in distilled and not isinstance(distilled["is_safe"], bool):
        errs.append("is_safe_not_bool")
    for s in ["domain_category", "skill_name", "summary", "standalone_python_code"]:
        if s in distilled and not isinstance(distilled[s], str):
            errs.append(f"{s}_not_str")

    if "schema_properties" in distilled and distilled["schema_properties"] is not None and not isinstance(distilled["schema_properties"], dict):
        errs.append("schema_properties_not_object")
    if "required_args" in distilled and distilled["required_args"] is not None:
        ra = distilled["required_args"]
        if not isinstance(ra, list) or any(not isinstance(x, str) for x in ra):
            errs.append("required_args_not_list_of_str")

    if "env_setup_bash" in distilled and distilled["env_setup_bash"] is not None and not isinstance(distilled["env_setup_bash"], str):
        errs.append("env_setup_bash_not_str")
    if "test_payload_json" in distilled and distilled["test_payload_json"] is not None and not isinstance(distilled["test_payload_json"], str):
        errs.append("test_payload_json_not_str")
    return errs

def validate_distilled_skill(distilled: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Returns (ok, errors, normalized_distilled)
    """
    errors: List[str] = []
    d = dict(distilled or {})

    # 1) JSON Schema validation
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=d, schema=SUBMIT_DISTILLED_SKILL_JSONSCHEMA)
        except Exception as e:
            errors.append(f"jsonschema_validation_error:{e}")
    else:
        errors.extend(_simple_type_check(d))

    # 2) Normalize skill_name
    d["skill_name"] = _normalize_skill_name(d.get("skill_name", ""))

    if not d["skill_name"]:
        errors.append("skill_name_empty_after_normalize")

    # 3) Ensure test_payload_json exists + is valid JSON string
    if not d.get("test_payload_json"):
        d["test_payload_json"] = "{}"
    try:
        json.loads(d["test_payload_json"])
    except Exception as e:
        errors.append(f"test_payload_json_invalid_json:{e}")

    # 4) Ensure schema_properties / required_args default
    if d.get("schema_properties") is None:
        d["schema_properties"] = {}
    if d.get("required_args") is None:
        d["required_args"] = []

    # 5) Basic “standalone tool” heuristics (you can tighten/loosen)
    code = d.get("standalone_python_code", "")
    if "sys.argv" not in code:
        errors.append("standalone_python_code_missing_sys_argv")
    if "json.loads" not in code:
        # not strictly required but strongly expected
        errors.append("standalone_python_code_missing_json_loads")
    if "__main__" not in code:
        errors.append("standalone_python_code_missing_main_guard")

    # 6) Safety scan (never trust model)
    ok_safe, reasons = safety_scan_distilled(d)
    if not ok_safe:
        errors.append(f"safety_scan_failed:{reasons}")

    ok = len(errors) == 0
    return ok, errors, d
# -----------------------------
# Utilities
# -----------------------------
def _now_ts() -> int:
    return int(time.time())

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _ensure_dirs():
    SKILL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKILL_CODE_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

def _load_index() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_dirs()
    if SKILL_INDEX_PATH.exists():
        with open(SKILL_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _write_index(index: Dict[str, List[Dict[str, Any]]]) -> None:
    _ensure_dirs()
    tmp = SKILL_INDEX_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SKILL_INDEX_PATH)

def _normalize_skill_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "custom_tool"
    return name

def parse_github_url(url: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Supports:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/tree/branch/path
    - https://github.com/owner/repo/blob/branch/path/file
    Returns: (owner, repo, path, branch)
    """
    m = re.search(r"github\.com/([^/]+)/([^/]+)(?:/(tree|blob)/([^/]+)(?:/(.*))?)?", url)
    if not m:
        return "", "", "", None
    owner, repo = m.group(1), m.group(2)
    mode = m.group(3)
    branch = m.group(4) if mode in ("tree", "blob") else None
    path = m.group(5) or ""
    return owner, repo, path, branch

def github_contents_api(owner: str, repo: str, path: str, branch: Optional[str]) -> str:
    # GET /repos/{owner}/{repo}/contents/{path}?ref=branch
    base = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}".rstrip("/")
    if branch:
        return f"{base}?ref={branch}"
    return base

def github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    # https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"

# -----------------------------
# Safety scanning
# -----------------------------
DANGEROUS_PATTERNS = [
    # destructive file ops
    r"\brm\s+-rf\b",
    r"\bshred\b",
    r"\bmkfs\.",
    r"\bdd\s+if=",
    r"\bchmod\s+777\b",
    r"\bchown\s+root\b",
    # suspicious network exfil
    r"\bcurl\s+.*\|\s*sh\b",
    r"\bwget\s+.*\|\s*sh\b",
    r"\bnc\s+-e\b",
    r"\bbash\s+-i\b",
    r"/dev/tcp/",
    # privilege escalation
    r"\bsudo\b",
    # python dangerous
    r"os\.system\(",
    r"subprocess\.(Popen|call|run)\(",
    r"eval\(",
    r"exec\(",
]

def safety_scan_text(text: str) -> Tuple[bool, List[str]]:
    hits = []
    for pat in DANGEROUS_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)
    return (len(hits) == 0), hits

def safety_scan_distilled(distilled: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok_code, hits_code = safety_scan_text(distilled.get("standalone_python_code", ""))
    if not ok_code:
        reasons.append(f"dangerous_code_patterns={hits_code}")

    env = distilled.get("env_setup_bash", "") or ""
    ok_env, hits_env = safety_scan_text(env)
    if not ok_env:
        reasons.append(f"dangerous_env_patterns={hits_env}")

    # Optional: pip allowlist
    if ALLOWED_PIP_PACKAGES is not None and env.strip():
        # naive parse: "pip install a b==1.2"
        toks = shlex.split(env)
        if len(toks) >= 2 and toks[0] == "pip" and toks[1] == "install":
            pkgs = [t for t in toks[2:] if not t.startswith("-")]
            bad = []
            for p in pkgs:
                base = re.split(r"[<>=]", p)[0].strip()
                if base and base not in ALLOWED_PIP_PACKAGES:
                    bad.append(base)
            if bad:
                reasons.append(f"pip_packages_not_allowed={bad}")

    return (len(reasons) == 0), reasons

# -----------------------------
# GitHub Remote Tools (async)
# -----------------------------
class GitHubRemote:
    def __init__(self, token: Optional[str] = None):
        self.token = token

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github+json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def list_remote_repo(self, url: str) -> str:
        owner, repo, path, branch = parse_github_url(url)
        if not owner:
            return "Error: Invalid GitHub URL."

        api_url = github_contents_api(owner, repo, path, branch)
        async with httpx.AsyncClient(timeout=DEFAULT_GITHUB_TIMEOUT_S, headers=self._headers()) as client:
            try:
                r = await client.get(api_url)
                if r.status_code != 200:
                    return f"Error fetching repo: {r.status_code} - {r.text}"
                items = r.json()
            except Exception as e:
                return f"Request failed: {e!r}"

        if isinstance(items, dict) and "message" in items:
            return items["message"]

        # If a file is passed, GitHub contents returns dict for a file
        if isinstance(items, dict) and items.get("type") == "file":
            return "Repository Contents:\n" + f"[FILE] {items.get('name')} (Path: {items.get('path')}, URL: {items.get('html_url')})"

        file_tree = []
        for it in items:
            t = "DIR " if it.get("type") == "dir" else "FILE"
            file_tree.append(f"[{t}] {it.get('name')} (Path: {it.get('path')}, URL: {it.get('html_url')})")
        return "Repository Contents:\n" + "\n".join(file_tree)

    async def read_remote_file(self, file_url: str) -> str:
        owner, repo, path, branch = parse_github_url(file_url)
        if not owner:
            # Also accept raw.githubusercontent.com directly
            if "raw.githubusercontent.com" in file_url:
                return await self._read_raw_url(file_url)
            return "Error: Invalid GitHub file URL."

        # If user passed /blob/ URL, branch is present; if not, fall back to main
        branch = branch or "main"
        raw_url = github_raw_url(owner, repo, branch, path)

        return await self._read_raw_url(raw_url)

    async def _read_raw_url(self, raw_url: str) -> str:
        async with httpx.AsyncClient(timeout=DEFAULT_GITHUB_TIMEOUT_S) as client:
            try:
                r = await client.get(raw_url)
                if r.status_code != 200:
                    return f"Error reading file: HTTP {r.status_code}"
                content = r.text
            except Exception as e:
                return f"Request failed: {e!r}"

        if len(content) > DEFAULT_MAX_FILE_CHARS:
            half = DEFAULT_MAX_FILE_CHARS // 2
            return content[:half] + "\n\n...[CONTENT TRUNCATED]...\n\n" + content[-half:]
        return content

# -----------------------------
# Distiller Mini-Agent
# -----------------------------
async def run_distiller_agent(
    client: Any,
    model: str,
    skill_repo_url: str,
    github: GitHubRemote,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> Optional[Dict[str, Any]]:
    system_prompt = f"""
You are an elite AI Skill Distiller and Security Auditor.

Goal: Explore a remote repository containing a skill, extract ONLY functional code, verify its safety, and submit a distilled version.

RULES:
1) NO INVENTION: Do not create new algorithms. Only reorganize/clean code you read.
2) SAFETY FIRST: If you see destructive behavior, data exfiltration, privilege escalation, or anything suspicious, set is_safe=false and still submit with a short summary.
3) EXPLORATION ORDER:
   a) list_remote_repo on the provided URL
   b) read_remote_file on SKILL.md / README.md / docs
   c) read only the minimal python files needed
4) OUTPUT: final standalone_python_code must be ONE script:
   - Reads JSON args from sys.argv[1]
   - Validates inputs
   - Prints JSON result to stdout
5) Keep schema_properties concise and correct.

Target Skill URL: {skill_repo_url}
Start by listing repository contents.
""".strip()

    messages = [{"role": "system", "content": system_prompt}]
    distilled_result: Optional[Dict[str, Any]] = None

    for turn in range(max_turns):
        logger.info("[Distiller] turn=%d/%d url=%s", turn + 1, max_turns, skill_repo_url)

        content, actions = await complete_with_continuation_async(
            client=client,
            model=model,
            messages=messages,
            temperature=DEFAULT_DISTILL_TEMPERATURE,
            max_output_tokens=2048,
            tools=DISTILLER_TOOLS_SCHEMA,
            tool_strategy="native",
            enable_thinking=True,
        )

        messages.append({"role": "assistant", "content": content})

        feedback_blocks: List[str] = []
        submitted = None

        for action in actions or []:
            name = getattr(action, "name", None)
            args = getattr(action, "args", None) or {}
            if not name:
                continue

            if name == "list_remote_repo":
                res = await github.list_remote_repo(args.get("url", ""))
                feedback_blocks.append(f"### Result for list_remote_repo:\n```\n{res}\n```")

            elif name == "read_remote_file":
                res = await github.read_remote_file(args.get("file_url", ""))
                feedback_blocks.append(f"### Result for read_remote_file:\n```\n{res}\n```")

            elif name == "submit_distilled_skill":
                submitted = args

                # Normalize and validate BEFORE accepting
                submitted["source_url"] = skill_repo_url
                ok, errs, normalized = validate_distilled_skill(submitted)

                if not ok:
                    # Feed errors back to distiller and continue turns
                    feedback_blocks.append(
                        "### submit_distilled_skill rejected by validator\n"
                        f"Errors:\n- " + "\n- ".join(errs) + "\n\n"
                        "Fix the submission and call submit_distilled_skill again. "
                        "Do NOT add new features; just correct schema/entrypoint/test payload."
                    )
                    # Important: do NOT set distilled_result, keep looping
                    submitted = None
                else:
                    # accept normalized
                    submitted = normalized
                    break

        if submitted:
            submitted["source_url"] = skill_repo_url
            # Normalize name early
            submitted["skill_name"] = _normalize_skill_name(submitted.get("skill_name", ""))
            distilled_result = submitted
            logger.info("[Distiller] submitted skill=%s safe=%s", submitted.get("skill_name"), submitted.get("is_safe"))
            break

        if not feedback_blocks:
            feedback_blocks.append(
                "System Warning: You did not use any tools. Use list_remote_repo/read_remote_file, or submit_distilled_skill if done."
            )

        messages.append({"role": "user", "content": "\n\n".join(feedback_blocks)})

    return distilled_result

# -----------------------------
# Distill raw text (JIT / fallback)
# -----------------------------
async def distill_raw_skill(
    client: Any,
    model: str,
    raw_skill_content: str,
    source_url: str,
) -> Optional[Dict[str, Any]]:
    system_prompt = """
You are an elite AI Code Distiller. Analyze a verbose Skill/Tool and extract core logic into a lightweight standalone Python module.

CRITICAL:
1) SAFETY: Set is_safe=false if destructive OS commands, exfiltration, privilege escalation, or suspicious payload.
2) DISTILLATION: Remove conversation glue; keep minimal functional code.
3) ENV: Identify pip packages if truly required.
4) CODE: Must accept JSON from sys.argv[1] and print JSON to stdout.
5) TEST: Provide a sample test_payload_json.

Return STRICT JSON only:
{
  "is_safe": true,
  "domain_category": "software_eng",
  "skill_name": "lowercase_with_underscores",
  "summary": "...",
  "env_setup_bash": "pip install ...",
  "schema_properties": {...},
  "required_args": [...],
  "standalone_python_code": "...",
  "test_payload_json": "{...}"
}
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Source: {source_url}\n\nRaw Content:\n{raw_skill_content[:DEFAULT_MAX_FILE_CHARS]}"},
    ]

    content, _ = await complete_with_async(
        client=client,
        model=model,
        messages=messages,
        temperature=0.1,
        max_output_tokens=2048,
        stream=False,
    )

    # Be strict: extract json from fences if any
    try:
        c = content.strip()
        if "```" in c:
            # pick first fenced block
            c = c.split("```", 1)[1]
            c = c.split("```", 1)[0]
            c = c.replace("json", "", 1).strip()

        distilled = json.loads(c)
        distilled["source_url"] = source_url
        distilled["skill_name"] = _normalize_skill_name(distilled.get("skill_name", ""))

        ok, reasons = safety_scan_distilled(distilled)
        if not ok:
            logger.warning("[distill_raw_skill] rejected by safety scan: %s", reasons)
            return None
        if not distilled.get("is_safe", False):
            return None
        return distilled

    except Exception as e:
        logger.error("[distill_raw_skill] parse failed: %r content=%s", e, content[:800])
        return None

# -----------------------------
# Index + storage
# -----------------------------
def save_distilled_to_index(distilled: Dict[str, Any]) -> None:
    _ensure_dirs()

    domain = (distilled.get("domain_category") or "general").strip() or "general"
    skill_name = _normalize_skill_name(distilled.get("skill_name", "custom_tool"))

    code = distilled.get("standalone_python_code", "")
    code_hash = _sha256_text(code)
    src = distilled.get("source_url", "")
    src_hash = _sha256_text(src)

    schema = {
        "name": skill_name,
        "description": distilled.get("summary", ""),
        "properties": distilled.get("schema_properties", {}) or {},
        "required": distilled.get("required_args", []) or [],
    }

    index = _load_index()
    index.setdefault(domain, [])

    # De-dup by (skill_name + code_hash) OR same source_url
    for existing in index[domain]:
        if existing.get("schema", {}).get("name") == skill_name and existing.get("code_hash") == code_hash:
            logger.info("[index] already exists skill=%s", skill_name)
            return
        if existing.get("source_url") == src:
            # update in place (newer distillation)
            existing.update({
                "schema": schema,
                "env_setup_bash": distilled.get("env_setup_bash", "") or "",
                "code_hash": code_hash,
                "source_hash": src_hash,
                "updated_at": _now_ts(),
            })
            # update code file
            _write_skill_code_file(skill_name, code)
            _write_index(index)
            logger.info("[index] updated skill=%s from source=%s", skill_name, src)
            return

    rec = {
        "source_url": src,
        "source_hash": src_hash,
        "schema": schema,
        "env_setup_bash": distilled.get("env_setup_bash", "") or "",
        "code_hash": code_hash,
        "created_at": _now_ts(),
        "updated_at": _now_ts(),
    }
    index[domain].append(rec)
    _write_skill_code_file(skill_name, code)
    _write_index(index)
    logger.info("[index] saved skill=%s domain=%s", skill_name, domain)

def _write_skill_code_file(skill_name: str, code: str) -> Path:
    _ensure_dirs()
    p = SKILL_CODE_DIR / f"{skill_name}.py"
    tmp = p.with_suffix(".py.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")
    os.replace(tmp, p)
    return p

def load_skill_schemas(domain: str) -> List[Dict[str, Any]]:
    idx = _load_index()
    return [x["schema"] for x in idx.get(domain, [])]

def search_skill_index(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    idx = _load_index()
    q = query.lower().strip()
    hits = []
    for domain, skills in idx.items():
        for s in skills:
            schema = s.get("schema", {})
            name = (schema.get("name") or "").lower()
            desc = (schema.get("description") or "").lower()
            if q in name or q in desc:
                hits.append({"domain": domain, **s})
    return hits[:limit]

# -----------------------------
# Verification
# -----------------------------
async def verify_distilled_skill_locally(
    distilled: Dict[str, Any],
    python_bin: str = "python3",
    timeout_s: int = 120,
) -> Tuple[bool, str]:
    """
    Executes distilled tool in a sandbox directory. This is a deterministic verifier,
    not relying on an LLM agent. Much more stable for “self-healing index”.
    """
    _ensure_dirs()
    ok, reasons = safety_scan_distilled(distilled)
    if not ok:
        return False, f"Rejected by safety scan: {reasons}"

    skill_name = _normalize_skill_name(distilled.get("skill_name", "custom_tool"))
    workdir = SKILL_SANDBOX_DIR / f"{skill_name}_{_now_ts()}"
    workdir.mkdir(parents=True, exist_ok=True)

    code_path = workdir / f"{skill_name}.py"
    code_path.write_text(distilled.get("standalone_python_code", ""), encoding="utf-8")

    env_cmd = (distilled.get("env_setup_bash") or "").strip()
    if env_cmd:
        # lightweight guard: only allow pip install by default
        if not env_cmd.startswith("pip install"):
            return False, f"env_setup_bash not allowed (only pip install permitted): {env_cmd}"

        # run pip install
        rc, out = run_shell(env_cmd, cwd=str(workdir), timeout_s=timeout_s)
        if rc != 0:
            return False, f"pip install failed rc={rc}\n{out}"

    payload = distilled.get("test_payload_json") or "{}"
    # Ensure payload is valid JSON string (distiller might produce dict-like)
    try:
        if isinstance(payload, str):
            json.loads(payload)
        else:
            payload = json.dumps(payload, ensure_ascii=False)
    except Exception:
        payload = "{}"

    cmd = f"{python_bin} {shlex.quote(str(code_path))} {shlex.quote(payload)}"
    rc, out = run_shell(cmd, cwd=str(workdir), timeout_s=timeout_s)
    if rc != 0:
        return False, f"execution failed rc={rc}\n{out}"

    # Optional: require JSON output
    try:
        _ = json.loads(out.strip().splitlines()[-1])
    except Exception:
        # allow non-json but warn
        return True, f"OK (non-JSON output)\n{out}"

    return True, f"OK\n{out}"

# -----------------------------
# Idle discovery daemon
# -----------------------------
@dataclass
class DiscoveryItem:
    url: str
    priority: int = 0

async def idle_skill_discovery_loop(
    client: Any,
    model: str,
    github_token: Optional[str] = None,
    discovery_queue: Optional[List[DiscoveryItem]] = None,
    sleep_between_s: int = 10,
    verify_before_save: bool = True,
):
    """
    Run when worker is idle: distill -> (verify) -> save to index.
    You can wire this into arq as a periodic job or a low-priority long-running task.
    """
    _ensure_dirs()
    github = GitHubRemote(token=github_token)

    if discovery_queue is None:
        discovery_queue = [
            DiscoveryItem("https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/data_analyzer"),
            DiscoveryItem("https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills/arxiv_search"),
        ]

    # process higher priority first
    discovery_queue = sorted(discovery_queue, key=lambda x: -x.priority)

    for item in discovery_queue:
        url = item.url
        idx = _load_index()
        already = any(s.get("source_url") == url for ds in idx.values() for s in ds)
        if already:
            continue

        logger.info("[IdleDaemon] distill start url=%s", url)
        distilled = await run_distiller_agent(client, model, url, github=github)
        if not distilled:
            logger.warning("[IdleDaemon] distill failed url=%s", url)
            continue

        # second-pass safety scan (don’t trust model)
        ok, reasons = safety_scan_distilled(distilled)
        if not ok:
            logger.warning("[IdleDaemon] rejected by safety scan url=%s reasons=%s", url, reasons)
            continue
        if not distilled.get("is_safe", False):
            logger.warning("[IdleDaemon] model flagged unsafe url=%s", url)
            continue

        # local verification
        if verify_before_save:
            v_ok, v_msg = await verify_distilled_skill_locally(distilled)
            if not v_ok:
                logger.warning("[IdleDaemon] verify failed url=%s msg=%s", url, v_msg[:800])
                continue
            logger.info("[IdleDaemon] verify OK url=%s", url)

        save_distilled_to_index(distilled)
        await asyncio.sleep(sleep_between_s)

# -----------------------------
# Tool interceptors
# -----------------------------
async def intercept_read_url(
    client: Any,
    model: str,
    url: str,
    current_task: str,
    github_token: Optional[str] = None,
) -> str:
    """
    If user gives a GitHub repo skill, distill and return dense summary + schema reference.
    """
    if "github.com" in url or "raw.githubusercontent.com" in url:
        github = GitHubRemote(token=github_token)

        # Try to treat it as a repo/subdir, run distiller agent
        distilled = await run_distiller_agent(client, model, url, github=github)
        if distilled:
            ok, reasons = safety_scan_distilled(distilled)
            if ok and distilled.get("is_safe", False):
                # optionally verify
                v_ok, v_msg = await verify_distilled_skill_locally(distilled)
                if v_ok:
                    save_distilled_to_index(distilled)
                    schema = {
                        "name": distilled["skill_name"],
                        "description": distilled.get("summary", ""),
                        "properties": distilled.get("schema_properties", {}) or {},
                        "required": distilled.get("required_args", []) or [],
                    }
                    return (
                        "[SYSTEM INTERCEPTION]\n"
                        "Detected a complex GitHub skill source. Distilled & verified locally.\n\n"
                        f"Summary: {distilled.get('summary','')}\n"
                        f"Skill name: {distilled['skill_name']}\n"
                        f"Schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
                        "Next: call search_code with the skill_name to view code, or load_domain_tools(domain_category)."
                    )
                else:
                    return (
                        "[SYSTEM INTERCEPTION]\n"
                        "Distillation succeeded but verification failed; not cached.\n"
                        f"Reason: {v_msg[:800]}"
                    )

        # Fallback: naive raw fetch & distill_raw_skill (not recommended for repos)
        # (You can implement actual web fetching here if you have a read_url tool)
        return "[SYSTEM INTERCEPTION] Could not distill GitHub source reliably. Provide a direct file URL (SKILL.md / .py) or enable repo tools."

    return "Raw web content (non-github): implement your normal read_url here."

def intercept_search_code(query: str) -> str:
    hits = search_skill_index(query)
    if not hits:
        return "No matching skills found in Local Skill Registry."

    blocks = []
    for h in hits:
        schema = h.get("schema", {})
        blocks.append(
            "### Distilled Skill Found\n"
            f"- domain: {h.get('domain')}\n"
            f"- name: {schema.get('name')}\n"
            f"- description: {schema.get('description')}\n"
            f"- usage_schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"- source_url: {h.get('source_url')}\n"
        )

    return "Found relevant skills in Local Skill Registry:\n\n" + "\n---\n".join(blocks)

def intercept_load_domain_tools(domain: str) -> List[Dict[str, Any]]:
    return load_skill_schemas(domain)

# -----------------------------
# Optional: agent-driven “self-healing” verification
# (only if you still want an LLM to patch minor syntax errors)
# -----------------------------
async def verify_skill_with_agent(
    client: Any,
    model: str,
    distilled_skill: Dict[str, Any],
    test_workspace: Path,
) -> bool:
    skill_name = _normalize_skill_name(distilled_skill.get("skill_name", "custom_tool"))
    test_goal = (
        f"Verify distilled skill '{skill_name}'.\n"
        f"1) If env_setup_bash is non-empty, run it.\n"
        f"2) Write the script to {skill_name}.py (use write_file).\n"
        f"3) Run: python3 {skill_name}.py '<test_payload_json>'\n"
        f"4) If successful, call finish_task(ok=true, notes='...'). Otherwise ok=false with error.\n\n"
        f"env_setup_bash:\n{distilled_skill.get('env_setup_bash','')}\n\n"
        f"standalone_python_code:\n{distilled_skill.get('standalone_python_code','')}\n\n"
        f"test_payload_json:\n{distilled_skill.get('test_payload_json','{}')}\n"
    )
    ok, _notes = await run_qa_agent_loop(
        client=client,
        model=model,
        goal=test_goal,
        session_dir=test_workspace,
        max_turns=8,
    )
    return ok

# -----------------------------
# arq integration sketch
# -----------------------------
async def arq_task_idle_discovery(ctx: Dict[str, Any]) -> str:
    """
    Example arq task. ctx should include LLM client/model.
    """
    llm_client = ctx["llm_client"]
    model = ctx["model"]
    github_token = ctx.get("github_token")

    await idle_skill_discovery_loop(
        client=llm_client,
        model=model,
        github_token=github_token,
        verify_before_save=True,
    )
    return "done"
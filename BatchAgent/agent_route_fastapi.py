from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import uuid
import zipfile
from urllib.parse import quote
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, FastAPI
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from BatchAgent.agent_services_fastapi import AgentService
from BatchAgent.tools_registry import get_base_tools

try:
    from BatchAgent.config import settings
except Exception:
    from config import settings


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _coerce_task_status(value: Any) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value
    raw = str(value or "").strip()
    if not raw:
        return TaskStatus.PENDING
    if raw.startswith("TaskStatus."):
        raw = raw.split(".", 1)[1].strip()
    normalized = raw.lower()
    for st in TaskStatus:
        if normalized == st.value or normalized == st.name.lower():
            return st
    return TaskStatus.PENDING


@dataclass
class TaskRecord:
    user_id: str
    task_id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    success: Optional[bool] = None
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    workspace_dir: Optional[str] = None
    session_dir: Optional[str] = None
    files: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    token_stats: Dict[str, Any] = field(default_factory=dict)
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


_AGENT_WORKSPACE_DIR = (_ROOT / "agent_workspace").resolve()
_USERS_ROOT_DIR = _AGENT_WORKSPACE_DIR / "users"
_USERS_ROOT_DIR.mkdir(parents=True, exist_ok=True)
_tasks: Dict[str, TaskRecord] = {}
_oauth2_optional = OAuth2PasswordBearer(tokenUrl="/api/auth/login/access-token", auto_error=False)


async def _get_current_user_optional(token: Annotated[Optional[str], Depends(_oauth2_optional)]) -> Dict[str, Any]:
    anonymous = {"id": "anonymous", "username": "anonymous", "_authenticated": False}
    token_str = str(token or "").strip()
    if not token_str:
        return anonymous
    try:
        payload = jwt.decode(token_str, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError:
        return anonymous
    except Exception:
        return anonymous
    username = str(payload.get("sub") or "").strip()
    if not username:
        return anonymous
    return {"id": username, "username": username, "_authenticated": True}


def _normalize_user_id(current_user: Optional[Dict[str, Any]]) -> str:
    if not current_user:
        return "anonymous"
    uid = current_user.get("id")
    if uid is not None and str(uid).strip():
        return str(uid).strip()
    username = current_user.get("username")
    if username is not None and str(username).strip():
        return str(username).strip()
    return "anonymous"


def _user_dir_name(user_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(user_id or "").strip())
    return safe or "anonymous"


def _task_cache_key(user_id: str, task_id: str) -> str:
    return f"{user_id}:{task_id}"


def _resolve_user_workspace_root(output_dir: str, user_id: str) -> str:
    _ = output_dir
    return str(_user_root_dir(user_id).resolve())


def _resolve_task_workspace_dir(output_dir: str, user_id: str, task_id: str) -> str:
    return str((Path(_resolve_user_workspace_root(output_dir, user_id)) / task_id).resolve())


def _user_root_dir(user_id: str) -> Path:
    d = _USERS_ROOT_DIR / _user_dir_name(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _agent_db_path(user_id: str) -> Path:
    return _user_root_dir(user_id) / "agent_sessions.db"


def _init_agent_db(user_id: str) -> None:
    db_path = _agent_db_path(user_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                task_id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                status TEXT NOT NULL,
                success INTEGER NULL,
                result TEXT NULL,
                error TEXT NULL,
                started_at TEXT NULL,
                finished_at TEXT NULL,
                workspace_dir TEXT NULL,
                session_dir TEXT NULL,
                files_json TEXT NULL,
                chat_history_json TEXT NULL,
                token_stats_json TEXT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(agent_sessions)").fetchall()}
        if "user_id" not in cols:
            conn.execute("ALTER TABLE agent_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'")
        if "token_stats_json" not in cols:
            conn.execute("ALTER TABLE agent_sessions ADD COLUMN token_stats_json TEXT NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_updated ON agent_sessions(user_id, updated_at DESC)")
        conn.commit()
    finally:
        conn.close()


def _upsert_task_db(record: TaskRecord) -> None:
    _init_agent_db(record.user_id)
    conn = sqlite3.connect(_agent_db_path(record.user_id))
    try:
        conn.execute(
            """
            INSERT INTO agent_sessions (
                user_id, task_id, goal, status, success, result, error, started_at, finished_at,
                workspace_dir, session_dir, files_json, chat_history_json, token_stats_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                user_id=excluded.user_id,
                goal=excluded.goal,
                status=excluded.status,
                success=excluded.success,
                result=excluded.result,
                error=excluded.error,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                workspace_dir=excluded.workspace_dir,
                session_dir=excluded.session_dir,
                files_json=excluded.files_json,
                chat_history_json=excluded.chat_history_json,
                token_stats_json=excluded.token_stats_json,
                updated_at=excluded.updated_at
            """,
            (
                record.user_id,
                record.task_id,
                record.goal,
                _coerce_task_status(record.status).value,
                None if record.success is None else (1 if record.success else 0),
                record.result,
                record.error,
                record.started_at,
                record.finished_at,
                record.workspace_dir,
                record.session_dir,
                json.dumps(record.files or [], ensure_ascii=False),
                json.dumps(record.chat_history or [], ensure_ascii=False),
                json.dumps(record.token_stats or {}, ensure_ascii=False),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _task_from_row(row: sqlite3.Row) -> TaskRecord:
    files = []
    history = []
    token_stats: Dict[str, Any] = {}
    try:
        files = json.loads(row["files_json"] or "[]")
    except Exception:
        files = []
    try:
        history = json.loads(row["chat_history_json"] or "[]")
    except Exception:
        history = []
    try:
        token_stats = json.loads(row["token_stats_json"] or "{}")
    except Exception:
        token_stats = {}
    success_val = row["success"]
    success: Optional[bool] = None if success_val is None else bool(success_val)
    return TaskRecord(
        user_id=str(row["user_id"] or "anonymous"),
        task_id=row["task_id"],
        goal=row["goal"],
        status=_coerce_task_status(row["status"]),
        success=success,
        result=row["result"],
        error=row["error"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        workspace_dir=row["workspace_dir"],
        session_dir=row["session_dir"],
        files=files if isinstance(files, list) else [],
        chat_history=history if isinstance(history, list) else [],
        token_stats=token_stats if isinstance(token_stats, dict) else {},
    )


def _load_task_from_db(task_id: str, user_id: str) -> Optional[TaskRecord]:
    db_path = _agent_db_path(user_id)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM agent_sessions WHERE task_id = ? AND user_id = ?", (task_id, user_id)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    rec = _task_from_row(row)
    _tasks[_task_cache_key(user_id, task_id)] = rec
    return rec


def _load_task_owner(task_id: str) -> Optional[str]:
    if not _USERS_ROOT_DIR.exists():
        return None
    for user_dir in sorted(_USERS_ROOT_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        db_path = user_dir / "agent_sessions.db"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT user_id FROM agent_sessions WHERE task_id = ?", (task_id,)).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            conn.close()
        if row:
            owner = str(row["user_id"] or "").strip()
            if owner:
                return owner
    return None


def _list_sessions_from_db(limit: int, user_id: str) -> List[Dict[str, Any]]:
    db_path = _agent_db_path(user_id)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM agent_sessions WHERE user_id = ? ORDER BY COALESCE(finished_at, started_at, updated_at) DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            rec = _task_from_row(row)
        except Exception:
            continue
        files = rec.files if rec.files else _build_workspace_file_entries(rec.workspace_dir, rec.task_id)
        if files and not rec.files:
            rec.files = files
            _persist(rec)
        out.append(
            {
                "task_id": rec.task_id,
                "goal": rec.goal,
                "status": rec.status,
                "success": rec.success,
                "result": rec.result,
                "error": rec.error,
                "started_at": rec.started_at,
                "finished_at": rec.finished_at,
                "workspace_dir": rec.workspace_dir,
                "session_dir": rec.session_dir,
                "token_stats": rec.token_stats or {},
                "files": files[:20],
            }
        )
    return out


def _delete_task_from_db(task_id: str, user_id: str) -> None:
    db_path = _agent_db_path(user_id)
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM agent_sessions WHERE task_id = ? AND user_id = ?", (task_id, user_id))
        conn.commit()
    finally:
        conn.close()


def _user_tasks_dir(user_id: str) -> Path:
    d = _user_root_dir(user_id) / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task_path(task_id: str, user_id: str) -> Path:
    return _user_tasks_dir(user_id) / f"{task_id}.json"


def _migrate_legacy_agent_storage() -> None:
    legacy_state_dir = (_ROOT / ".agent").resolve()
    legacy_db_path = legacy_state_dir / "agent_sessions.db"
    if legacy_db_path.exists():
        conn = sqlite3.connect(legacy_db_path)
        conn.row_factory = sqlite3.Row
        try:
            cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(agent_sessions)").fetchall()}
            if cols:
                rows = conn.execute("SELECT * FROM agent_sessions").fetchall()
            else:
                rows = []
        except sqlite3.Error:
            rows = []
            cols = set()
        finally:
            conn.close()
        for row in rows:
            raw_user_id = row["user_id"] if "user_id" in cols else "anonymous"
            user_id = str(raw_user_id or "").strip() or "anonymous"
            task_id = str(row["task_id"] if "task_id" in cols else "").strip()
            if not task_id:
                continue
            _init_agent_db(user_id)
            target_conn = sqlite3.connect(_agent_db_path(user_id))
            try:
                target_conn.execute(
                    """
                    INSERT INTO agent_sessions (
                        user_id, task_id, goal, status, success, result, error, started_at, finished_at,
                        workspace_dir, session_dir, files_json, chat_history_json, token_stats_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        user_id=excluded.user_id,
                        goal=excluded.goal,
                        status=excluded.status,
                        success=excluded.success,
                        result=excluded.result,
                        error=excluded.error,
                        started_at=excluded.started_at,
                        finished_at=excluded.finished_at,
                        workspace_dir=excluded.workspace_dir,
                        session_dir=excluded.session_dir,
                        files_json=excluded.files_json,
                        chat_history_json=excluded.chat_history_json,
                        token_stats_json=excluded.token_stats_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        user_id,
                        task_id,
                        str(row["goal"] if "goal" in cols else ""),
                        str(row["status"] if "status" in cols else "pending"),
                        row["success"] if "success" in cols else None,
                        row["result"] if "result" in cols else None,
                        row["error"] if "error" in cols else None,
                        row["started_at"] if "started_at" in cols else None,
                        row["finished_at"] if "finished_at" in cols else None,
                        row["workspace_dir"] if "workspace_dir" in cols else None,
                        row["session_dir"] if "session_dir" in cols else None,
                        row["files_json"] if "files_json" in cols else None,
                        row["chat_history_json"] if "chat_history_json" in cols else None,
                        row["token_stats_json"] if "token_stats_json" in cols else None,
                        str(row["updated_at"] if "updated_at" in cols else datetime.utcnow().isoformat()),
                    ),
                )
                target_conn.commit()
            finally:
                target_conn.close()

    legacy_tasks_root = legacy_state_dir / "tasks_by_user"
    if legacy_tasks_root.exists() and legacy_tasks_root.is_dir():
        for legacy_user_dir in legacy_tasks_root.iterdir():
            if not legacy_user_dir.is_dir():
                continue
            target_dir = _user_tasks_dir(legacy_user_dir.name)
            for src in legacy_user_dir.glob("*.json"):
                dst = target_dir / src.name
                if dst.exists():
                    continue
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    continue


def _persist(record: TaskRecord) -> None:
    _user_tasks_dir(record.user_id)
    data = {
        "user_id": record.user_id,
        "task_id": record.task_id,
        "goal": record.goal,
        "status": _coerce_task_status(record.status).value,
        "success": record.success,
        "result": record.result,
        "error": record.error,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "workspace_dir": record.workspace_dir,
        "session_dir": record.session_dir,
        "files": record.files,
        "chat_history": record.chat_history,
        "token_stats": record.token_stats,
    }
    _task_path(record.task_id, record.user_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
    _upsert_task_db(record)


def _load_task(task_id: str, user_id: str) -> Optional[TaskRecord]:
    db_rec = _load_task_from_db(task_id, user_id)
    if db_rec:
        return db_rec
    p = _task_path(task_id, user_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rec = TaskRecord(
            user_id=str(data.get("user_id") or user_id),
            task_id=data["task_id"],
            goal=data["goal"],
            status=_coerce_task_status(data.get("status")),
            success=data.get("success"),
            result=data.get("result"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            workspace_dir=data.get("workspace_dir"),
            session_dir=data.get("session_dir"),
            files=data.get("files") or [],
            chat_history=data.get("chat_history") or [],
            token_stats=data.get("token_stats") or {},
        )
        _tasks[_task_cache_key(user_id, task_id)] = rec
        return rec
    except Exception:
        return None


def _get_task(task_id: str, user_id: str) -> Optional[TaskRecord]:
    return _tasks.get(_task_cache_key(user_id, task_id)) or _load_task(task_id, user_id)


def _get_task_with_owner_fallback(task_id: str, user_id: str) -> Optional[TaskRecord]:
    record = _get_task(task_id, user_id)
    if record:
        return record
    owner = _load_task_owner(task_id)
    if not owner or owner == user_id:
        return None
    return _get_task(task_id, owner)


def _file_api_path(task_id: str, name: str) -> str:
    return f"/agent/sessions/{task_id}/files/raw?name={quote(name, safe='')}"


def _file_download_api_path(task_id: str, name: str) -> str:
    return f"/agent/sessions/{task_id}/files/download?name={quote(name, safe='')}"


def _extract_pdf_preview(data: bytes, limit: int = 4000) -> str:
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        chunks: List[str] = []
        for page in reader.pages[:8]:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                chunks.append(page_text)
            if len("\n\n".join(chunks)) >= limit:
                break
        return "\n\n".join(chunks)[:limit]
    except Exception:
        return ""


def _extract_docx_preview(data: bytes, limit: int = 4000) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml_content = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", xml_content)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:limit]
    except Exception:
        return ""


def _decode_text_preview(data: bytes, limit: int = 4000) -> str:
    candidates = ["utf-8", "utf-16", "utf-16le", "utf-16be", "gb18030", "gbk", "big5", "latin-1"]
    best = ""
    best_score = -1.0
    for enc in candidates:
        try:
            text = data.decode(enc)
        except Exception:
            continue
        snippet = text[:limit]
        if not snippet:
            continue
        replacement_ratio = snippet.count("\ufffd") / max(len(snippet), 1)
        printable = sum(1 for ch in snippet if ch.isprintable() or ch in "\n\r\t")
        printable_ratio = printable / max(len(snippet), 1)
        score = printable_ratio - (replacement_ratio * 2.0)
        if score > best_score:
            best = snippet
            best_score = score
    if best:
        return best
    return data.decode("utf-8", errors="replace")[:limit]


def _extract_file_preview(name: str, data: bytes, limit: int = 4000) -> str:
    ext = Path(name).suffix.lower()
    if ext in {
        ".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".xml",
        ".html", ".htm", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
        ".rs", ".cpp", ".c", ".h", ".sql", ".log", ".ini", ".toml", ".rtf",
    }:
        return _decode_text_preview(data, limit=limit)
    if ext == ".pdf":
        return _extract_pdf_preview(data, limit=limit)
    if ext == ".docx":
        return _extract_docx_preview(data, limit=limit)
    if ext in {".doc", ".ppt", ".pptx", ".xls", ".xlsx"}:
        return f"[{ext.lstrip('.').upper()} file attached. Preview is not available.]"
    return ""


def _resolve_existing_file_path(record: TaskRecord, name: str) -> Optional[Path]:
    target_name = str(name or "").strip()
    if not target_name:
        return None
    target_lower = target_name.lower()
    local_candidate = None
    for item in record.files or []:
        item_name = str(item.get("name") or "").strip()
        if item_name.lower() == target_lower:
            local_candidate = str(item.get("local_path") or "").strip()
            break
    if local_candidate:
        p = Path(local_candidate)
        if p.exists() and p.is_file():
            return p
    if record.workspace_dir:
        p = (Path(record.workspace_dir) / target_name).resolve()
        if p.exists() and p.is_file():
            return p
    candidate_dirs: List[Path] = []
    for item in record.files or []:
        local_path = str(item.get("local_path") or "").strip()
        if local_path:
            candidate_dirs.append(Path(local_path).parent)
    if record.workspace_dir:
        ws = Path(record.workspace_dir)
        candidate_dirs.append(ws)
        candidate_dirs.append(ws.parent)
    if record.session_dir:
        session_dir = Path(record.session_dir)
        candidate_dirs.append(session_dir)
        candidate_dirs.append(session_dir.parent)
    user_root = _user_root_dir(record.user_id)
    candidate_dirs.append(user_root)
    candidate_dirs.append(user_root / record.task_id)
    candidate_dirs.append(_AGENT_WORKSPACE_DIR)
    candidate_dirs.append(_USERS_ROOT_DIR)
    seen_dirs = set()
    for base in candidate_dirs:
        if not base.exists() or not base.is_dir():
            continue
        try:
            resolved = str(base.resolve())
        except Exception:
            resolved = str(base)
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        exact = base / target_name
        if exact.exists() and exact.is_file():
            return exact
        try:
            for child in base.iterdir():
                if not child.is_file():
                    continue
                if child.name.lower() == target_lower:
                    return child
        except Exception:
            pass
    seen_roots = set()
    for root in candidate_dirs:
        if not root.exists() or not root.is_dir():
            continue
        try:
            root_resolved = str(root.resolve())
        except Exception:
            root_resolved = str(root)
        if root_resolved in seen_roots:
            continue
        seen_roots.add(root_resolved)
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.lower() == target_lower:
                    return p
        except Exception:
            continue
    return None


def _normalize_file_entry(task_id: str, entry: Dict[str, Any], workspace_dir: Optional[str]) -> Dict[str, Any]:
    normalized = dict(entry)
    name = str(normalized.get("name") or "").strip()
    ext = str(normalized.get("ext") or Path(name).suffix.lstrip(".")).lower()
    normalized["ext"] = ext
    if name:
        normalized["file_api_path"] = _file_api_path(task_id, name)
        normalized["download_api_path"] = _file_download_api_path(task_id, name)
    content = normalized.get("content")
    if content is None or content == "":
        local_path = str(normalized.get("local_path") or "").strip()
        path_obj: Optional[Path] = None
        if local_path:
            p = Path(local_path)
            if p.exists() and p.is_file():
                path_obj = p
        if path_obj is None and workspace_dir and name:
            p = Path(workspace_dir) / name
            if p.exists() and p.is_file():
                path_obj = p
                normalized["local_path"] = str(p)
        if path_obj is not None:
            try:
                raw = path_obj.read_bytes()
                normalized["content"] = _extract_file_preview(name, raw)
                normalized["size"] = path_obj.stat().st_size
            except Exception:
                pass
    return normalized


def _build_workspace_file_entries(workspace: Optional[str], task_id: str) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    if not workspace:
        return files
    ws = Path(workspace)
    if not ws.exists():
        return files

    from BatchAgent.minio_uploader import upload_file

    for f in sorted(ws.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not f.is_file() or f.name.startswith(".") or f.suffix == ".db":
            continue
        ext = f.suffix.lstrip(".").lower()
        object_name = f"agent/{task_id}/{f.name}"
        url = upload_file(str(f), object_name=object_name) or ""
        files.append(
            {
                "name": f.name,
                "ext": ext,
                "size": f.stat().st_size,
                "url": url,
                "content": _extract_file_preview(f.name, f.read_bytes()),
                "local_path": str(f),
                "file_api_path": _file_api_path(task_id, f.name),
                "download_api_path": _file_download_api_path(task_id, f.name),
            }
        )
    return files


def _init_minio():
    import yaml

    cfg_path = os.getenv("APP_CONFIG_PATH", "backend/config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        mc = cfg.get("minio", {})
        if mc.get("endpoint"):
            from BatchAgent.minio_uploader import init_minio

            init_minio(
                endpoint=mc["endpoint"],
                access_key=mc.get("access_key", ""),
                secret_key=mc.get("secret_key", ""),
                bucket=mc.get("bucket", "aiagent"),
                secure=mc.get("secure", False),
            )
            from BatchAgent.minio_uploader import _client as _mc_check

            if _mc_check is not None:
                print(f"[MinIO] Ready → {mc['endpoint']}/{mc.get('bucket','aiagent')}")
    except Exception as exc:
        print(f"[MinIO] Config load failed: {exc}")


_init_minio()
_migrate_legacy_agent_storage()

router = APIRouter(tags=["Agent"])


@router.get("/agent/health", tags=["Meta"])
def health():
    return {"status": "ok", "service": "BatchAgent"}


class RunRequest(BaseModel):
    task_id: Optional[str] = None
    goal: str = Field(..., description="Natural language task goal")
    tool_strategy: str = Field("hybrid", description="'native_all' | 'hybrid' | 'text_only'")
    provider: str = Field("openai", description="'openai' or 'anthropic'")
    model: str = Field(default_factory=lambda: os.environ.get("VLLM_MODEL", "qwen3.5-9b"))
    base_url: str = Field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1"))
    api_key: str = Field(default_factory=lambda: os.environ.get("VLLM_API_KEY", "EMPTY"))
    max_context: int = 16384
    max_output: int = 4096
    verbose: bool = False
    output_dir: str = "./agent_workspace"
    domain: str = "general"
    location: str = "California, United States"
    serper_api_key: str = Field(default_factory=lambda: os.environ.get("SERPER_API_KEY", ""))
    tavily_api_key: str = Field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    enable_youtube: bool = Field(False, description="Enable YouTube search (costs an extra Serper credit per call)")
    allowlist: List[str] = []
    temperature: float = Field(0.1, description="LLM temperature, lower is better for strict format")
    memory_strategy: str = Field("sliding_window", description="'sliding_window' or 'summarize'")
    enable_turn_limits: bool = Field(True, description="Expose current and max turns in prompt")
    max_turns: int = Field(15, description="Maximum ReAct turns before aborting")
    backend: str = Field(default_factory=lambda: os.environ.get("VLLM_BACKEND", "vllm"))
    enable_thinking: bool = Field(default_factory=lambda: os.environ.get("ENABLE_THINKING", "true").lower() == "true")
    context_resources: List[Dict[str, Any]] = Field(default_factory=list)


class RunResponse(BaseModel):
    task_id: str
    success: bool
    status: str
    result: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    workspace_dir: Optional[str] = None
    session_dir: Optional[str] = None


class StatusResponse(BaseModel):
    task_id: str
    status: str
    goal: str
    success: Optional[bool] = None
    result: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    workspace_dir: Optional[str] = None
    session_dir: Optional[str] = None
    token_stats: Optional[Dict[str, Any]] = None


class SaveFileRequest(BaseModel):
    name: str
    content: str


class AttachResourceRequest(BaseModel):
    url: str


class InitSessionRequest(BaseModel):
    task_id: Optional[str] = None
    goal: str = ""


def _make_service(req: RunRequest) -> AgentService:
    return AgentService(
        base_url=req.base_url,
        api_key=req.api_key,
        model=req.model,
        provider=req.provider,
        model_max_context=req.max_context,
        max_output_tokens=req.max_output,
        tool_strategy=req.tool_strategy,
        domain=req.domain,
        location=req.location,
        output_dir=req.output_dir,
        verbose=req.verbose,
        serper_api_key=req.serper_api_key,
        tavily_api_key=req.tavily_api_key,
        enable_youtube=req.enable_youtube,
        temperature=req.temperature,
        memory_strategy=req.memory_strategy,
        enable_turn_limits=req.enable_turn_limits,
        max_turns=req.max_turns,
        backend=req.backend,
        enable_thinking=req.enable_thinking,
    )


def _finalize_token_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    prompt = int(stats.get("total_prompt_tokens", 0) or 0)
    completion = int(stats.get("total_completion_tokens", 0) or 0)
    elapsed = float(stats.get("total_elapsed_s", 0.0) or 0.0)
    speed = (completion / elapsed) if elapsed > 0 else 0.0
    return {
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_elapsed_s": elapsed,
        "output_tokens_per_sec": speed,
    }


def _clean_context_text(value: Any, *, limit: int = 1200) -> str:
    text = str(value or "").replace("\x00", "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _build_goal_with_context(goal: str, context_resources: List[Dict[str, Any]]) -> str:
    normalized_goal = str(goal or "").strip()
    resources = context_resources if isinstance(context_resources, list) else []
    compact_lines: List[str] = []
    for idx, item in enumerate(resources, start=1):
        if not isinstance(item, dict):
            continue
        title = _clean_context_text(
            item.get("title")
            or item.get("name")
            or item.get("resource_title")
            or item.get("source_url")
            or item.get("url")
            or f"resource {idx}",
            limit=180,
        )
        source_url = _clean_context_text(item.get("source_url") or item.get("url"), limit=500)
        description = _clean_context_text(item.get("description") or item.get("summary"), limit=500)
        transcript = _clean_context_text(item.get("transcript"), limit=700)
        content = _clean_context_text(item.get("content"), limit=1000)
        include_all = bool(item.get("include_all") or item.get("is_short"))
        lines = [f"{idx}. Title: {title}"]
        if source_url:
            lines.append(f"   URL: {source_url}")
        if description:
            lines.append(f"   Description: {description}")
        if transcript and include_all:
            lines.append(f"   Transcript: {transcript}")
        elif transcript:
            lines.append(f"   Transcript: {transcript[:240].rstrip()}…")
        if content and include_all:
            lines.append(f"   Content: {content}")
        elif content:
            lines.append(f"   Content: {content[:240].rstrip()}…")
        if source_url and not include_all:
            lines.append(f"   Access: Use read_url with this URL for full details.")
        compact_lines.append("\n".join(lines))
        if len(compact_lines) >= 15:
            break
    if not compact_lines:
        return normalized_goal
    context_block = (
        "[Context Resources]\n"
        "Treat these as available resources, similar to web_search results.\n"
        + "\n\n".join(compact_lines)
    )
    return f"{normalized_goal}\n\n{context_block}".strip()


def _load_resume_payload(record: TaskRecord) -> Dict[str, Any]:
    resume_messages: List[Dict[str, str]] = []
    resume_rl_trajectory: List[Dict[str, str]] = []
    start_turn_index = 0
    session_dir_path: Optional[Path] = None

    if record.session_dir:
        p = Path(record.session_dir)
        if p.exists() and p.is_dir():
            session_dir_path = p

    if session_dir_path is not None:
        try:
            turn_indices = [int(d.name) for d in session_dir_path.iterdir() if d.is_dir() and d.name.isdigit()]
            if turn_indices:
                start_turn_index = max(turn_indices) + 1
        except Exception:
            start_turn_index = 0
        rl_path = session_dir_path / "rl_trajectory.jsonl"
        if rl_path.exists() and rl_path.is_file():
            try:
                lines = [ln for ln in rl_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
                if lines:
                    payload = json.loads(lines[-1])
                    messages = payload.get("messages") or []
                    if isinstance(messages, list):
                        for m in messages:
                            if not isinstance(m, dict):
                                continue
                            role = str(m.get("role", "")).strip()
                            if role not in {"system", "user", "assistant"}:
                                continue
                            content = str(m.get("content", ""))
                            resume_messages.append({"role": role, "content": content})
                        resume_rl_trajectory = list(resume_messages)
            except Exception:
                resume_messages = []
                resume_rl_trajectory = []

    return {
        "continue_session_dir": str(session_dir_path) if session_dir_path is not None else None,
        "continue_workspace_dir": record.workspace_dir,
        "resume_messages": resume_messages or None,
        "resume_rl_trajectory": resume_rl_trajectory or None,
        "start_turn_index": start_turn_index,
    }


async def _run_blocking(record: TaskRecord, req: RunRequest) -> bool:
    record.status = TaskStatus.RUNNING
    record.started_at = datetime.utcnow().isoformat()
    _persist(record)
    try:
        svc = _make_service(req)
        resume_payload = _load_resume_payload(record)
        effective_goal = _build_goal_with_context(req.goal, req.context_resources)
        success, session_dir, workspace_dir, final_result = await svc.run(
            goal=effective_goal,
            allowlist=req.allowlist,
            continue_session_dir=resume_payload["continue_session_dir"],
            continue_workspace_dir=resume_payload["continue_workspace_dir"],
            resume_messages=resume_payload["resume_messages"],
            resume_rl_trajectory=resume_payload["resume_rl_trajectory"],
            start_turn_index=resume_payload["start_turn_index"],
        )
        record.success = success
        record.result = final_result
        record.status = TaskStatus.DONE
        record.session_dir = str(session_dir)
        record.workspace_dir = str(workspace_dir)
    except Exception as exc:
        record.success = False
        record.error = str(exc)
        record.status = TaskStatus.FAILED
    finally:
        record.finished_at = datetime.utcnow().isoformat()
        _persist(record)
    return record.success or False


async def _run_streaming(
    record: TaskRecord,
    req: RunRequest,
    on_event: Callable[[Dict[str, Any]], Awaitable[None]],
) -> bool:
    record.status = TaskStatus.RUNNING
    record.started_at = datetime.utcnow().isoformat()
    _persist(record)
    try:
        svc = _make_service(req)
        resume_payload = _load_resume_payload(record)
        effective_goal = _build_goal_with_context(req.goal, req.context_resources)
        success, session_dir, workspace_dir, final_result = await svc.run_stream(
            goal=effective_goal,
            on_event=on_event,
            allowlist=req.allowlist,
            continue_session_dir=resume_payload["continue_session_dir"],
            continue_workspace_dir=resume_payload["continue_workspace_dir"],
            resume_messages=resume_payload["resume_messages"],
            resume_rl_trajectory=resume_payload["resume_rl_trajectory"],
            start_turn_index=resume_payload["start_turn_index"],
        )
        record.success = success
        record.result = final_result
        record.status = TaskStatus.DONE
        record.session_dir = str(session_dir)
        record.workspace_dir = str(workspace_dir)
    except Exception as exc:
        record.success = False
        record.error = str(exc)
        record.status = TaskStatus.FAILED
        _persist(record)
        raise exc
    finally:
        record.finished_at = datetime.utcnow().isoformat()
        _persist(record)
    return record.success or False


@router.post("/agent/run", response_model=RunResponse)
async def agent_run(
    req: RunRequest,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    is_authenticated = bool(current_user.get("_authenticated"))
    task_id = str(req.task_id or "").strip() or str(uuid.uuid4())
    owner = _load_task_owner(task_id)
    if owner and owner != user_id:
        if is_authenticated:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        user_id = owner
    record = _get_task(task_id, user_id)
    if record is None:
        record = TaskRecord(user_id=user_id, task_id=task_id, goal=req.goal)
        record.workspace_dir = _resolve_task_workspace_dir(req.output_dir, user_id, task_id)
    else:
        record.goal = req.goal
    req.output_dir = record.workspace_dir or _resolve_task_workspace_dir(req.output_dir, user_id, task_id)
    _tasks[_task_cache_key(user_id, task_id)] = record

    await _run_blocking(record, req)

    return RunResponse(
        task_id=task_id,
        success=record.success or False,
        status=record.status,
        result=record.result,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
        workspace_dir=record.workspace_dir,
        session_dir=record.session_dir,
    )


@router.get("/agent/tools", summary="Get active tools available for the agent")
def get_active_tools(strategy: str = "native_all"):
    tools = get_base_tools(strategy=strategy)
    return {"tools": tools}


@router.post("/agent/sessions/init", summary="Initialize an empty task session")
def init_session(
    req: InitSessionRequest,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    is_authenticated = bool(current_user.get("_authenticated"))
    task_id = str(req.task_id or "").strip() or str(uuid.uuid4())
    owner = _load_task_owner(task_id)
    if owner and owner != user_id:
        if is_authenticated:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        user_id = owner
    record = _get_task(task_id, user_id)
    if record is None:
        goal = str(req.goal or "").strip()
        record = TaskRecord(user_id=user_id, task_id=task_id, goal=goal)
    elif str(req.goal or "").strip():
        record.goal = str(req.goal or "").strip()
    workspace_dir = Path(record.workspace_dir or _resolve_task_workspace_dir("./agent_workspace", user_id, task_id)).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    record.workspace_dir = str(workspace_dir)
    _tasks[_task_cache_key(user_id, task_id)] = record
    _persist(record)
    return {
        "task_id": task_id,
        "status": record.status,
        "goal": record.goal,
        "workspace_dir": record.workspace_dir,
        "files": record.files or [],
    }


@router.post("/agent/stream", summary="Stream agent task via SSE")
async def agent_stream(
    req: RunRequest,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    is_authenticated = bool(current_user.get("_authenticated"))
    task_id = str(req.task_id or "").strip() or str(uuid.uuid4())
    owner = _load_task_owner(task_id)
    if owner and owner != user_id:
        if is_authenticated:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        user_id = owner
    record = _get_task(task_id, user_id)
    if record is None:
        record = TaskRecord(user_id=user_id, task_id=task_id, goal=req.goal)
        record.workspace_dir = _resolve_task_workspace_dir(req.output_dir, user_id, task_id)
    else:
        record.goal = req.goal
    req.output_dir = record.workspace_dir or _resolve_task_workspace_dir(req.output_dir, user_id, task_id)
    _tasks[_task_cache_key(user_id, task_id)] = record
    queue: asyncio.Queue[Optional[str]] = record.token_queue

    token_stats: Dict[str, Any] = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_elapsed_s": 0.0}
    stream_messages: List[Dict[str, Any]] = []
    stream_files_by_name: Dict[str, Dict[str, Any]] = {}

    async def on_event(event: Dict[str, Any]) -> None:
        if event.get("type") == "usage":
            token_stats["total_prompt_tokens"] += event.get("prompt_tokens", 0)
            token_stats["total_completion_tokens"] += event.get("completion_tokens", 0)
            token_stats["total_elapsed_s"] += event.get("elapsed_s", 0.0)
            return
        ev_type = event.get("type")
        if ev_type in {"message", "token", "think", "delta"}:
            stream_messages.append({"type": ev_type, "data": event.get("data", "")})
        elif ev_type == "turn_start":
            stream_messages.append({"type": "turn_start", "turn": event.get("turn"), "max_turns": event.get("max_turns")})
        elif ev_type == "tool":
            tool_name = str(event.get("name") or "")
            tool_status = str(event.get("status") or "")
            if tool_name and tool_name != "finish_task" and tool_status == "started":
                stream_messages.append({"type": "tool", "name": tool_name, "status": tool_status})
        elif ev_type == "file_written":
            name = str(event.get("name", "")).strip()
            if name:
                file_api_path = _file_api_path(task_id, name)
                download_api_path = _file_download_api_path(task_id, name)
                event["file_api_path"] = file_api_path
                event["download_api_path"] = download_api_path
                local_path = str(event.get("local_path") or "")
                file_entry = {
                    "name": name,
                    "ext": Path(name).suffix.lstrip(".").lower(),
                    "size": int(event.get("size", 0) or 0),
                    "url": event.get("url") or "",
                    "content": event.get("content") or "",
                    "source_url": event.get("source_url") or "",
                    "local_path": local_path,
                    "file_api_path": file_api_path,
                    "download_api_path": download_api_path,
                }
                stream_files_by_name[name] = file_entry
                existing_by_name: Dict[str, Dict[str, Any]] = {}
                for item in record.files or []:
                    item_name = str(item.get("name") or "").strip()
                    if item_name:
                        existing_by_name[item_name] = item
                existing_by_name[name] = file_entry
                record.files = list(existing_by_name.values())
                if not record.workspace_dir and local_path:
                    try:
                        record.workspace_dir = str(Path(local_path).resolve().parent)
                    except Exception:
                        pass
                _persist(record)
                stream_messages.append(
                    {
                        "type": "file_written",
                        "name": name,
                        "url": event.get("url") or "",
                        "content": event.get("content") or "",
                        "file_api_path": file_api_path,
                        "download_api_path": download_api_path,
                        "source_url": event.get("source_url") or "",
                    }
                )
        elif ev_type == "done":
            stream_messages.append(
                {
                    "type": "done",
                    "success": bool(event.get("success", False)),
                    "stats": _finalize_token_stats(token_stats),
                    "isFinalSummary": False,
                }
            )
        elif ev_type == "error":
            stream_messages.append({"type": "error", "detail": event.get("detail", "")})
        payload = json.dumps(event)
        await queue.put(f"data: {payload}\n\n")

    async def producer() -> None:
        try:
            intro = json.dumps({"type": "start", "task_id": task_id, "goal": req.goal})
            await queue.put(f"data: {intro}\n\n")

            success = await _run_streaming(record, req, on_event=on_event)
            finalized_stats = _finalize_token_stats(token_stats)
            if stream_files_by_name:
                record.files = list(stream_files_by_name.values())
            if stream_messages:
                record.chat_history = stream_messages
            record.token_stats = finalized_stats
            if not record.files:
                record.files = _build_workspace_file_entries(record.workspace_dir, record.task_id)
            _persist(record)

            done = json.dumps(
                {
                    "type": "done",
                    "success": success,
                    "task_id": task_id,
                    "result": record.result,
                    "stats": finalized_stats,
                    "isFinalSummary": True,
                    "files": record.files,
                }
            )
            await queue.put(f"data: {done}\n\n")
        except Exception as exc:
            import traceback

            trace_str = traceback.format_exc()
            print(f"!!! CRITICAL PRODUCER EXCEPTION !!!\n{trace_str}")
            err = json.dumps({"type": "error", "detail": f"{str(exc)}\n{trace_str}"})
            await queue.put(f"data: {err}\n\n")
        finally:
            await queue.put(None)

    async def event_generator():
        asyncio.create_task(producer())
        yield ":" + (" " * 2048) + "\n\n"
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/status/{task_id}", response_model=StatusResponse)
def agent_status(
    task_id: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    record = _get_task(task_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return StatusResponse(
        task_id=record.task_id,
        status=record.status,
        goal=record.goal,
        success=record.success,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error=record.error,
        workspace_dir=record.workspace_dir,
        session_dir=record.session_dir,
        token_stats=record.token_stats or {},
    )


def _parse_response_md(content: str) -> dict:
    think_matches = re.findall(r"<think>([\s\S]*?)</think>", content, re.DOTALL)
    thinking = "\n".join(m.strip() for m in think_matches if m.strip())

    text = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.DOTALL).strip()
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```json[\s\S]*?```", "", text, flags=re.DOTALL).strip()
    clean_lines, brace_depth = [], 0
    for line in text.splitlines():
        s = line.strip()
        if brace_depth == 0:
            if s.startswith("{") and ('"name"' in s or '"function"' in s):
                brace_depth += s.count("{") - s.count("}")
                continue
            clean_lines.append(line)
        else:
            brace_depth += s.count("{") - s.count("}")
            if brace_depth <= 0:
                brace_depth = 0
    text = "\n".join(clean_lines)
    text = re.sub(r"(?m)^-{3,}\s*$", "", text).strip()
    return {"thinking": thinking, "text": text}


@router.get("/agent/sessions/{task_id}/turns", summary="Get per-turn logs for a session")
def get_session_turns(
    task_id: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    turns = []
    session_dir = Path(record.session_dir) if record.session_dir else None
    if session_dir and session_dir.exists():
        for turn_dir in sorted(session_dir.iterdir()):
            if not (turn_dir.is_dir() and turn_dir.name.isdigit()):
                continue
            turn_num = int(turn_dir.name)
            parsed: dict = {"turn": turn_num}

            resp_f = turn_dir / "response.md"
            if resp_f.exists():
                parsed.update(_parse_response_md(resp_f.read_text(encoding="utf-8")))

            act_f = turn_dir / "parsed_actions.json"
            if act_f.exists():
                try:
                    parsed["actions"] = json.loads(act_f.read_text(encoding="utf-8"))
                except Exception:
                    parsed["actions"] = []

            turns.append(parsed)

    files_raw = list(record.files) if record.files else _build_workspace_file_entries(record.workspace_dir, record.task_id)
    files = [_normalize_file_entry(record.task_id, f, record.workspace_dir) for f in files_raw]
    if not record.files and files:
        record.files = files
        _persist(record)

    return {
        "task_id": task_id,
        "goal": record.goal,
        "status": record.status,
        "success": record.success,
        "result": record.result,
        "total_turns": len(turns),
        "turns": turns,
        "chat_history": record.chat_history,
        "token_stats": record.token_stats or {},
        "files": files[:20],
    }


@router.get("/agent/sessions", summary="List recent agent task sessions")
def list_sessions(
    limit: int = 50,
    current_user: Annotated[dict, Depends(_get_current_user_optional)] = None,
):
    user_id = _normalize_user_id(current_user)
    db_records = _list_sessions_from_db(limit, user_id)
    if db_records:
        for rec in db_records:
            task_id = str(rec.get("task_id") or "").strip()
            workspace_dir = rec.get("workspace_dir")
            rec["files"] = [_normalize_file_entry(task_id, f, workspace_dir) for f in (rec.get("files") or [])][:20]
        return {"sessions": db_records}
    records = []
    user_tasks_dir = _user_tasks_dir(user_id)
    if user_tasks_dir.exists():
        paths = sorted(user_tasks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                task_id = str(data.get("task_id", "")).strip()
                if data.get("files"):
                    files = data.get("files") or []
                else:
                    files = _build_workspace_file_entries(data.get("workspace_dir"), task_id) if task_id else []
                    if files:
                        data["files"] = files
                        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                data["files"] = [_normalize_file_entry(task_id, f, data.get("workspace_dir")) for f in files][:20]
                records.append(data)
            except Exception:
                pass
    return {"sessions": records}


@router.get("/agent/sessions/{task_id}/files/content", summary="Get session file content")
def get_session_file_content(
    task_id: str,
    name: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    raw_name = str(name or "").strip()
    target_name = Path(raw_name).name.strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Missing file name")
    if target_name != raw_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    def _cache_found_file(path: Path, text: str) -> None:
        files_by_name: Dict[str, Dict[str, Any]] = {}
        for item in record.files or []:
            item_name = str(item.get("name") or "").strip()
            if item_name:
                files_by_name[item_name] = item
        files_by_name[target_name] = {
            "name": target_name,
            "ext": path.suffix.lstrip(".").lower(),
            "size": path.stat().st_size,
            "url": files_by_name.get(target_name, {}).get("url", ""),
            "content": text,
            "local_path": str(path),
            "file_api_path": _file_api_path(task_id, target_name),
        }
        record.files = list(files_by_name.values())
        if not record.workspace_dir:
            record.workspace_dir = str(path.parent)
        _persist(record)

    candidate_dirs: List[Path] = []
    for f in (record.files or []):
        entry_name = str(f.get("name") or "").strip()
        local_path = str(f.get("local_path") or "").strip()
        if local_path:
            p = Path(local_path)
            if p.exists() and p.is_file():
                same_entry = entry_name and entry_name.lower() == target_name.lower()
                same_basename = p.name.lower() == target_name.lower()
                if same_entry or same_basename:
                    content = f.get("content")
                    if isinstance(content, str) and content != "":
                        return {"task_id": task_id, "name": target_name, "content": content}
                    try:
                        text = p.read_text(encoding="utf-8", errors="replace")
                        _cache_found_file(p, text)
                        return {"task_id": task_id, "name": target_name, "content": text}
                    except Exception:
                        pass
                candidate_dirs.append(p.parent)
        if entry_name and entry_name.lower() == target_name.lower():
            content = f.get("content")
            if isinstance(content, str) and content != "":
                return {"task_id": task_id, "name": target_name, "content": content}

    if record and record.workspace_dir:
        workspace_path = Path(record.workspace_dir)
        candidate_dirs.append(workspace_path)
        candidate_dirs.append(workspace_path.parent)
    if record and record.session_dir:
        session_path = Path(record.session_dir)
        candidate_dirs.append(session_path)
        candidate_dirs.append(session_path.parent)
    user_root = _user_root_dir(user_id)
    candidate_dirs.append(user_root)
    candidate_dirs.append(user_root / task_id)
    candidate_dirs.append(_AGENT_WORKSPACE_DIR)
    candidate_dirs.append(_USERS_ROOT_DIR)

    seen_dirs = set()
    for base in candidate_dirs:
        if not base.exists() or not base.is_dir():
            continue
        try:
            resolved = str(base.resolve())
        except Exception:
            resolved = str(base)
        if resolved in seen_dirs:
            continue
        seen_dirs.add(resolved)
        p = base / target_name
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            _cache_found_file(p, text)
            return {"task_id": task_id, "name": target_name, "content": text}

    search_roots: List[Path] = candidate_dirs
    seen_roots = set()
    for root in search_roots:
        try:
            root_resolved = str(root.resolve())
        except Exception:
            root_resolved = str(root)
        if root_resolved in seen_roots:
            continue
        seen_roots.add(root_resolved)
        if not root.exists() or not root.is_dir():
            continue
        found_path: Optional[Path] = None
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.lower() == target_name.lower():
                    found_path = p
                    break
        except Exception:
            found_path = None
        if found_path is None:
            continue
        text = found_path.read_text(encoding="utf-8", errors="replace")
        _cache_found_file(found_path, text)
        return {"task_id": task_id, "name": target_name, "content": text}

    raise HTTPException(status_code=404, detail=f"File '{target_name}' not found for task '{task_id}'")


@router.put("/agent/sessions/{task_id}/files/content", summary="Save session file content")
def save_session_file_content(
    task_id: str,
    req: SaveFileRequest,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    record = _get_task(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    raw_name = str(req.name or "").strip()
    target_name = Path(raw_name).name.strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Missing file name")
    if target_name != raw_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    workspace_dir = Path(record.workspace_dir or _resolve_task_workspace_dir("./agent_workspace", record.user_id, task_id)).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    target_path = (workspace_dir / target_name).resolve()
    target_path.write_text(str(req.content or ""), encoding="utf-8")

    url = ""
    try:
        from BatchAgent.minio_uploader import upload_file
        url = upload_file(str(target_path), object_name=f"agent/{task_id}/{target_name}") or ""
    except Exception:
        url = ""

    files_by_name: Dict[str, Dict[str, Any]] = {}
    for item in record.files or []:
        item_name = str(item.get("name") or "").strip()
        if item_name:
            files_by_name[item_name] = item
    prev = files_by_name.get(target_name, {})
    files_by_name[target_name] = {
        "name": target_name,
        "ext": target_path.suffix.lstrip(".").lower(),
        "size": target_path.stat().st_size,
        "url": url or str(prev.get("url") or ""),
        "content": str(req.content or ""),
        "local_path": str(target_path),
        "file_api_path": _file_api_path(task_id, target_name),
        "download_api_path": _file_download_api_path(task_id, target_name),
    }
    record.files = list(files_by_name.values())
    record.workspace_dir = str(workspace_dir)
    _persist(record)
    return {
        "task_id": task_id,
        "name": target_name,
        "saved": True,
        "file_api_path": _file_api_path(task_id, target_name),
        "download_api_path": _file_download_api_path(task_id, target_name),
        "url": url,
    }


@router.post("/agent/sessions/{task_id}/files/upload", summary="Upload attached file")
async def upload_session_file(
    task_id: str,
    file: UploadFile = File(...),
    current_user: Annotated[dict, Depends(_get_current_user_optional)] = None,
):
    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    raw_name = str(file.filename or "").strip()
    target_name = Path(raw_name).name.strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Missing file name")
    if target_name != raw_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    workspace_dir = Path(record.workspace_dir or _resolve_task_workspace_dir("./agent_workspace", record.user_id, task_id)).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    target_path = (workspace_dir / target_name).resolve()
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    target_path.write_bytes(data)
    content_preview = _extract_file_preview(target_name, data, limit=4000)
    url = ""
    try:
        from BatchAgent.minio_uploader import upload_file

        url = upload_file(str(target_path), object_name=f"agent/{task_id}/{target_name}") or ""
    except Exception:
        url = ""
    files_by_name: Dict[str, Dict[str, Any]] = {}
    for item in record.files or []:
        item_name = str(item.get("name") or "").strip()
        if item_name:
            files_by_name[item_name] = item
    prev = files_by_name.get(target_name, {})
    files_by_name[target_name] = {
        "name": target_name,
        "ext": target_path.suffix.lstrip(".").lower(),
        "size": target_path.stat().st_size,
        "url": url or str(prev.get("url") or ""),
        "content": content_preview,
        "local_path": str(target_path),
        "file_api_path": _file_api_path(task_id, target_name),
        "download_api_path": _file_download_api_path(task_id, target_name),
    }
    record.files = list(files_by_name.values())
    record.workspace_dir = str(workspace_dir)
    _persist(record)
    return {
        "task_id": task_id,
        "name": target_name,
        "saved": True,
        "size": target_path.stat().st_size,
        "file_api_path": _file_api_path(task_id, target_name),
        "download_api_path": _file_download_api_path(task_id, target_name),
        "url": url,
        "snapshot": content_preview[:500],
    }


@router.post("/agent/sessions/{task_id}/resources/attach", summary="Attach a URL resource")
def attach_session_resource(
    task_id: str,
    req: AttachResourceRequest,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    source_url = str(req.url or "").strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="Missing URL")
    if not re.match(r"^https?://", source_url, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    resource_type = "youtube" if re.search(r"(youtube\.com|youtu\.be)", source_url, re.IGNORECASE) else "web"
    try:
        from BatchAgent.tool_handler import fetch_and_parse_url

        parsed_content = fetch_and_parse_url(source_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse URL: {exc}") from exc
    if not parsed_content or parsed_content.startswith("HTTP Error") or parsed_content.startswith("Network error"):
        raise HTTPException(status_code=400, detail="Could not fetch or parse the URL")
    resource_title = ""
    title_match = re.search(r"^Title:\s*(.+?)\s*$", parsed_content, flags=re.MULTILINE)
    if title_match:
        resource_title = title_match.group(1).strip()
    workspace_dir = Path(record.workspace_dir or _resolve_task_workspace_dir("./agent_workspace", record.user_id, task_id)).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    resource_name = f"resource_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}.md"
    target_path = (workspace_dir / resource_name).resolve()
    heading = "# 🎬 Attached YouTube Resource" if resource_type == "youtube" else "# 🌐 Attached Web Resource"
    title_line = f"{heading}\n\nSource: {source_url}\n\n"
    target_path.write_text(title_line + parsed_content, encoding="utf-8")
    url = ""
    try:
        from BatchAgent.minio_uploader import upload_file

        url = upload_file(str(target_path), object_name=f"agent/{task_id}/{resource_name}") or ""
    except Exception:
        url = ""
    files_by_name: Dict[str, Dict[str, Any]] = {}
    for item in record.files or []:
        item_name = str(item.get("name") or "").strip()
        if item_name:
            files_by_name[item_name] = item
    files_by_name[resource_name] = {
        "name": resource_name,
        "ext": "md",
        "size": target_path.stat().st_size,
        "url": url,
        "content": (title_line + parsed_content)[:4000],
        "source_url": source_url,
        "resource_type": resource_type,
        "resource_title": resource_title,
        "local_path": str(target_path),
        "file_api_path": _file_api_path(task_id, resource_name),
        "download_api_path": _file_download_api_path(task_id, resource_name),
    }
    record.files = list(files_by_name.values())
    record.workspace_dir = str(workspace_dir)
    _persist(record)
    return {
        "task_id": task_id,
        "name": resource_name,
        "saved": True,
        "source_url": source_url,
        "resource_type": resource_type,
        "resource_title": resource_title,
        "size": target_path.stat().st_size,
        "file_api_path": _file_api_path(task_id, resource_name),
        "download_api_path": _file_download_api_path(task_id, resource_name),
        "url": url,
        "snapshot": (title_line + parsed_content)[:500],
    }


@router.get("/agent/sessions/{task_id}/files/download", summary="Download session file")
def download_session_file(
    task_id: str,
    name: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    raw_name = str(name or "").strip()
    target_name = Path(raw_name).name.strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Missing file name")
    if target_name != raw_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    target_path = _resolve_existing_file_path(record, target_name)
    if target_path is None:
        raise HTTPException(status_code=404, detail=f"File '{target_name}' not found")
    media_type, _ = mimetypes.guess_type(str(target_path))
    return FileResponse(
        path=str(target_path),
        media_type=media_type or "application/octet-stream",
        filename=target_name,
    )


@router.get("/agent/sessions/{task_id}/files/raw", summary="Get raw session file text")
def get_session_file_raw(
    task_id: str,
    name: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    payload = get_session_file_content(task_id, name, current_user)
    raw_text = str(payload.get("content") or "")
    ext = Path(str(payload.get("name") or "")).suffix.lower()
    media_type = "text/markdown; charset=utf-8" if ext in {".md", ".markdown"} else "text/plain; charset=utf-8"
    return PlainTextResponse(content=raw_text, media_type=media_type)


@router.delete("/agent/sessions/{task_id}/files", summary="Delete a session file")
def delete_session_file(
    task_id: str,
    name: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    raw_name = str(name or "").strip()
    target_name = Path(raw_name).name.strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="Missing file name")
    if target_name != raw_name:
        raise HTTPException(status_code=400, detail="Invalid file name")
    user_id = _normalize_user_id(current_user)
    record = _get_task_with_owner_fallback(task_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    existing_files = list(record.files or [])
    kept_files: List[Dict[str, Any]] = []
    removed_entry: Optional[Dict[str, Any]] = None
    for item in existing_files:
        item_name = str(item.get("name") or "").strip()
        if removed_entry is None and item_name.lower() == target_name.lower():
            removed_entry = item
            continue
        kept_files.append(item)
    target_path: Optional[Path] = None
    if removed_entry:
        local_path = str(removed_entry.get("local_path") or "").strip()
        if local_path:
            p = Path(local_path)
            if p.exists() and p.is_file():
                target_path = p
    if target_path is None:
        target_path = _resolve_existing_file_path(record, target_name)
    removed_from_workspace = False
    if target_path and target_path.exists() and target_path.is_file():
        try:
            target_path.unlink()
            removed_from_workspace = True
        except Exception:
            removed_from_workspace = False
    record.files = kept_files
    _persist(record)
    return {
        "task_id": task_id,
        "name": target_name,
        "deleted": removed_entry is not None,
        "removed_from_workspace": removed_from_workspace,
    }


@router.delete("/agent/sessions/{task_id}", summary="Delete a task session")
def delete_session(
    task_id: str,
    current_user: Annotated[dict, Depends(_get_current_user_optional)],
):
    user_id = _normalize_user_id(current_user)
    path = _task_path(task_id, user_id)
    db_rec = _load_task_from_db(task_id, user_id)
    if not path.exists() and not db_rec:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if path.exists():
        path.unlink()
    _delete_task_from_db(task_id, user_id)
    _tasks.pop(_task_cache_key(user_id, task_id), None)
    return {"deleted": task_id}


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BatchAgent API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime
from pathlib import Path
import io
import os
import re
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore, LexicalIndex, VectorStore
from backend.app.services.indexing import IndexingService
from backend.app.services.search import SearchService, SearchResult
from backend.app.services.jobs import JobRunner
from backend.app.config.schema import AppConfig
from backend.app.dependencies import (
    get_metadata_store,
    get_indexing_service,
    get_search_service,
    get_job_runner,
    get_config,
    get_lexical_index,
    get_vector_store,
)

router = APIRouter()

UPLOADS_SOURCE_NAME = "My Documents"
UPLOADS_DIR_NAME = "uploads"

# Default user ID for document storage paths (mirrors agent's "anonymous" user)
_DOC_USER_ID = os.environ.get("DOC_USER_ID", "anonymous")


def _doc_cache_dir() -> Path:
    """Return the shared agent document cache directory for uploaded files."""
    workspace = Path(os.environ.get("AGENT_WORKSPACE", "./agent_workspace"))
    d = workspace / "users" / _DOC_USER_ID / ".cache" / "documents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _minio_upload(config, object_name: str, data: bytes, content_type: str) -> Optional[str]:
    """Upload bytes to MinIO and return the public URL, or None if unavailable."""
    mc = getattr(config, "minio", None)
    if not mc or not mc.endpoint:
        return None
    # Env vars take priority over (potentially cached) config values
    endpoint = os.environ.get("MINIO_ENDPOINT") or mc.endpoint
    access_key = os.environ.get("MINIO_ACCESS_KEY") or mc.access_key
    secret_key = os.environ.get("MINIO_SECRET_KEY") or mc.secret_key
    bucket = os.environ.get("MINIO_BUCKET") or mc.bucket
    secure = mc.secure
    try:
        from minio import Minio
        endpoint_host = endpoint.replace("http://", "").replace("https://", "").rstrip("/")
        client = Minio(endpoint_host, access_key=access_key, secret_key=secret_key, secure=secure)
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.put_object(bucket, object_name, io.BytesIO(data), len(data), content_type=content_type)
        scheme = "https" if secure else "http"
        return f"{scheme}://{endpoint_host}/{bucket}/{object_name}"
    except Exception as exc:
        print(f"[MinIO] Upload failed for {object_name}: {exc}")
        return None


def _guess_mime(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        return "application/pdf"
    if ext in ("md", "markdown"):
        return "text/markdown"
    if ext in ("html", "htm"):
        return "text/html"
    if ext == "txt":
        return "text/plain"
    return "application/octet-stream"


# Schemas

class CreateSourceReq(BaseModel):
    name: str
    path: str
    config: Dict = {}

class SearchReq(BaseModel):
    query: str
    top_k: int = 10

# --- Routes ---

@router.post("/sources", response_model=models.Source)
def create_source(
    req: CreateSourceReq,
    store: MetadataStore = Depends(get_metadata_store)
):
    source = models.Source(name=req.name, path=req.path, config=req.config)
    return store.upsert_source(source)

@router.get("/sources", response_model=List[models.Source])
def list_sources(store: MetadataStore = Depends(get_metadata_store)):
    return store.list_sources()

@router.post("/sources/{source_id}/scan", response_model=models.Job)
def scan_source(
    source_id: UUID,
    runner: JobRunner = Depends(get_job_runner),
):
    try:
        job = runner.enqueue_job(
            type=models.JobType.SCAN_SOURCE,
            payload={"source_id": str(source_id)},
        )
        job_response = models.Job(**job.model_dump())
        runner._process_job(job)
        return job_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs", response_model=List[models.Job])
def list_jobs(store: MetadataStore = Depends(get_metadata_store)):
    return store.list_jobs()

@router.get("/jobs/{job_id}", response_model=models.Job)
def get_job(job_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/documents", response_model=List[models.Document])
def list_documents(
    source_id: Optional[UUID] = None,
    store: MetadataStore = Depends(get_metadata_store)
):
    if source_id:
        return store.list_documents_by_source(source_id)
    docs: List[models.Document] = []
    for source in store.list_sources():
        docs.extend(store.list_documents_by_source(source.id))
    return docs

@router.get("/documents/{doc_id}", response_model=models.Document)
def get_document(doc_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(
    doc_id: UUID,
    store: MetadataStore = Depends(get_metadata_store),
    lexical: LexicalIndex = Depends(get_lexical_index),
    vector: VectorStore = Depends(get_vector_store),
    config: AppConfig = Depends(get_config),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    uri = doc.uri or ""
    if uri.startswith("file://"):
        uploads_root = (config.storage.data_dir / UPLOADS_DIR_NAME).resolve()
        cache_root = _doc_cache_dir().resolve()
        file_path = Path(uri.replace("file://", "", 1)).resolve()
        in_uploads = False
        try:
            file_path.relative_to(uploads_root)
            in_uploads = True
        except ValueError:
            pass
        in_cache = False
        try:
            file_path.relative_to(cache_root)
            in_cache = True
        except ValueError:
            pass
        if (in_uploads or in_cache) and file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        # Also remove the extracted .md cache file if it exists
        if in_cache or in_uploads:
            stem = Path(doc.title or file_path.stem).stem
            cache_md = cache_root / f"{stem}.md"
            if cache_md.exists():
                try:
                    cache_md.unlink()
                except Exception:
                    pass
    lexical.delete_doc(doc_id)
    vector.delete_doc(doc_id)
    store.delete_document(doc_id)

@router.get("/documents/{doc_id}/sections", response_model=List[models.Section])
def list_doc_sections(doc_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    return store.list_sections(doc_id)

@router.get("/documents/{doc_id}/chunks", response_model=List[models.Chunk])
def list_doc_chunks(doc_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    return store.list_chunks(doc_id)


@router.get("/documents/{doc_id}/file")
def get_document_file(
    doc_id: UUID,
    download: bool = Query(False),
    store: MetadataStore = Depends(get_metadata_store),
):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    uri = doc.uri
    if uri.startswith("file://"):
        path = uri.replace("file://", "", 1)
        file_path = Path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on server")
        media_type = doc.mime_type or "application/octet-stream"
        disposition = "attachment" if download else "inline"
        headers = {
            "Content-Disposition": f'{disposition}; filename="{(doc.title or file_path.name)}"'
        }
        return FileResponse(
            file_path,
            media_type=media_type,
            headers=headers,
        )

    if uri.startswith("http://") or uri.startswith("https://"):
        return RedirectResponse(uri)

    raise HTTPException(status_code=400, detail="Unsupported document URI scheme")


@router.post("/documents/upload", response_model=models.Document)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    store: MetadataStore = Depends(get_metadata_store),
    indexing: IndexingService = Depends(get_indexing_service),
    config: AppConfig = Depends(get_config),
):
    contents = await file.read()
    size_bytes = len(contents)
    max_bytes = config.ingestion.max_file_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(status_code=400, detail="File too large")

    original_name = file.filename or "document"
    mime = _guess_mime(original_name)
    base = Path(original_name).stem
    suffix = Path(original_name).suffix

    # Save to shared document cache (agent_workspace/users/{user_id}/.cache/documents/)
    cache_dir = _doc_cache_dir()
    local_path = cache_dir / original_name
    idx = 1
    while local_path.exists():
        local_path = cache_dir / f"{base}_{idx}{suffix}"
        idx += 1
    with open(local_path, "wb") as f:
        f.write(contents)

    # Always use the local file:// URI for indexing (MinIO objects are private,
    # direct HTTP GET would be 403). Upload to MinIO as a background backup copy.
    document_uri = f"file://{local_path.resolve()}"
    safe_name = re.sub(r"[^\w.\-]", "_", local_path.name)
    minio_object = f"documents/{_DOC_USER_ID}/{safe_name}"
    _minio_upload(config, minio_object, contents, mime)  # best-effort; result not stored

    sources = store.list_sources()
    upload_source = next((s for s in sources if s.name == UPLOADS_SOURCE_NAME), None)
    if upload_source is None:
        upload_source = models.Source(
            name=UPLOADS_SOURCE_NAME,
            path=str(cache_dir),
            config={},
        )
        upload_source = store.upsert_source(upload_source)

    doc = models.Document(
        source_id=upload_source.id,
        uri=document_uri,
        title=original_name,
        mime_type=mime,
        size_bytes=size_bytes,
        mtime=datetime.utcnow(),
        status="pending",
    )
    saved = store.upsert_document(doc)
    # Run indexing as a background task — no job-queue DB writes required
    background_tasks.add_task(indexing.index_document, saved.id)
    return saved

@router.post("/search", response_model=List[SearchResult])
def search(
    req: SearchReq,
    service: SearchService = Depends(get_search_service)
):
    return service.search(req.query, req.top_k)

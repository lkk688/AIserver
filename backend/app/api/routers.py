from typing import List, Optional, Dict
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore
from backend.app.services.indexing import IndexingService
from backend.app.services.search import SearchService, SearchResult
from backend.app.services.jobs import JobRunner
from backend.app.dependencies import get_metadata_store, get_indexing_service, get_search_service, get_job_runner

router = APIRouter()

# --- Schemas ---

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
    runner: JobRunner = Depends(get_job_runner)
):
    try:
        # Enqueue job instead of running synchronously
        return runner.enqueue_job(
            type=models.JobType.SCAN_SOURCE, 
            payload={"source_id": str(source_id)}
        )
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
    source_id: UUID,
    store: MetadataStore = Depends(get_metadata_store)
):
    return store.list_documents_by_source(source_id)

@router.get("/documents/{doc_id}", response_model=models.Document)
def get_document(doc_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.get("/documents/{doc_id}/chunks", response_model=List[models.Chunk])
def list_doc_chunks(doc_id: UUID, store: MetadataStore = Depends(get_metadata_store)):
    return store.list_chunks(doc_id)

@router.post("/search", response_model=List[SearchResult])
def search(
    req: SearchReq,
    service: SearchService = Depends(get_search_service)
):
    return service.search(req.query, req.top_k)

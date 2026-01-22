from enum import Enum
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

class DocType(str, Enum):
    FILE = "file"
    WEB = "web"

class JobType(str, Enum):
    SCAN_SOURCE = "scan_source"
    INDEX_DOC = "index_doc"
    REINDEX_ALL = "reindex_all"

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

class DomainModel(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Source(DomainModel):
    name: str
    path: str
    config: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

class Document(DomainModel):
    source_id: UUID
    uri: str
    title: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None
    mtime: Optional[datetime] = None
    doc_hash: Optional[str] = None
    status: str = "new"

    class Config:
        from_attributes = True

class Chunk(DomainModel):
    doc_id: UUID
    chunk_index: int
    text: str
    start_offset: int
    end_offset: int
    chunk_hash: str

    class Config:
        from_attributes = True

class Job(DomainModel):
    type: JobType
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True

class ExtractedContent(BaseModel):
    text: str
    title: Optional[str] = None
    mime_type: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

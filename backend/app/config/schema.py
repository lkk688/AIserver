from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class MetadataBackend(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"

class LexicalBackend(str, Enum):
    FTS5 = "fts5"
    PG_FTS = "pg_fts"

class VectorBackend(str, Enum):
    FAISS = "faiss"
    PGVECTOR = "pgvector"

class StorageConfig(BaseModel):
    data_dir: Path
    sqlite_path: Path
    faiss_dir: Path

    @field_validator("data_dir", "sqlite_path", "faiss_dir")
    @classmethod
    def normalize_path(cls, v: Path) -> Path:
        return v.expanduser().resolve()

class IngestionConfig(BaseModel):
    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)
    max_file_mb: int = Field(gt=0)

class BookmarksConfig(BaseModel):
    chrome_bookmarks_path: Optional[Path] = None

    @field_validator("chrome_bookmarks_path")
    @classmethod
    def normalize_optional_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v:
            return v.expanduser().resolve()
        return v

class WebFetchConfig(BaseModel):
    enabled: bool = False
    timeout_sec: int = Field(gt=0, default=30)
    user_agent: str = "AIserver/1.0"

class EmbeddingConfig(BaseModel):
    provider: str
    model_name: str
    dim: int = Field(gt=0)

class AppConfig(BaseModel):
    metadata_backend: MetadataBackend
    lexical_backend: LexicalBackend
    vector_backend: VectorBackend
    storage: StorageConfig
    ingestion: IngestionConfig
    bookmarks: BookmarksConfig
    web_fetch: WebFetchConfig
    embedding: EmbeddingConfig

from typing import List, Optional, Tuple
from uuid import UUID
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore, LexicalIndex, VectorStore, ContentExtractor, EmbeddingProvider
from backend.app.config.schema import AppConfig

class PostgresMetadataStore(MetadataStore):
    """
    Stub implementation for Postgres metadata store.
    """
    def __init__(self, config: AppConfig):
        # TODO: Initialize SQLAlchemy engine for Postgres
        # self.engine = create_engine(config.storage.postgres_url)
        pass

    def upsert_source(self, source: models.Source) -> models.Source:
        raise NotImplementedError("Postgres backend not implemented yet")

    def get_source(self, source_id: UUID) -> Optional[models.Source]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def list_sources(self) -> List[models.Source]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def upsert_document(self, doc: models.Document) -> models.Document:
        raise NotImplementedError("Postgres backend not implemented yet")

    def get_document(self, doc_id: UUID) -> Optional[models.Document]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def list_documents_by_source(self, source_id: UUID) -> List[models.Document]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def mark_document_deleted(self, doc_id: UUID) -> None:
        raise NotImplementedError("Postgres backend not implemented yet")

    def upsert_chunk(self, chunk: models.Chunk) -> models.Chunk:
        raise NotImplementedError("Postgres backend not implemented yet")

    def list_chunks(self, doc_id: UUID) -> List[models.Chunk]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def get_chunk(self, chunk_id: UUID) -> Optional[models.Chunk]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def upsert_job(self, job: models.Job) -> models.Job:
        raise NotImplementedError("Postgres backend not implemented yet")

    def get_job(self, job_id: UUID) -> Optional[models.Job]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def list_jobs(self) -> List[models.Job]:
        raise NotImplementedError("Postgres backend not implemented yet")

    def get_pending_jobs(self, limit: int = 10) -> List[models.Job]:
        raise NotImplementedError("Postgres backend not implemented yet")

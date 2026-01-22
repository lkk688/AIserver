from typing import List, Tuple
from uuid import UUID
from backend.app.domain import models
from backend.app.domain.ports import VectorStore
from backend.app.config.schema import AppConfig

class PgVectorStore(VectorStore):
    """
    Stub implementation for Postgres/pgvector vector store.
    
    Intended Schema:
    TABLE embeddings (
        chunk_id UUID PRIMARY KEY REFERENCES chunks(id),
        embedding vector(DIM)
    );
    """
    def __init__(self, config: AppConfig):
        # TODO: Check for pgvector extension
        pass

    def upsert_embeddings(self, chunks: List[models.Chunk], embeddings: List[List[float]]) -> None:
        raise NotImplementedError("PgVector backend not implemented yet")

    def delete_doc(self, doc_id: UUID) -> None:
        raise NotImplementedError("PgVector backend not implemented yet")

    def query(self, vector: List[float], top_k: int) -> List[Tuple[UUID, float]]:
        raise NotImplementedError("PgVector backend not implemented yet")

from typing import List, Tuple
from uuid import UUID
from backend.app.domain import models
from backend.app.domain.ports import LexicalIndex
from backend.app.config.schema import AppConfig

class PgFTSIndex(LexicalIndex):
    """
    Stub implementation for Postgres Full-Text Search.
    
    Intended Schema:
    Uses tsvector column on chunks table or separate table.
    TABLE chunk_fts (
        chunk_id UUID REFERENCES chunks(id),
        tsv tsvector
    );
    CREATE INDEX ON chunk_fts USING GIN (tsv);
    """
    def __init__(self, config: AppConfig):
        pass

    def upsert_chunks(self, chunks: List[models.Chunk]) -> None:
        raise NotImplementedError("Postgres FTS backend not implemented yet")

    def delete_doc(self, doc_id: UUID) -> None:
        raise NotImplementedError("Postgres FTS backend not implemented yet")

    def search(self, query: str, top_k: int) -> List[Tuple[UUID, float]]:
        raise NotImplementedError("Postgres FTS backend not implemented yet")

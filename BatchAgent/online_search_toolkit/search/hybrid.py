from __future__ import annotations

import logging
from typing import List, Optional

from ..embedding_client import EmbeddingClient
from ..models import SearchRecord
from ..reranker import BaseReranker
from ..store.base import SearchStore
from ..utils import build_embedding_text, dedupe_preserve_order, utc_now

logger = logging.getLogger(__name__)


class HybridSearcher:
    def __init__(
        self,
        file_store: SearchStore,
        pg_store: Optional[SearchStore],
        embedding_client: Optional[EmbeddingClient],
        reranker: BaseReranker,
    ):
        self.file_store = file_store
        self.pg_store = pg_store
        self.embedding_client = embedding_client
        self.reranker = reranker

    def _normalize_language(self, language: str) -> Optional[str]:
        return None if language == "mixed" else language

    def enrich_embeddings(self, records: List[SearchRecord]) -> List[SearchRecord]:
        if not self.embedding_client:
            return records

        pending = [r for r in records if r.embedding is None]
        if not pending:
            return records

        batch_size = self.embedding_client.config.batch_size
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            texts = [
                build_embedding_text(
                    r.title,
                    r.source,
                    r.category,
                    r.summary,
                    r.content or "",
                )
                for r in chunk
            ]
            try:
                vectors = self.embedding_client.embed_texts(texts)
            except Exception:
                logger.warning("Embedding service unavailable, skipping embeddings for this batch")
                continue
            for record, vec in zip(chunk, vectors):
                record.embedding = vec
                record.embedding_model = self.embedding_client.config.api_model

        return records

    def search(
        self,
        query: str,
        limit: int,
        language: str = "mixed",
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = 48,
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        normalized_lang = self._normalize_language(language)

        keyword_candidates = self.file_store.keyword_search(
            query=query,
            limit=max(limit * 3, 20),
            language=normalized_lang,
            domain=domain,
            category=category,
            recent_hours=recent_hours,
            record_types=record_types,
        )

        semantic_candidates: List[SearchRecord] = []
        if self.embedding_client:
            try:
                query_embedding = self.embedding_client.embed_texts([query])[0]

                semantic_candidates.extend(
                    self.file_store.semantic_search(
                        query_embedding=query_embedding,
                        limit=max(limit * 3, 20),
                        language=normalized_lang,
                        domain=domain,
                        category=category,
                        recent_hours=recent_hours,
                        record_types=record_types,
                    )
                )

                if self.pg_store:
                    semantic_candidates.extend(
                        self.pg_store.semantic_search(
                            query_embedding=query_embedding,
                            limit=max(limit * 3, 20),
                            language=normalized_lang,
                            domain=domain,
                            category=category,
                            recent_hours=recent_hours,
                            record_types=record_types,
                        )
                    )
            except Exception:
                # Embedding server unavailable — fall back to keyword-only search
                logger.warning("Embedding search failed for query=%r, falling back to keyword-only", query)

        all_items = dedupe_preserve_order(
            keyword_candidates + semantic_candidates,
            key_fn=lambda x: x.id,
        )

        all_items.sort(
            key=lambda x: (
                1 if x.is_breaking else 0,
                x.published_at or x.fetched_at or utc_now(),
            ),
            reverse=True,
        )

        reranked = self.reranker.rerank(query, all_items)
        return reranked[:limit]

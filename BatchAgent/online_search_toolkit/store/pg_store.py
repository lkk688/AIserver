from __future__ import annotations

import logging
from typing import List, Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from ..config import SearchConfig
from ..models import SearchRecord
from .base import SearchStore

logger = logging.getLogger(__name__)


class PostgresSearchStore(SearchStore):
    def __init__(self, config: SearchConfig):
        self.config = config
        self.dsn = config.store.postgres_dsn
        self.embedding_dim = config.store.embedding_dim

    def _connect(self):
        conn = psycopg.connect(self.dsn, row_factory=dict_row)
        register_vector(conn)
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS search_records (
                        id TEXT PRIMARY KEY,
                        record_type TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT,
                        url TEXT NOT NULL,
                        content TEXT NULL,
                        source TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        language TEXT,
                        category TEXT,
                        query TEXT NULL,
                        published_at TIMESTAMPTZ NULL,
                        fetched_at TIMESTAMPTZ NOT NULL,
                        is_breaking BOOLEAN NOT NULL DEFAULT FALSE,
                        embedding_model TEXT NULL,
                        embedding VECTOR({self.embedding_dim}) NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                    )
                    """
                )
                cur.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_search_records_url ON search_records (url)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_published_at ON search_records (published_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_fetched_at ON search_records (fetched_at DESC)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_domain ON search_records (domain)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_language ON search_records (language)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_category ON search_records (category)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_record_type ON search_records (record_type)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_search_records_breaking ON search_records (is_breaking)"
                )
            conn.commit()

    def upsert_records(self, records: List[SearchRecord]) -> None:
        if not records:
            return

        with self._connect() as conn:
            with conn.cursor() as cur:
                for record in records:
                    cur.execute(
                        """
                        INSERT INTO search_records (
                            id, record_type, source_type, title, summary, url, content,
                            source, domain, language, category, query,
                            published_at, fetched_at, is_breaking,
                            embedding_model, embedding, metadata
                        ) VALUES (
                            %(id)s, %(record_type)s, %(source_type)s, %(title)s, %(summary)s, %(url)s, %(content)s,
                            %(source)s, %(domain)s, %(language)s, %(category)s, %(query)s,
                            %(published_at)s, %(fetched_at)s, %(is_breaking)s,
                            %(embedding_model)s, %(embedding)s, %(metadata)s
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            record_type = EXCLUDED.record_type,
                            source_type = EXCLUDED.source_type,
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            url = EXCLUDED.url,
                            content = COALESCE(EXCLUDED.content, search_records.content),
                            source = EXCLUDED.source,
                            domain = EXCLUDED.domain,
                            language = EXCLUDED.language,
                            category = EXCLUDED.category,
                            query = EXCLUDED.query,
                            published_at = EXCLUDED.published_at,
                            fetched_at = EXCLUDED.fetched_at,
                            is_breaking = EXCLUDED.is_breaking,
                            embedding_model = EXCLUDED.embedding_model,
                            embedding = COALESCE(EXCLUDED.embedding, search_records.embedding),
                            metadata = EXCLUDED.metadata
                        """,
                        record.model_dump(),
                    )
            conn.commit()

    def _rows_to_records(self, rows) -> List[SearchRecord]:
        return [SearchRecord.model_validate(dict(row)) for row in rows]

    def get_by_url(self, url: str) -> Optional[SearchRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM search_records
                WHERE url = %(url)s
                ORDER BY fetched_at DESC
                LIMIT 1
                """,
                {"url": url},
            )
            row = cur.fetchone()
            return SearchRecord.model_validate(dict(row)) if row else None

    def get_recent_records(
        self,
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
    ) -> List[SearchRecord]:
        filters = []
        params = {"limit": limit}

        if language:
            filters.append("language = %(language)s")
            params["language"] = language
        if domain:
            filters.append("domain = %(domain)s")
            params["domain"] = domain
        if category:
            filters.append("category = %(category)s")
            params["category"] = category
        if recent_hours is not None:
            filters.append(
                "COALESCE(published_at, fetched_at) >= NOW() - (%(recent_hours)s || ' hours')::interval"
            )
            params["recent_hours"] = recent_hours

        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        sql = f"""
            SELECT * FROM search_records
            {where_sql}
            ORDER BY is_breaking DESC, COALESCE(published_at, fetched_at) DESC
            LIMIT %(limit)s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return self._rows_to_records(cur.fetchall())

    def keyword_search(
        self,
        query: str,
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        filters = [
            "(title ILIKE %(q)s OR summary ILIKE %(q)s OR content ILIKE %(q)s OR source ILIKE %(q)s)"
        ]
        params = {"q": f"%{query}%", "limit": limit}

        if language:
            filters.append("language = %(language)s")
            params["language"] = language
        if domain:
            filters.append("domain = %(domain)s")
            params["domain"] = domain
        if category:
            filters.append("category = %(category)s")
            params["category"] = category
        if recent_hours is not None:
            filters.append(
                "COALESCE(published_at, fetched_at) >= NOW() - (%(recent_hours)s || ' hours')::interval"
            )
            params["recent_hours"] = recent_hours
        if record_types:
            filters.append("record_type = ANY(%(record_types)s)")
            params["record_types"] = record_types

        sql = f"""
            SELECT * FROM search_records
            WHERE {' AND '.join(filters)}
            ORDER BY is_breaking DESC, COALESCE(published_at, fetched_at) DESC
            LIMIT %(limit)s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return self._rows_to_records(cur.fetchall())

    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        filters = ["embedding IS NOT NULL"]
        params = {"embedding": query_embedding, "limit": limit}

        if language:
            filters.append("language = %(language)s")
            params["language"] = language
        if domain:
            filters.append("domain = %(domain)s")
            params["domain"] = domain
        if category:
            filters.append("category = %(category)s")
            params["category"] = category
        if recent_hours is not None:
            filters.append(
                "COALESCE(published_at, fetched_at) >= NOW() - (%(recent_hours)s || ' hours')::interval"
            )
            params["recent_hours"] = recent_hours
        if record_types:
            filters.append("record_type = ANY(%(record_types)s)")
            params["record_types"] = record_types

        sql = f"""
            SELECT * FROM search_records
            WHERE {' AND '.join(filters)}
            ORDER BY embedding <=> %(embedding)s, is_breaking DESC, COALESCE(published_at, fetched_at) DESC
            LIMIT %(limit)s
        """

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return self._rows_to_records(cur.fetchall())

    def archive_daily_snapshot(self) -> None:
        logger.info("Postgres store keeps history directly; daily snapshot is handled by file store.")
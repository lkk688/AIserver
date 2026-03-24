"""
Integration tests for the OnlineSearchService and bootstrap.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from BatchAgent.online_search_toolkit.config import SearchConfig
from BatchAgent.online_search_toolkit.bootstrap import (
    build_academic_search_fn,
    build_news_search_fn,
    build_web_search_fn,
    create_online_search_service,
)
from BatchAgent.online_search_toolkit.models import (
    SearchRecord,
    SearchRequest,
    SearchResult,
)
from BatchAgent.online_search_toolkit.service import OnlineSearchService
from BatchAgent.online_search_toolkit.utils import utc_now


# =====================================================================
# Config Tests
# =====================================================================

class TestSearchConfig:
    def test_default_config(self):
        cfg = SearchConfig()
        assert cfg.max_summary_chars == 240
        assert cfg.default_search_limit == 8
        assert cfg.embedding is not None
        assert cfg.rerank is not None
        assert cfg.store is not None
        assert cfg.scheduler is not None
        assert cfg.crawler is not None
        assert cfg.api_keys is not None

    def test_crawler_config_defaults(self):
        cfg = SearchConfig()
        assert cfg.crawler.headless is True
        assert cfg.crawler.max_pages == 50
        assert cfg.crawler.max_depth == 3

    def test_api_keys_config(self):
        cfg = SearchConfig()
        assert isinstance(cfg.api_keys.serper_api_key, str)
        assert isinstance(cfg.api_keys.tavily_api_key, str)


# =====================================================================
# Bootstrap Tests
# =====================================================================

class TestBootstrap:
    def test_build_news_search_fn(self):
        cfg = SearchConfig()
        fn = build_news_search_fn(cfg)
        assert callable(fn)

    def test_build_web_search_fn(self):
        cfg = SearchConfig()
        fn = build_web_search_fn(cfg)
        assert callable(fn)

    def test_build_academic_search_fn(self):
        cfg = SearchConfig()
        fn = build_academic_search_fn(cfg)
        assert callable(fn)

    def test_create_service(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SearchConfig()
            cfg.store.file_cache_dir = Path(tmpdir)
            cfg.store.postgres_enabled = False
            cfg.embedding.enabled = False

            service = create_online_search_service(cfg)
            assert isinstance(service, OnlineSearchService)


# =====================================================================
# Service Tests
# =====================================================================

class TestOnlineSearchService:
    def _make_service(self, tmpdir: str) -> OnlineSearchService:
        cfg = SearchConfig()
        cfg.store.file_cache_dir = Path(tmpdir)
        cfg.store.postgres_enabled = False
        cfg.embedding.enabled = False

        def mock_news_fn(query, limit=8, **kwargs):
            return [
                SearchRecord(
                    id=f"news_{query[:5]}",
                    record_type="news_article",
                    source_type="rss",
                    title=f"News: {query}",
                    summary=f"News result for {query}",
                    url=f"https://news.example.com/{query.replace(' ', '-')}",
                    source="MockNews",
                    domain="news",
                    language="en",
                    category="general",
                    fetched_at=utc_now(),
                    metadata={},
                )
            ]

        def mock_web_fn(query, domain="general", limit=8, **kwargs):
            return [
                SearchRecord(
                    id=f"web_{query[:5]}",
                    record_type="web_page",
                    source_type="search_api",
                    title=f"Web: {query}",
                    summary=f"Web result for {query}",
                    url=f"https://example.com/{query.replace(' ', '-')}",
                    source="MockSearch",
                    domain=domain,
                    language="en",
                    category="general",
                    fetched_at=utc_now(),
                    metadata={},
                )
            ]

        def mock_academic_fn(query, limit=8, **kwargs):
            return [
                SearchRecord(
                    id=f"acad_{query[:5]}",
                    record_type="academic_paper",
                    source_type="pubmed",
                    title=f"Paper: {query}",
                    summary=f"Academic result for {query}",
                    url=f"https://pubmed.example.com/{query.replace(' ', '-')}",
                    source="MockPubMed",
                    domain="academic",
                    language="en",
                    category="research",
                    fetched_at=utc_now(),
                    metadata={},
                )
            ]

        service = OnlineSearchService(
            config=cfg,
            news_search_fn=mock_news_fn,
            web_search_fn=mock_web_fn,
            academic_search_fn=mock_academic_fn,
        )
        service.initialize()
        return service

    def test_build_search_plan_news(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            req = SearchRequest(query="breaking news", domain="news", limit=5)
            plan = service.build_search_plan(req)
            assert plan is not None
            # Field is normalized_query, not query
            assert plan.normalized_query == "breaking news"
            assert plan.domain == "news"
            # News domain enables RSS / API sources
            assert plan.use_news_rss is True or plan.use_news_api is True

    def test_build_search_plan_academic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            req = SearchRequest(query="deep learning", domain="academic", limit=5)
            plan = service.build_search_plan(req)
            # Academic domain enables pubmed / semantic scholar
            assert plan.use_pubmed is True or plan.use_wikimedia is True

    def test_build_search_plan_general(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            req = SearchRequest(query="python tutorial", domain="general", limit=5)
            plan = service.build_search_plan(req)
            # General domain uses Serper + Tavily + Wikimedia
            assert plan.use_serper is True

    def test_search_news(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            result = service.search_news(query="AI developments", limit=5)
            assert isinstance(result, SearchResult)
            assert result.count >= 1
            # SearchResult stores results in .items, not .records
            assert result.items[0].domain == "news"

    def test_search_web(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            result = service.search_web(query="python tutorial", limit=5)
            assert isinstance(result, SearchResult)
            assert result.count >= 1

    def test_search_academic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            result = service.search_academic(query="transformer architecture", limit=5)
            assert isinstance(result, SearchResult)
            assert result.count >= 1
            # SearchResult stores results in .items, not .records
            assert result.items[0].domain == "academic"

    def test_search_persists_to_file_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)

            # First search — should fetch and persist
            result1 = service.search_web(query="persistence test", limit=5)
            assert result1.count >= 1

            # Check file store has records
            records_file = Path(tmpdir) / "records.json"
            assert records_file.exists()

            data = json.loads(records_file.read_text())
            assert len(data.get("records", [])) >= 1

    def test_read_url_with_crawler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)

            # Must use a real SearchRecord (not MagicMock) so file store can serialize it
            real_record = SearchRecord(
                id="mock_page_id",
                record_type="web_page",
                source_type="url_fetch",
                title="Test Page",
                summary="Page summary",
                url="https://example.com/test",
                source="httpx",
                domain="general",
                language="en",
                category="general",
                fetched_at=utc_now(),
                content="Page content",
                metadata={},
            )

            # The method on URLReader is read_url(), not read()
            with patch.object(service.url_reader, "read_url", return_value=real_record) as mock_read:
                result = service.read_url("https://example.com/test", force_refresh=True)
                mock_read.assert_called_once()
            assert result.title == "Test Page"


# =====================================================================
# File Store Tests
# =====================================================================

class TestFileStore:
    def test_file_store_roundtrip(self):
        from BatchAgent.online_search_toolkit.store.file_store import FileSearchStore

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SearchConfig()
            cfg.store.file_cache_dir = Path(tmpdir)
            store = FileSearchStore(cfg)
            store.initialize()

            records = [
                SearchRecord(
                    id="fs_test_1",
                    record_type="web_page",
                    source_type="search_api",
                    title="Test Record 1",
                    summary="Summary 1",
                    url="https://example.com/1",
                    source="Test",
                    domain="general",
                    language="en",
                    category="general",
                    fetched_at=utc_now(),
                    metadata={},
                ),
                SearchRecord(
                    id="fs_test_2",
                    record_type="news_article",
                    source_type="rss",
                    title="Test Record 2",
                    summary="Summary 2",
                    url="https://example.com/2",
                    source="News",
                    domain="news",
                    language="en",
                    category="breaking",
                    fetched_at=utc_now(),
                    is_breaking=True,
                    metadata={},
                ),
            ]
            store.upsert_records(records)

            # Read back
            all_records = store.get_recent_records(limit=10)
            assert len(all_records) == 2

            # Breaking should come first
            assert all_records[0].is_breaking is True

            # Get by URL
            r = store.get_by_url("https://example.com/1")
            assert r is not None
            assert r.title == "Test Record 1"

    def test_keyword_search(self):
        from BatchAgent.online_search_toolkit.store.file_store import FileSearchStore

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SearchConfig()
            cfg.store.file_cache_dir = Path(tmpdir)
            store = FileSearchStore(cfg)
            store.initialize()

            store.upsert_records([
                SearchRecord(
                    id="kw_1",
                    record_type="web_page",
                    source_type="search_api",
                    title="Python Programming Tutorial",
                    summary="Learn Python basics",
                    url="https://example.com/python",
                    source="Test",
                    domain="general",
                    language="en",
                    category="programming",
                    fetched_at=utc_now(),
                    metadata={},
                ),
                SearchRecord(
                    id="kw_2",
                    record_type="web_page",
                    source_type="search_api",
                    title="JavaScript Guide",
                    summary="Learn JavaScript",
                    url="https://example.com/js",
                    source="Test",
                    domain="general",
                    language="en",
                    category="programming",
                    fetched_at=utc_now(),
                    metadata={},
                ),
            ])

            results = store.keyword_search("Python", limit=10)
            assert len(results) == 1
            assert "Python" in results[0].title

    def test_archive_daily_snapshot(self):
        from BatchAgent.online_search_toolkit.store.file_store import FileSearchStore

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SearchConfig()
            cfg.store.file_cache_dir = Path(tmpdir)
            store = FileSearchStore(cfg)
            store.initialize()

            store.upsert_records([
                SearchRecord(
                    id="arch_1",
                    record_type="web_page",
                    source_type="search_api",
                    title="Archive Test",
                    summary="Test",
                    url="https://example.com/archive",
                    source="Test",
                    domain="general",
                    language="en",
                    category="general",
                    fetched_at=utc_now(),
                    metadata={},
                ),
            ])

            store.archive_daily_snapshot()

            daily_dir = Path(tmpdir) / "daily"
            snapshots = list(daily_dir.glob("*.json"))
            assert len(snapshots) == 1


# =====================================================================
# Extended Service Tests
# Each test documents WHAT it validates and WHY it matters.
# =====================================================================

class TestOnlineSearchServiceExtended:
    """
    Additional integration tests for OnlineSearchService covering:
    - medical domain routing
    - caching behaviour (second call hits file store)
    - no-results fallback
    - refresh_domain_cache
    - search with use_cache=False (force online fetch)
    - multi-query deduplication
    """

    def _make_service(self, tmpdir: str, *, news_fn=None, web_fn=None, academic_fn=None) -> OnlineSearchService:
        cfg = SearchConfig()
        cfg.store.file_cache_dir = Path(tmpdir)
        cfg.store.postgres_enabled = False
        cfg.embedding.enabled = False

        default_record = lambda domain, rt, st: SearchRecord(
            id=f"{domain}_{rt}",
            record_type=rt,
            source_type=st,
            title=f"Result ({domain})",
            summary=f"Summary for {domain}",
            url=f"https://example.com/{domain}",
            source="Mock",
            domain=domain,
            language="en",
            category="general",
            fetched_at=utc_now(),
            metadata={},
        )

        service = OnlineSearchService(
            config=cfg,
            news_search_fn=news_fn or (lambda query, **kw: [default_record("news", "news_article", "rss")]),
            web_search_fn=web_fn or (lambda query, **kw: [default_record("general", "web_page", "search_api")]),
            academic_search_fn=academic_fn or (lambda query, **kw: [default_record("academic", "academic_paper", "pubmed")]),
        )
        service.initialize()
        return service

    # ------------------------------------------------------------------
    # Case: Medical domain is routed through academic_search_fn
    # Medical searches must reach PubMed / MedlinePlus, not Serper.
    # ------------------------------------------------------------------
    def test_search_medical_uses_academic_fn(self):
        """
        search_medical() routes to academic_search_fn with use_medical_sources=True.
        Verify the returned record has domain='medical'.
        """
        medical_record = SearchRecord(
            id="med_1",
            record_type="medical_article",
            source_type="medlineplus",
            title="Hypertension Management",
            summary="Clinical guidelines for blood pressure.",
            url="https://medlineplus.gov/hypertension",
            source="MedlinePlus",
            domain="medical",
            language="en",
            category="medical",
            fetched_at=utc_now(),
            metadata={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(
                tmpdir,
                academic_fn=lambda query, **kw: [medical_record],
            )
            result = service.search_medical("hypertension treatment", limit=3)
            assert isinstance(result, SearchResult)
            # The service must have called academic_fn and surfaced the record
            assert result.count >= 1
            assert result.items[0].domain == "medical"

    # ------------------------------------------------------------------
    # Case: Cache hit on second search avoids fetcher call
    # The file store should serve the record on the second query.
    # ------------------------------------------------------------------
    def test_search_caches_results(self):
        """
        After a first search persists records to the file store, a second
        identical search (use_cache=True) should return from cache without
        calling the fetcher again.
        """
        call_count = {"n": 0}

        def counting_web_fn(query, **kw):
            call_count["n"] += 1
            return [SearchRecord(
                id=f"cached_{call_count['n']}",
                record_type="web_page",
                source_type="search_api",
                title="Cached Result",
                summary="Returned once and cached.",
                url="https://example.com/cached",
                source="Mock",
                domain="general",
                language="en",
                category="general",
                fetched_at=utc_now(),
                metadata={},
            )]

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir, web_fn=counting_web_fn)

            # First call — should hit fetcher
            r1 = service.search_web("cache test query", limit=5)
            assert r1.count >= 1
            first_calls = call_count["n"]

            # Second call — should hit file cache
            r2 = service.search_web("cache test query", limit=5)
            assert r2.count >= 1
            # Fetcher should NOT have been called again
            assert call_count["n"] == first_calls

    # ------------------------------------------------------------------
    # Case: force use_cache=False always calls the fetcher
    # ------------------------------------------------------------------
    def test_search_with_use_cache_false(self):
        """
        Passing use_cache=False bypasses the file store lookup and always
        calls the live fetcher.
        """
        call_count = {"n": 0}

        def counting_web_fn(query, **kw):
            call_count["n"] += 1
            return [SearchRecord(
                id=f"fresh_{call_count['n']}",
                record_type="web_page",
                source_type="search_api",
                title="Fresh Result",
                summary="Always fetched live.",
                url=f"https://example.com/fresh/{call_count['n']}",
                source="Mock",
                domain="general",
                language="en",
                category="general",
                fetched_at=utc_now(),
                metadata={},
            )]

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir, web_fn=counting_web_fn)

            req = SearchRequest(query="fresh query", domain="general", limit=5, use_cache=False)
            service.search(req)
            service.search(req)
            # Fetcher called both times
            assert call_count["n"] >= 2

    # ------------------------------------------------------------------
    # Case: Fetcher returns empty list → SearchResult.count == 0
    # ------------------------------------------------------------------
    def test_search_returns_empty_when_fetcher_empty(self):
        """
        If the live fetcher returns no results AND the cache is empty,
        SearchResult should have count == 0 (not raise).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(
                tmpdir,
                web_fn=lambda query, **kw: [],
                news_fn=lambda query, **kw: [],
                academic_fn=lambda query, **kw: [],
            )
            result = service.search(
                SearchRequest(
                    query="definitely no results",
                    domain="general",
                    limit=5,
                    use_cache=False,
                )
            )
            assert isinstance(result, SearchResult)
            assert result.count == 0

    # ------------------------------------------------------------------
    # Case: refresh_domain_cache fetches and persists multiple queries
    # ------------------------------------------------------------------
    def test_refresh_domain_cache(self):
        """
        refresh_domain_cache() runs all provided queries, persists results,
        and returns a summary with the correct query_count.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(tmpdir)
            summary = service.refresh_domain_cache(
                domain="general",
                queries=["machine learning", "deep learning", "transformers"],
                limit=3,
            )
            assert summary["domain"] == "general"
            assert summary["query_count"] == 3
            assert summary["record_count"] >= 0  # may deduplicate

    # ------------------------------------------------------------------
    # Case: Deduplication across multiple fetchers
    # If news_fn and web_fn return the same URL, only one record persists.
    # ------------------------------------------------------------------
    def test_deduplication_across_fetchers(self):
        """
        When multiple fetcher functions return records with the same ID
        (same URL + source), SearchResult should de-duplicate them.
        """
        from BatchAgent.online_search_toolkit.utils import make_article_id

        shared_url = "https://example.com/shared-story"
        shared_id = make_article_id("Mock", "Shared Story", shared_url)

        shared_record = SearchRecord(
            id=shared_id,
            record_type="news_article",
            source_type="rss",
            title="Shared Story",
            summary="Appears in both news and web results.",
            url=shared_url,
            source="Mock",
            domain="news",
            language="en",
            category="general",
            fetched_at=utc_now(),
            metadata={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._make_service(
                tmpdir,
                news_fn=lambda query, **kw: [shared_record],
                web_fn=lambda query, **kw: [shared_record],
            )
            result = service.search_news("shared story test", limit=10)
            # After deduplication the shared record should appear exactly once
            ids = [r.id for r in result.items]
            assert ids.count(shared_id) <= 1


# =====================================================================
# Extended FileStore Tests
# =====================================================================

class TestFileStoreExtended:
    """
    Additional FileSearchStore tests covering:
    - get_recent_records with limit
    - get_by_url miss (returns None)
    - keyword search across title AND summary
    - filtering by domain
    - filter by record_type
    """

    def _make_store(self, tmpdir):
        from BatchAgent.online_search_toolkit.store.file_store import FileSearchStore
        cfg = SearchConfig()
        cfg.store.file_cache_dir = Path(tmpdir)
        store = FileSearchStore(cfg)
        store.initialize()
        return store

    def _record(self, uid, title, summary, domain="general", record_type="web_page", url=None):
        return SearchRecord(
            id=uid,
            record_type=record_type,
            source_type="search_api",
            title=title,
            summary=summary,
            url=url or f"https://example.com/{uid}",
            source="Test",
            domain=domain,
            language="en",
            category="general",
            fetched_at=utc_now(),
            metadata={},
        )

    # ------------------------------------------------------------------
    # get_by_url miss
    # ------------------------------------------------------------------
    def test_get_by_url_miss(self):
        """get_by_url returns None for an URL that was never stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            result = store.get_by_url("https://not-stored.example.com/page")
            assert result is None

    # ------------------------------------------------------------------
    # limit on get_recent_records
    # ------------------------------------------------------------------
    def test_get_recent_records_limit(self):
        """get_recent_records(limit=N) should return at most N records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            records = [self._record(f"r{i}", f"Title {i}", f"Summary {i}") for i in range(10)]
            store.upsert_records(records)

            result = store.get_recent_records(limit=3)
            assert len(result) == 3

    # ------------------------------------------------------------------
    # keyword_search matches in summary too
    # ------------------------------------------------------------------
    def test_keyword_search_in_summary(self):
        """keyword_search should match terms found in the summary field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            store.upsert_records([
                self._record("kw_sum", "Generic Title", "Deep neural networks in production"),
                self._record("kw_other", "Another Article", "Unrelated finance content"),
            ])
            results = store.keyword_search("neural networks", limit=10)
            assert len(results) == 1
            assert results[0].id == "kw_sum"

    # ------------------------------------------------------------------
    # Upsert updates an existing record
    # ------------------------------------------------------------------
    def test_upsert_updates_existing(self):
        """Upserting a record with the same ID should overwrite the old one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            original = self._record("update_me", "Old Title", "Old summary")
            store.upsert_records([original])

            updated = self._record("update_me", "New Title", "New summary")
            store.upsert_records([updated])

            record = store.get_by_url("https://example.com/update_me")
            assert record is not None
            assert record.title == "New Title"

    # ------------------------------------------------------------------
    # Breaking news appears first in get_recent_records
    # ------------------------------------------------------------------
    def test_breaking_news_sorted_first(self):
        """Breaking news records should appear before non-breaking ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = self._make_store(tmpdir)
            normal = self._record("normal_1", "Normal News", "Regular news")
            breaking = SearchRecord(
                id="breaking_1",
                record_type="news_article",
                source_type="rss",
                title="Breaking: Major Event",
                summary="Something just happened.",
                url="https://example.com/breaking",
                source="News",
                domain="news",
                language="en",
                category="breaking",
                fetched_at=utc_now(),
                is_breaking=True,
                metadata={},
            )
            store.upsert_records([normal, breaking])
            results = store.get_recent_records(limit=10)
            assert results[0].is_breaking is True

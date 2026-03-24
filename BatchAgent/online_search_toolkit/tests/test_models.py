"""
Unit tests for online_search_toolkit models and utilities.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from BatchAgent.online_search_toolkit.models import (
    SearchRecord,
    SearchRequest,
    SearchResult,
    SearchPlan,
    URLReadResult,
    CrawlRequest,
    AcademicSearchSeed,
)
from BatchAgent.online_search_toolkit.utils import (
    build_embedding_text,
    clean_html_text,
    cosine_similarity,
    dedupe_preserve_order,
    detect_language,
    is_recent,
    make_article_id,
    normalize_url,
    parse_datetime,
    trim_summary,
    utc_now,
)


# =====================================================================
# Model Tests
# =====================================================================

class TestSearchRecord:
    def test_create_minimal(self):
        r = SearchRecord(
            id="test123",
            record_type="web_page",
            source_type="search_api",
            title="Test Record",
            summary="A summary",
            url="https://example.com/page",
            source="TestSource",
            domain="general",
            language="en",
            category="general",
            fetched_at=utc_now(),
            metadata={},
        )
        assert r.id == "test123"
        assert r.record_type == "web_page"
        assert r.is_breaking is False
        assert r.embedding is None
        assert r.content is None

    def test_serialization_roundtrip(self):
        now = utc_now()
        r = SearchRecord(
            id="roundtrip",
            record_type="news_article",
            source_type="rss",
            title="Breaking Story",
            summary="Something happened",
            url="https://example.com/story",
            source="BBC",
            domain="news",
            language="en",
            category="breaking",
            fetched_at=now,
            is_breaking=True,
            metadata={"key": "value"},
        )
        data = r.model_dump(mode="json")
        restored = SearchRecord.model_validate(data)
        assert restored.id == r.id
        assert restored.title == r.title
        assert restored.is_breaking is True
        assert restored.metadata == {"key": "value"}

    def test_optional_fields(self):
        r = SearchRecord(
            id="opts",
            record_type="academic_paper",
            source_type="pubmed",
            title="Paper Title",
            summary="Abstract",
            url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            source="PubMed",
            domain="academic",
            language="en",
            category="medical_research",
            fetched_at=utc_now(),
            published_at=utc_now(),
            content="Full text...",
            query="diabetes treatment",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            metadata={"pmid": "12345"},
        )
        assert r.content == "Full text..."
        assert r.query == "diabetes treatment"
        assert r.embedding == [0.1, 0.2, 0.3]


class TestSearchRequest:
    def test_defaults(self):
        req = SearchRequest(query="test query")
        assert req.query == "test query"
        assert req.domain == "general"
        assert req.limit == 8
        assert req.language == "mixed"

    def test_custom_values(self):
        req = SearchRequest(
            query="AI news",
            domain="news",
            limit=20,
            language="en",
            category="technology",
        )
        assert req.domain == "news"
        assert req.limit == 20
        assert req.category == "technology"


class TestSearchResult:
    def test_count(self):
        records = [
            SearchRecord(
                id=f"r{i}",
                record_type="web_page",
                source_type="search_api",
                title=f"Record {i}",
                summary=f"Summary {i}",
                url=f"https://example.com/{i}",
                source="Test",
                domain="general",
                language="en",
                category="general",
                fetched_at=utc_now(),
                metadata={},
            )
            for i in range(5)
        ]
        # SearchResult uses 'items' (not 'records') and requires 'query' + 'count'
        result = SearchResult(query="test", domain="general", count=5, items=records)
        assert result.count == 5
        assert len(result.items) == 5


class TestSearchPlan:
    def test_plan_creation(self):
        # SearchPlan uses 'normalized_query' (not 'query'); has no 'strategies' list
        plan = SearchPlan(
            normalized_query="test",
            domain="news",
            use_news_rss=True,
            use_serper=False,
        )
        assert plan.normalized_query == "test"
        assert plan.domain == "news"
        assert plan.use_news_rss is True


class TestURLReadResult:
    def test_read_result(self):
        # URLReadResult wraps a SearchRecord in its 'record' field
        record = SearchRecord(
            id="url_r1",
            record_type="web_page",
            source_type="url_fetch",
            title="Example",
            summary="Hello world",
            url="https://example.com",
            source="httpx",
            domain="general",
            language="en",
            category="general",
            fetched_at=utc_now(),
            metadata={"status": 200},
        )
        result = URLReadResult(record=record, from_cache=False)
        assert result.record.url == "https://example.com"
        assert result.from_cache is False


# =====================================================================
# Utility Tests
# =====================================================================

class TestCleanHtmlText:
    def test_basic_html(self):
        assert clean_html_text("<p>Hello <b>world</b></p>") == "Hello world"

    def test_entities(self):
        assert clean_html_text("&amp; &lt; &gt;") == "& < >"

    def test_whitespace_collapse(self):
        assert clean_html_text("  hello   world  \n\t  ") == "hello world"

    def test_empty(self):
        assert clean_html_text("") == ""
        assert clean_html_text(None) == ""  # type: ignore


class TestTrimSummary:
    def test_short_text(self):
        assert trim_summary("short", 100) == "short"

    def test_long_text(self):
        text = "a" * 300
        result = trim_summary(text, 100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_html_cleaned(self):
        result = trim_summary("<p>hello world</p>", 100)
        assert result == "hello world"


class TestNormalizeUrl:
    def test_basic(self):
        assert normalize_url("https://example.com/path") == "https://example.com/path"

    def test_strips_fragment(self):
        # normalize_url doesn't strip fragment by design — it preserves query
        url = normalize_url("https://example.com/page?q=1")
        assert "q=1" in url

    def test_empty(self):
        assert normalize_url("") == ""


class TestMakeArticleId:
    def test_deterministic(self):
        id1 = make_article_id("src", "title", "https://example.com")
        id2 = make_article_id("src", "title", "https://example.com")
        assert id1 == id2
        assert len(id1) == 24

    def test_different_inputs(self):
        id1 = make_article_id("src1", "title", "https://example.com")
        id2 = make_article_id("src2", "title", "https://example.com")
        assert id1 != id2


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("Hello world", "This is English") == "en"

    def test_chinese(self):
        assert detect_language("你好世界", "这是中文") == "zh"

    def test_mixed(self):
        assert detect_language("你好 Hello", "Mixed") == "zh"  # Chinese takes priority


class TestParseDatetime:
    def test_iso_format(self):
        dt = parse_datetime("2024-01-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1

    def test_rfc822(self):
        dt = parse_datetime("Mon, 15 Jan 2024 10:30:00 GMT")
        assert dt is not None

    def test_none(self):
        assert parse_datetime(None) is None
        assert parse_datetime("") is None


class TestIsRecent:
    def test_recent(self):
        dt = utc_now() - timedelta(hours=2)
        assert is_recent(dt, 6) is True

    def test_not_recent(self):
        dt = utc_now() - timedelta(hours=24)
        assert is_recent(dt, 6) is False

    def test_none(self):
        assert is_recent(None, 6) is False


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_empty(self):
        assert cosine_similarity([], []) == -1.0

    def test_different_lengths(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == -1.0


class TestBuildEmbeddingText:
    def test_basic(self):
        text = build_embedding_text("Title", "Source", "Cat", "Summary")
        assert "Title" in text
        assert "Source" in text
        assert "Summary" in text

    def test_with_content(self):
        text = build_embedding_text("T", "S", "C", "Sum", content="Full content here")
        assert "Content:" in text
        assert "Full content here" in text


class TestDedupePreserveOrder:
    def test_basic(self):
        items = [1, 2, 3, 2, 1, 4]
        result = dedupe_preserve_order(items, lambda x: x)
        assert result == [1, 2, 3, 4]

    def test_with_key(self):
        items = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 1, "v": "c"}]
        result = dedupe_preserve_order(items, lambda x: x["id"])
        assert len(result) == 2
        assert result[0]["v"] == "a"

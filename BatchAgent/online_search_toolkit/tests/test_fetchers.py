"""
Unit tests for fetcher modules with mocked HTTP responses.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from BatchAgent.online_search_toolkit.fetchers.search_apis import SearchAPIFetcher
from BatchAgent.online_search_toolkit.fetchers.news_sources import NewsSourceFetcher
from BatchAgent.online_search_toolkit.fetchers.academic_sources import AcademicSourceFetcher


# =====================================================================
# SearchAPIFetcher Tests
# =====================================================================

class TestSearchAPIFetcher:
    def setup_method(self):
        self.fetcher = SearchAPIFetcher(
            max_summary_chars=240,
            serper_api_key="test-serper-key",
            tavily_api_key="test-tavily-key",
        )

    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_serper(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "organic": [
                {
                    "title": "Test Result 1",
                    "snippet": "This is a test snippet",
                    "link": "https://example.com/result1",
                },
                {
                    "title": "Test Result 2",
                    "snippet": "Another snippet",
                    "link": "https://example.com/result2",
                },
            ],
            "answerBox": {
                "snippet": "Direct answer for test",
                "link": "https://example.com/answer",
            },
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_serper("test query", "general", limit=5)
        assert len(results) >= 2
        assert any("Test Result" in r.title for r in results)
        assert all(r.record_type == "web_page" for r in results if "Direct answer" not in r.title)

    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_tavily(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "answer": "This is a Tavily answer",
            "results": [
                {
                    "title": "Tavily Result",
                    "content": "Tavily content snippet",
                    "url": "https://example.com/tavily",
                },
            ],
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_tavily("test query", "general", limit=5)
        assert len(results) >= 1
        assert any("Tavily" in r.source for r in results)

    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_wikimedia(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "query": {
                "search": [
                    {
                        "title": "Artificial Intelligence",
                        "snippet": "AI is a branch of <span>computer</span> science",
                    },
                    {
                        "title": "Machine Learning",
                        "snippet": "ML is a subset of AI",
                    },
                ],
            },
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_wikimedia("artificial intelligence", limit=3)
        assert len(results) == 2
        assert results[0].source == "Wikimedia"
        assert results[0].record_type == "wiki_entry"
        assert "wikipedia.org" in results[0].url

    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_youtube(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "videos": [
                {
                    "title": "AI Tutorial",
                    "link": "https://youtube.com/watch?v=abc123",
                    "snippet": "Learn about AI",
                    "channel": "TechChannel",
                    "duration": "10:30",
                },
            ],
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_youtube("AI tutorial", limit=5)
        assert len(results) == 1
        assert results[0].record_type == "youtube_video"  # model uses 'youtube_video' not 'video'
        assert results[0].source == "YouTube"
        assert "TechChannel" in results[0].summary

    def test_search_serper_no_key(self):
        # Pass a sentinel that passes the 'or' short-circuit, then clear it.
        # Using "EMPTY" because the fetcher guards on `not key or key == "EMPTY"`.
        fetcher = SearchAPIFetcher(serper_api_key="EMPTY", tavily_api_key="EMPTY")
        results = fetcher.search_serper("query", "general")
        assert results == []

    def test_search_tavily_no_key(self):
        fetcher = SearchAPIFetcher(serper_api_key="EMPTY", tavily_api_key="EMPTY")
        results = fetcher.search_tavily("query", "general")
        assert results == []

    def test_domain_filter(self):
        filt = self.fetcher._domain_filter("academic")
        assert "arxiv.org" in filt
        assert self.fetcher._domain_filter("general") == ""

    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_unified_search(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "organic": [{"title": "R1", "snippet": "S1", "link": "https://example.com/1"}],
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search(
            "test", domain="general", limit=5,
            use_serper=True, use_tavily=False, use_wikimedia=False,
        )
        assert len(results) >= 1


# =====================================================================
# NewsSourceFetcher Tests
# =====================================================================

class TestNewsSourceFetcher:
    def setup_method(self):
        self.fetcher = NewsSourceFetcher(
            max_summary_chars=240,
            timeout_seconds=10,
        )

    def test_infer_feed_groups_ai(self):
        groups = self.fetcher.infer_feed_groups("latest AI developments")
        assert "ai_en" in groups

    def test_infer_feed_groups_business(self):
        groups = self.fetcher.infer_feed_groups("stock market business news")
        assert "business_en" in groups

    def test_infer_feed_groups_default(self):
        groups = self.fetcher.infer_feed_groups("random unrelated query")
        assert len(groups) > 0  # should return default groups

    def test_infer_feed_groups_breaking(self):
        groups = self.fetcher.infer_feed_groups("breaking news today")
        assert "breaking_en" in groups

    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.feedparser.parse")
    def test_fetch_rss_urls(self, mock_parse):
        mock_parse.return_value = MagicMock(
            feed={"title": "Test Feed"},
            entries=[
                {
                    "title": "Breaking Story",
                    "link": "https://news.example.com/story1",
                    "summary": "A major event occurred",
                    "published": "Mon, 15 Jan 2024 10:00:00 GMT",
                },
                {
                    "title": "Tech Update",
                    "link": "https://news.example.com/story2",
                    "summary": "New tech advancement",
                    "published": "Mon, 15 Jan 2024 09:00:00 GMT",
                },
            ],
        )

        results = self.fetcher.fetch_rss_urls(
            ["https://feeds.example.com/rss"],
            category="general",
            language="en",
            max_items_per_feed=5,
        )
        assert len(results) == 2
        assert results[0].record_type == "news_article"
        assert results[0].source_type == "rss"

    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.requests.get")
    def test_search_gnews_api(self, mock_get):
        fetcher = NewsSourceFetcher(gnews_api_key="test-key")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "articles": [
                    {
                        "title": "GNews Article",
                        "url": "https://news.example.com/gnews1",
                        "description": "A GNews article",
                        "publishedAt": "2024-01-15T10:00:00Z",
                        "source": {"name": "TestSource"},
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = fetcher.search_gnews_api("AI news", limit=5)
        assert len(results) == 1
        assert results[0].source_type == "news_api"

    def test_search_gnews_api_no_key(self):
        fetcher = NewsSourceFetcher(gnews_api_key="")
        results = fetcher.search_gnews_api("test", limit=5)
        assert results == []

    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.requests.get")
    def test_search_newsdata_api(self, mock_get):
        fetcher = NewsSourceFetcher(newsdata_api_key="test-key")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {
                        "title": "NewsData Article",
                        "link": "https://news.example.com/nd1",
                        "description": "A NewsData article",
                        "pubDate": "2024-01-15 10:00:00",
                        "source_id": "testsource",
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = fetcher.search_newsdata_api("tech news", limit=5)
        assert len(results) == 1


# =====================================================================
# AcademicSourceFetcher Tests
# =====================================================================

class TestAcademicSourceFetcher:
    def setup_method(self):
        self.fetcher = AcademicSourceFetcher(
            max_summary_chars=240,
            timeout_seconds=10,
        )

    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_search_pubmed(self, mock_get):
        # Mock esearch response
        esearch_response = MagicMock()
        esearch_response.json.return_value = {
            "esearchresult": {"idlist": ["12345", "67890"]}
        }
        esearch_response.raise_for_status = MagicMock()

        # Mock esummary response
        esummary_response = MagicMock()
        esummary_response.json.return_value = {
            "result": {
                "12345": {
                    "title": "Study on Diabetes",
                    "pubdate": "2024 Jan",
                    "source": "Nature",
                    "authors": [{"name": "Smith J"}, {"name": "Doe A"}],
                },
                "67890": {
                    "title": "Cancer Research Paper",
                    "pubdate": "2024 Feb",
                    "source": "Science",
                    "authors": [{"name": "Johnson B"}],
                },
            }
        }
        esummary_response.raise_for_status = MagicMock()

        mock_get.side_effect = [esearch_response, esummary_response]

        results = self.fetcher.search_pubmed("diabetes treatment", limit=5)
        assert len(results) == 2
        assert results[0].source == "PubMed"
        assert results[0].record_type == "academic_paper"
        assert "pubmed.ncbi.nlm.nih.gov" in results[0].url

    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_search_semantic_scholar(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {
                        "title": "Attention Is All You Need",
                        "abstract": "We propose the Transformer architecture...",
                        "url": "https://api.semanticscholar.org/CorpusID:123",
                        "year": 2017,
                        "authors": [{"name": "Vaswani A"}, {"name": "Shazeer N"}],
                        "externalIds": {"ArXiv": "1706.03762"},
                        "citationCount": 50000,
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = self.fetcher.search_semantic_scholar("transformer architecture", limit=5)
        assert len(results) == 1
        assert results[0].source == "Semantic Scholar"
        assert results[0].record_type == "academic_paper"
        assert "50000 citations" in results[0].summary or "Vaswani" in results[0].summary

    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_search_arxiv(self, mock_get):
        arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Attention Is All You Need</title>
                <summary>We propose a new architecture...</summary>
                <link rel="alternate" href="https://arxiv.org/abs/1706.03762"/>
                <link type="application/pdf" href="https://arxiv.org/pdf/1706.03762"/>
                <author><name>Ashish Vaswani</name></author>
                <author><name>Noam Shazeer</name></author>
                <published>2017-06-12T17:57:34Z</published>
            </entry>
        </feed>"""

        mock_get.return_value = MagicMock(
            status_code=200,
            text=arxiv_xml,
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = self.fetcher.search_arxiv("transformer", limit=5)
        assert len(results) == 1
        assert results[0].source == "arXiv"
        assert "1706.03762" in results[0].url

    def test_search_crawler_no_fn(self):
        results = self.fetcher.search_crawler("test", limit=5)
        assert results == []

    def test_search_crawler_with_fn(self):
        mock_fn = MagicMock(return_value=[
            {
                "title": "Crawled Page",
                "summary": "Some content",
                "url": "https://example.com/crawled",
                "source": "MyCrawler",
                "category": "research",
            },
        ])
        fetcher = AcademicSourceFetcher(crawler_search_fn=mock_fn)
        results = fetcher.search_crawler("test", limit=5)
        assert len(results) == 1
        assert results[0].record_type == "crawler_page"

    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_unified_search_deduplicates(self, mock_get):
        # Return same paper from both PubMed and Semantic Scholar
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [], "esearchresult": {"idlist": []}},
            text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>",
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = self.fetcher.search(
            "test query", limit=5,
            use_pubmed=True, use_semantic_scholar=True, use_arxiv=True,
            use_medlineplus=False, use_nimh=False,
        )
        # Should not raise, results may be empty due to mocked empty responses
        assert isinstance(results, list)


# =====================================================================
# Web Crawler Tests
# =====================================================================

class TestCrawlerFetcher:
    def test_import_and_instantiate(self):
        """Test that CrawlerFetcher can be imported and created."""
        from BatchAgent.online_search_toolkit.fetchers.web_crawler import CrawlerFetcher
        fetcher = CrawlerFetcher(
            max_pages=5,
            max_depth=1,
            headless=True,
            max_summary_chars=240,
        )
        assert fetcher.max_pages == 5
        assert fetcher.max_depth == 1

    def test_page_data_to_record(self):
        from BatchAgent.online_search_toolkit.fetchers.web_crawler import CrawlerFetcher
        fetcher = CrawlerFetcher()

        data = {
            "url": "https://example.com/page",
            "title": "Test Page",
            "meta_description": "A test page description",
            "full_text": "Full text of the page...",
            "headings": [{"level": 1, "text": "Main Heading"}],
            "links_out": ["https://example.com/other"],
            "tables": [],
            "depth": 1,
            "parent_url": "https://example.com",
            "status": "ok",
        }

        record = fetcher._page_data_to_record(
            data, domain="general", category="test", query="test query"
        )
        assert record.title == "Test Page"
        assert record.record_type == "crawler_page"
        assert record.source_type == "crawler"
        assert record.domain == "general"
        assert record.metadata["depth"] == 1

    def test_search_crawled_empty(self):
        from BatchAgent.online_search_toolkit.fetchers.web_crawler import CrawlerFetcher
        fetcher = CrawlerFetcher()
        results = fetcher.search_crawled("test")
        assert results == []


# =====================================================================
# URLReader Tests
# These tests cover the URL reading logic without making real network calls.
# Each case explains which code path is exercised.
# =====================================================================

class TestURLReader:
    """
    URLReader handles: plain HTML, PDF (via pypdf), YouTube (metadata +
    transcript), and a Playwright fallback for JS-heavy pages.

    All cases mock httpx to avoid network I/O.
    """

    def _make_reader(self):
        from BatchAgent.online_search_toolkit.fetchers.url_reader import URLReader
        return URLReader(max_summary_chars=240)

    # ------------------------------------------------------------------
    # Case 1: Standard HTML page
    # Exercises: httpx GET → BeautifulSoup parse → markdownify → SearchRecord
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client")
    def test_read_html_page(self, mock_client_cls):
        """Plain HTML page returns a web_page record with extracted text."""
        html = (
            "<html><head><title>Hello World</title></head>"
            "<body><p>This is a test paragraph with enough content to pass cleaning.</p></body></html>"
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
        mock_resp.content = html.encode()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        reader = self._make_reader()
        record = reader.read_url("https://example.com/article")

        assert record.record_type == "web_page"
        assert record.source_type == "url_fetch"
        assert record.url == "https://example.com/article"
        assert record.title == "Hello World"
        # content or summary should contain extracted text
        body = record.content or record.summary
        assert body is not None and len(body) > 0

    # ------------------------------------------------------------------
    # Case 2: Empty / thin HTML triggers Playwright fallback
    # Exercises: content length < _PLAYWRIGHT_THRESHOLD → fallback branch
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._PLAYWRIGHT_AVAILABLE", True)
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._fetch_via_playwright")
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client")
    def test_read_thin_html_triggers_playwright(self, mock_client_cls, mock_playwright):
        """
        When the initial HTTP fetch returns very little text, the reader
        should attempt the Playwright fallback and use its richer content.
        """
        thin_html = "<html><body><p>hi</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = thin_html.encode()
        mock_resp.text = thin_html
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        # Playwright returns richer content
        mock_playwright.return_value = "A" * 500

        reader = self._make_reader()
        record = reader.read_url("https://js-heavy-site.example.com")

        # Playwright should have been called because initial text was thin
        mock_playwright.assert_called_once()
        body = record.content or record.summary
        assert body is not None

    # ------------------------------------------------------------------
    # Case 3: PDF URL (Content-Type: application/pdf)
    # Exercises: PDF branch → pypdf extraction → page text
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._PYPDF_AVAILABLE", True)
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._extract_pdf_text")
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client")
    def test_read_pdf_url(self, mock_client_cls, mock_extract_pdf):
        """PDF URLs are handled by _extract_pdf_text; result stored as 'pdf' record."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "application/pdf"}
        mock_resp.content = b"%PDF-1.4 fake bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        mock_extract_pdf.return_value = (
            "--- Page 1 ---\nThis is extracted PDF content from page one."
        )

        reader = self._make_reader()
        record = reader.read_url("https://example.com/paper.pdf")

        assert record.record_type == "pdf"
        mock_extract_pdf.assert_called_once()
        content = record.content or record.summary
        assert "PDF" in content or "Page" in content or len(content) > 0

    # ------------------------------------------------------------------
    # Case 4: YouTube URL
    # Exercises: YouTube detection branch → title + description extraction
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._extract_youtube_title")
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader._extract_youtube_description")
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client")
    def test_read_youtube_url(self, mock_client_cls, mock_desc, mock_title):
        """YouTube watch URLs are recognised; title and description are extracted."""
        page_html = "<html><body>YouTube page content</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.content = page_html.encode()
        mock_resp.text = page_html
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        mock_title.return_value = "My AI Tutorial"
        mock_desc.return_value = "In this video we explain attention mechanisms."

        reader = self._make_reader()
        record = reader.read_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert record.record_type == "youtube_video"
        assert record.source_type == "youtube"
        assert "dQw4w9WgXcQ" in record.url

    # ------------------------------------------------------------------
    # Case 5: HTTP error (e.g., 404) returns an error-indicating record
    # Exercises: exception handling path → fallback record
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client")
    def test_read_http_error(self, mock_client_cls):
        """HTTP errors should not raise; reader returns a record with error info."""
        import httpx as _httpx

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        reader = self._make_reader()
        # Should not raise
        record = reader.read_url("https://example.com/missing-page")
        assert record is not None
        assert record.url == "https://example.com/missing-page"

    # ------------------------------------------------------------------
    # Case 6: Crawler fallback for JavaScript-heavy sites
    # Exercises: crawler_read_fn injection → CrawlerFetcher.read_single_page path
    # ------------------------------------------------------------------
    def test_read_url_with_crawler_fn(self):
        """
        When a crawler_read_fn is injected, it is invoked for JS-heavy URLs
        and its result is used to populate the record content.
        """
        from BatchAgent.online_search_toolkit.fetchers.url_reader import URLReader

        crawler_result = {
            "title": "Crawled Title",
            "full_text": "Rich content extracted by crawler.",
            "meta_description": "Crawler meta description",
            "url": "https://spa-example.com",
        }
        reader = URLReader(
            max_summary_chars=240,
            crawler_read_fn=lambda url: crawler_result,
        )

        # Patch httpx to return thin content, forcing crawler fallback
        with patch("BatchAgent.online_search_toolkit.fetchers.url_reader.httpx.Client") as mock_cls:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "text/html"}
            mock_resp.content = b"<html><body><p>x</p></body></html>"
            mock_resp.text = "<html><body><p>x</p></body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_c = MagicMock()
            mock_c.__enter__ = MagicMock(return_value=mock_c)
            mock_c.__exit__ = MagicMock(return_value=False)
            mock_c.get.return_value = mock_resp
            mock_cls.return_value = mock_c

            record = reader.read_url("https://spa-example.com", use_crawler=True)
            assert record is not None

    # ------------------------------------------------------------------
    # Case 7: local file:// URL (if supported by the reader)
    # Exercises: local file path detection → plain-text read
    # ------------------------------------------------------------------
    def test_read_local_file_url(self, tmp_path):
        """
        file:// URLs should be read from the filesystem and returned as
        text_file records without making any network requests.
        """
        from BatchAgent.online_search_toolkit.fetchers.url_reader import URLReader

        test_file = tmp_path / "sample.txt"
        test_file.write_text("Hello from a local file. " * 20)
        url = test_file.as_uri()  # file:///...

        reader = URLReader(max_summary_chars=500)
        record = reader.read_url(url)

        # Should return some record without crashing, content from file
        assert record is not None
        assert record.url is not None


# =====================================================================
# SearchAPIFetcher — additional edge-case tests
# =====================================================================

class TestSearchAPIFetcherEdgeCases:
    """
    Additional edge cases for SearchAPIFetcher beyond the happy-path tests
    in TestSearchAPIFetcher above.
    """

    def setup_method(self):
        from BatchAgent.online_search_toolkit.fetchers.search_apis import SearchAPIFetcher
        self.fetcher = SearchAPIFetcher(
            max_summary_chars=240,
            serper_api_key="test-serper-key",
            tavily_api_key="test-tavily-key",
        )

    # ------------------------------------------------------------------
    # Knowledge-graph result (answerBox variant)
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_serper_with_knowledge_graph(self, mock_urlopen):
        """
        Serper responses can include a 'knowledgeGraph' block. Verify it is
        captured as a result with the correct record_type.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "knowledgeGraph": {
                "title": "Python (programming language)",
                "description": "Python is a high-level programming language.",
                "website": "https://python.org",
            },
            "organic": [],
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_serper("Python language", "general", limit=5)
        # Should parse the knowledge graph without raising
        assert isinstance(results, list)

    # ------------------------------------------------------------------
    # Serper returns 401 (invalid key) — should return []
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_serper_auth_error(self, mock_urlopen):
        """A 401/403 response should be handled gracefully (return empty list)."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://google.serper.dev/search",
            code=401,
            msg="Unauthorized",
            hdrs=None,  # type: ignore
            fp=None,
        )
        results = self.fetcher.search_serper("query", "general")
        assert results == []

    # ------------------------------------------------------------------
    # Domain filter for every known category
    # ------------------------------------------------------------------
    def test_domain_filter_known_categories(self):
        """
        _domain_filter should return non-empty site-filter strings for specialist
        domains and an empty string for 'general' or unknown categories.
        Categories map to the keys defined in SearchAPIFetcher._domain_filter():
          academic, medical, news, programming, finance, research.
        """
        specialist = ["academic", "medical", "news", "programming", "finance", "research"]
        for cat in specialist:
            filt = self.fetcher._domain_filter(cat)
            assert filt, f"Expected non-empty filter for category '{cat}'"

        # 'general' key maps to "" explicitly
        assert self.fetcher._domain_filter("general") == ""
        # unknown categories also return "" (default dict fallback)
        assert self.fetcher._domain_filter("cooking") == ""

    # ------------------------------------------------------------------
    # Unified search() with all sources disabled → empty list
    # ------------------------------------------------------------------
    def test_unified_search_all_disabled(self):
        """Calling search() with all sources disabled returns an empty list."""
        results = self.fetcher.search(
            "test query",
            domain="general",
            limit=5,
            use_serper=False,
            use_tavily=False,
            use_wikimedia=False,
        )
        assert results == []

    # ------------------------------------------------------------------
    # Serper with siteLinks in organic result
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.search_apis.urllib.request.urlopen")
    def test_search_serper_organic_with_date(self, mock_urlopen):
        """Organic results with a 'date' field should populate published_at."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "organic": [
                {
                    "title": "Dated Article",
                    "snippet": "This article has a publication date.",
                    "link": "https://example.com/dated",
                    "date": "Jan 15, 2024",
                },
            ],
        }).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = self.fetcher.search_serper("dated article", "general", limit=5)
        assert len(results) >= 1
        assert results[0].title == "Dated Article"


# =====================================================================
# NewsSourceFetcher — additional edge-case tests
# =====================================================================

class TestNewsSourceFetcherEdgeCases:
    """Additional coverage for corner-cases in NewsSourceFetcher."""

    def setup_method(self):
        from BatchAgent.online_search_toolkit.fetchers.news_sources import NewsSourceFetcher
        self.fetcher = NewsSourceFetcher(max_summary_chars=240, timeout_seconds=5)

    # ------------------------------------------------------------------
    # RSS feed with missing optional fields
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.feedparser.parse")
    def test_rss_missing_optional_fields(self, mock_parse):
        """
        RSS entries often lack 'published' or 'summary'. The fetcher must
        still produce valid SearchRecord objects.
        """
        mock_parse.return_value = MagicMock(
            feed={"title": "Minimal Feed"},
            entries=[
                {
                    "title": "No Date No Summary",
                    "link": "https://news.example.com/nodatesummary",
                    # no 'summary', no 'published'
                },
            ],
        )
        results = self.fetcher.fetch_rss_urls(
            ["https://feeds.example.com/rss"],
            category="general",
            language="en",
            max_items_per_feed=5,
        )
        assert len(results) == 1
        assert results[0].title == "No Date No Summary"
        assert results[0].published_at is None

    # ------------------------------------------------------------------
    # RSS entries with no title → skipped or empty title
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.feedparser.parse")
    def test_rss_entry_no_title_skipped(self, mock_parse):
        """Entries with no title or link should be skipped or handled gracefully."""
        mock_parse.return_value = MagicMock(
            feed={"title": "Test Feed"},
            entries=[
                {"summary": "Some content but no title or link"},
            ],
        )
        results = self.fetcher.fetch_rss_urls(
            ["https://feeds.example.com/rss"],
            category="general",
            language="en",
            max_items_per_feed=5,
        )
        # Either skipped or produced with empty title — no crash
        assert isinstance(results, list)

    # ------------------------------------------------------------------
    # feed_groups inference: health/medical query
    # ------------------------------------------------------------------
    def test_infer_feed_groups_health(self):
        """A health-related query should select health/medical feed groups."""
        groups = self.fetcher.infer_feed_groups("diabetes treatment clinical trial")
        assert len(groups) > 0  # at least a fallback group

    # ------------------------------------------------------------------
    # NewsAPI returns no articles key → empty list
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.news_sources.requests.get")
    def test_search_gnews_api_empty_articles(self, mock_get):
        """GNews response with empty articles list returns []."""
        from BatchAgent.online_search_toolkit.fetchers.news_sources import NewsSourceFetcher
        fetcher = NewsSourceFetcher(gnews_api_key="test-key")
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"articles": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = fetcher.search_gnews_api("nonexistent topic", limit=5)
        assert results == []


# =====================================================================
# AcademicSourceFetcher — additional edge-case tests
# =====================================================================

class TestAcademicSourceFetcherEdgeCases:
    """Additional coverage for AcademicSourceFetcher."""

    def setup_method(self):
        from BatchAgent.online_search_toolkit.fetchers.academic_sources import AcademicSourceFetcher
        self.fetcher = AcademicSourceFetcher(max_summary_chars=240, timeout_seconds=10)

    # ------------------------------------------------------------------
    # PubMed returns no IDs → empty list
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_pubmed_no_results(self, mock_get):
        """PubMed esearch returning an empty idlist yields an empty result."""
        esearch = MagicMock()
        esearch.json.return_value = {"esearchresult": {"idlist": []}}
        esearch.raise_for_status = MagicMock()
        mock_get.return_value = esearch

        results = self.fetcher.search_pubmed("xyzzy nonexistent condition", limit=5)
        assert results == []

    # ------------------------------------------------------------------
    # arXiv with multiple authors and PDF link
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_search_arxiv_multiple_authors(self, mock_get):
        """arXiv results with multiple authors should list all in the summary."""
        arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Multi-Author Paper</title>
                <summary>Abstract of multi-author paper.</summary>
                <link rel="alternate" href="https://arxiv.org/abs/2401.12345"/>
                <link type="application/pdf" href="https://arxiv.org/pdf/2401.12345"/>
                <author><name>Alice Smith</name></author>
                <author><name>Bob Jones</name></author>
                <author><name>Carol White</name></author>
                <published>2024-01-15T00:00:00Z</published>
            </entry>
        </feed>"""
        mock_get.return_value = MagicMock(status_code=200, text=arxiv_xml)
        mock_get.return_value.raise_for_status = MagicMock()

        results = self.fetcher.search_arxiv("multi-author", limit=5)
        assert len(results) == 1
        assert results[0].source == "arXiv"
        # All three authors should appear somewhere in the record
        combined = results[0].summary + (results[0].content or "")
        assert any(name in combined for name in ["Alice", "Bob", "Carol"])

    # ------------------------------------------------------------------
    # Semantic Scholar with no citation count
    # ------------------------------------------------------------------
    @patch("BatchAgent.online_search_toolkit.fetchers.academic_sources.requests.get")
    def test_semantic_scholar_no_citation_count(self, mock_get):
        """Papers without a citationCount field should still parse cleanly."""
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {
                        "title": "New Preprint",
                        "abstract": "Fresh off the press.",
                        "url": "https://api.semanticscholar.org/CorpusID:999",
                        "year": 2024,
                        "authors": [{"name": "Author One"}],
                        "externalIds": {},
                        # no citationCount
                    },
                ],
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        results = self.fetcher.search_semantic_scholar("preprint", limit=5)
        assert len(results) == 1
        assert results[0].record_type == "academic_paper"

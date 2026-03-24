"""
test_web_tools_adapter.py

Tests for BatchAgent/tools/web_tools.py — the adapter layer that bridges
tool_router.py to online_search_toolkit.OnlineSearchService.

Design notes
------------
Each test class covers a distinct scenario so that failures immediately
point to the broken behaviour.  All tests mock the OnlineSearchService so
they run without real network access or API keys.

Scenario map
------------
TestPerformDomainAwareSearch
  ├── test_returns_formatted_string          — happy path, markdown output
  ├── test_domain_news_routes_to_news        — "news" → domain="news"
  ├── test_domain_academic_routes            — "academic" → domain="academic"
  ├── test_domain_medical_routes             — "medical" → domain="medical"
  ├── test_alias_categories                  — "health", "research", "paper" etc.
  ├── test_unknown_category_falls_to_general — unmapped category → "general"
  ├── test_document_search_fn_appended       — RAG callback output included
  ├── test_service_exception_returns_message — fetcher crash → graceful message
  └── test_no_results_returns_unavailable    — empty items → fallback message

TestFetchAndParseUrl
  ├── test_returns_content_string            — record.content → plain text
  ├── test_uses_record_title_in_header       — title rendered as H1
  ├── test_no_content_returns_error_message  — empty content → informative msg
  └── test_service_exception_returns_error   — read_url crash → error message

TestDomainResolution
  └── test_all_known_mappings                — exhaustive mapping table check

TestFormatRecords
  ├── test_empty_list_returns_empty          — [] → ""
  ├── test_basic_formatting                  — title / URL / source / summary
  ├── test_published_date_for_news           — news_article gets date line
  └── test_no_date_for_web_page              — web_page skips date line
"""
from __future__ import annotations

import importlib
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    record_type="web_page",
    source_type="search_api",
    title="Test Result",
    summary="A useful summary.",
    url="https://example.com/result",
    source="MockSearch",
    domain="general",
    content=None,
    published_at=None,
):
    from BatchAgent.online_search_toolkit.models import SearchRecord
    from BatchAgent.online_search_toolkit.utils import utc_now

    return SearchRecord(
        id="test_id_1",
        record_type=record_type,
        source_type=source_type,
        title=title,
        summary=summary,
        url=url,
        source=source,
        domain=domain,
        language="en",
        category="general",
        fetched_at=utc_now(),
        published_at=published_at,
        content=content,
        metadata={},
    )


def _make_search_result(items):
    from BatchAgent.online_search_toolkit.models import SearchResult
    return SearchResult(
        query="test",
        domain="general",
        count=len(items),
        items=items,
    )


def _patch_service(mock_search_items=None, mock_read_record=None, fail_search=False, fail_read=False):
    """
    Context manager that replaces the module-level _service singleton in
    web_tools with a MagicMock and resets it after the test.
    """
    import BatchAgent.tools.web_tools as wt

    mock_svc = MagicMock()

    if fail_search:
        mock_svc.search.side_effect = RuntimeError("Search failed")
    else:
        mock_svc.search.return_value = _make_search_result(mock_search_items or [])

    if fail_read:
        mock_svc.read_url.side_effect = RuntimeError("Read failed")
    elif mock_read_record is not None:
        mock_svc.read_url.return_value = mock_read_record
    else:
        mock_svc.read_url.return_value = _make_record(content="Default page content.")

    return mock_svc


# =====================================================================
# TestPerformDomainAwareSearch
# =====================================================================

class TestPerformDomainAwareSearch:
    """Tests for perform_domain_aware_search() — the primary agent-facing search."""

    def setup_method(self):
        # Reset the singleton before each test so tests don't bleed into each other
        import BatchAgent.tools.web_tools as wt
        wt._service = None

    def teardown_method(self):
        import BatchAgent.tools.web_tools as wt
        wt._service = None

    # ------------------------------------------------------------------
    # Happy path: results come back and are formatted as markdown
    # ------------------------------------------------------------------
    def test_returns_formatted_string(self):
        """
        When the service returns results, perform_domain_aware_search should
        return a non-empty markdown string containing the result title and URL.
        """
        import BatchAgent.tools.web_tools as wt

        record = _make_record(title="AI Breakthrough", url="https://example.com/ai")
        mock_svc = _patch_service(mock_search_items=[record])

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.perform_domain_aware_search(
                query="AI research",
                category="general",
                serper_api_key="test-key",
            )

        assert isinstance(output, str)
        assert len(output) > 0
        assert "AI Breakthrough" in output
        assert "https://example.com/ai" in output

    # ------------------------------------------------------------------
    # Domain routing: "news" must reach OnlineSearchService with domain="news"
    # ------------------------------------------------------------------
    def test_domain_news_routes_to_news(self):
        """
        Category 'news' should produce a SearchRequest with domain='news',
        enabling RSS and news-API fetchers in OnlineSearchService.
        """
        import BatchAgent.tools.web_tools as wt
        from BatchAgent.online_search_toolkit.models import SearchRequest

        mock_svc = _patch_service(mock_search_items=[_make_record(domain="news")])
        captured_request: list = []

        def capturing_search(req: SearchRequest):
            captured_request.append(req)
            return _make_search_result([_make_record(domain="news")])

        mock_svc.search.side_effect = capturing_search

        with patch.object(wt, "_get_service", return_value=mock_svc):
            wt.perform_domain_aware_search(
                query="latest world news",
                category="news",
                serper_api_key="key",
            )

        assert captured_request, "Service.search() was never called"
        assert captured_request[0].domain == "news"
        assert captured_request[0].use_news_sources is True

    # ------------------------------------------------------------------
    # Domain routing: "academic"
    # ------------------------------------------------------------------
    def test_domain_academic_routes(self):
        """Category 'academic' → domain='academic', use_academic_sources=True."""
        import BatchAgent.tools.web_tools as wt
        from BatchAgent.online_search_toolkit.models import SearchRequest

        mock_svc = _patch_service()
        captured: list = []
        mock_svc.search.side_effect = lambda req: (captured.append(req), _make_search_result([]))[1]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            wt.perform_domain_aware_search("transformer paper", "academic", "key")

        assert captured[0].domain == "academic"
        assert captured[0].use_academic_sources is True

    # ------------------------------------------------------------------
    # Domain routing: "medical"
    # ------------------------------------------------------------------
    def test_domain_medical_routes(self):
        """Category 'medical' → domain='medical', use_medical_sources=True."""
        import BatchAgent.tools.web_tools as wt
        from BatchAgent.online_search_toolkit.models import SearchRequest

        mock_svc = _patch_service()
        captured: list = []
        mock_svc.search.side_effect = lambda req: (captured.append(req), _make_search_result([]))[1]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            wt.perform_domain_aware_search("diabetes treatment", "medical", "key")

        assert captured[0].domain == "medical"
        assert captured[0].use_medical_sources is True

    # ------------------------------------------------------------------
    # Alias categories
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("alias,expected_domain", [
        ("health",    "medical"),
        ("medicine",  "medical"),
        ("biology",   "medical"),
        ("research",  "academic"),
        ("paper",     "academic"),
        ("papers",    "academic"),
    ])
    def test_alias_categories(self, alias, expected_domain):
        """
        Alias categories (health, research, biology, …) must map to the
        correct OnlineSearchService domain so the right fetchers are used.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service()
        captured: list = []
        mock_svc.search.side_effect = lambda req: (captured.append(req), _make_search_result([]))[1]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            wt.perform_domain_aware_search("test query", alias, "key")

        assert captured, f"Service not called for alias '{alias}'"
        assert captured[0].domain == expected_domain, (
            f"Alias '{alias}' → expected domain='{expected_domain}', "
            f"got '{captured[0].domain}'"
        )

    # ------------------------------------------------------------------
    # Unknown category falls back to "general"
    # ------------------------------------------------------------------
    def test_unknown_category_falls_to_general(self):
        """
        An unmapped category string (e.g. 'cooking') must not raise and
        must fall back to domain='general'.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service()
        captured: list = []
        mock_svc.search.side_effect = lambda req: (captured.append(req), _make_search_result([]))[1]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.perform_domain_aware_search("best pasta recipe", "cooking", "key")

        assert captured[0].domain == "general"

    # ------------------------------------------------------------------
    # Document RAG callback is appended when provided
    # ------------------------------------------------------------------
    def test_document_search_fn_appended(self):
        """
        When document_search_fn is provided it should be called and its
        output appended after the main search results.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service(mock_search_items=[_make_record()])
        doc_fn = MagicMock(return_value="Relevant excerpt from your PDF.")

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.perform_domain_aware_search(
                query="summary",
                category="general",
                serper_api_key="key",
                document_search_fn=doc_fn,
            )

        doc_fn.assert_called_once_with("summary", 3)
        assert "Relevant excerpt from your PDF." in output

    # ------------------------------------------------------------------
    # Service exception → graceful degradation, not a crash
    # ------------------------------------------------------------------
    def test_service_exception_returns_message(self):
        """
        If OnlineSearchService.search() raises, the function must catch the
        exception and return a human-readable error string.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service(fail_search=True)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.perform_domain_aware_search("crash test", "general", "key")

        assert isinstance(output, str)
        assert "[Web Search Unavailable]" in output or len(output) > 0

    # ------------------------------------------------------------------
    # Empty results → unavailable message
    # ------------------------------------------------------------------
    def test_no_results_returns_unavailable(self):
        """
        When the fetcher returns zero results and there is no RAG callback,
        the function should return the '[Web Search Unavailable]' message.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service(mock_search_items=[])

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.perform_domain_aware_search("obscure topic", "general", "key")

        assert "[Web Search Unavailable]" in output


# =====================================================================
# TestFetchAndParseUrl
# =====================================================================

class TestFetchAndParseUrl:
    """Tests for fetch_and_parse_url() — reads a URL and returns markdown text."""

    def setup_method(self):
        import BatchAgent.tools.web_tools as wt
        wt._service = None

    def teardown_method(self):
        import BatchAgent.tools.web_tools as wt
        wt._service = None

    # ------------------------------------------------------------------
    # Happy path: record has content
    # ------------------------------------------------------------------
    def test_returns_content_string(self):
        """When record.content is populated, it is returned as the body."""
        import BatchAgent.tools.web_tools as wt

        record = _make_record(
            title="Deep Learning Guide",
            url="https://example.com/dl",
            content="Neural networks consist of layers...",
        )
        mock_svc = _patch_service(mock_read_record=record)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://example.com/dl")

        assert "Neural networks consist of layers" in output
        mock_svc.read_url.assert_called_once_with("https://example.com/dl")

    # ------------------------------------------------------------------
    # Title rendered as H1 header
    # ------------------------------------------------------------------
    def test_uses_record_title_in_header(self):
        """A non-empty title should appear as a markdown H1 at the top."""
        import BatchAgent.tools.web_tools as wt

        record = _make_record(title="Awesome Article", content="Body text here.")
        mock_svc = _patch_service(mock_read_record=record)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://example.com/awesome")

        assert output.startswith("# Awesome Article")

    # ------------------------------------------------------------------
    # Falls back to summary when content is None
    # ------------------------------------------------------------------
    def test_falls_back_to_summary(self):
        """If record.content is None, record.summary is used as the body."""
        import BatchAgent.tools.web_tools as wt

        record = _make_record(
            title="",
            content=None,
            summary="Summary text is used as fallback.",
        )
        mock_svc = _patch_service(mock_read_record=record)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://example.com/no-content")

        assert "Summary text is used as fallback." in output

    # ------------------------------------------------------------------
    # No content and no summary → informative error string
    # ------------------------------------------------------------------
    def test_no_content_returns_error_message(self):
        """
        If neither content nor summary is present, the function should
        return a '[URL Read Failed]' message rather than an empty string.
        """
        import BatchAgent.tools.web_tools as wt

        record = _make_record(content=None, summary="")
        mock_svc = _patch_service(mock_read_record=record)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://example.com/empty")

        assert "[URL Read Failed]" in output

    # ------------------------------------------------------------------
    # Service exception → error string
    # ------------------------------------------------------------------
    def test_service_exception_returns_error(self):
        """
        If read_url() raises, fetch_and_parse_url must catch the error
        and return a '[URL Read Error]' string with the URL in it.
        """
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service(fail_read=True)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://crash.example.com")

        assert "[URL Read Error]" in output
        assert "https://crash.example.com" in output

    def test_github_folder_returns_listing_without_reading_page(self):
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service()
        listing = (
            "# GitHub Folder Listing\n"
            "**Requested URL:** https://github.com/org/repo/tree/main/skills\n\n"
            "- README.md: https://github.com/org/repo/blob/main/skills/README.md (file)"
        )

        with patch.object(wt, "_get_service", return_value=mock_svc):
            with patch.object(wt, "_render_github_listing", return_value=listing):
                output = wt.fetch_and_parse_url("https://github.com/org/repo/tree/main/skills")

        assert output == listing
        mock_svc.read_url.assert_not_called()

    def test_github_folder_listing_supports_offset_and_name_filter(self):
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service()
        entries = [
            ("alpha", "https://github.com/org/repo/tree/main/skills/alpha", "dir"),
            ("beta", "https://github.com/org/repo/tree/main/skills/beta", "dir"),
            ("gamma", "https://github.com/org/repo/tree/main/skills/gamma", "dir"),
        ]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            with patch.object(wt, "_fetch_github_contents", return_value=(200, [])):
                with patch.object(wt, "_github_entries_from_payload", return_value=entries):
                    output = wt.fetch_and_parse_url(
                        "https://github.com/org/repo/tree/main/skills",
                        offset=1,
                        limit=1,
                        name_contains="a",
                    )

        assert "**Filter:** name contains `a`" in output
        assert "**Showing:** 2-2 of 3 entries" in output
        assert "beta" in output
        assert "alpha" not in output
        assert "offset=2" in output
        mock_svc.read_url.assert_not_called()

    def test_github_folder_listing_clamps_large_offset(self):
        import BatchAgent.tools.web_tools as wt

        mock_svc = _patch_service()
        entries = [
            ("alpha", "https://github.com/org/repo/tree/main/skills/alpha", "dir"),
            ("beta", "https://github.com/org/repo/tree/main/skills/beta", "dir"),
        ]

        with patch.object(wt, "_get_service", return_value=mock_svc):
            with patch.object(wt, "_fetch_github_contents", return_value=(200, [])):
                with patch.object(wt, "_github_entries_from_payload", return_value=entries):
                    output = wt.fetch_and_parse_url(
                        "https://github.com/org/repo/tree/main/skills",
                        offset=99,
                        limit=1,
                    )

        assert "Requested offset was beyond the available range" in output
        assert "**Showing:** 2-2 of 2 entries" in output
        assert "beta" in output
        mock_svc.read_url.assert_not_called()

    def test_github_blob_reads_raw_url(self):
        import BatchAgent.tools.web_tools as wt

        record = _make_record(
            title="README.md",
            url="https://raw.githubusercontent.com/org/repo/main/README.md",
            content="Project overview",
        )
        mock_svc = _patch_service(mock_read_record=record)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            output = wt.fetch_and_parse_url("https://github.com/org/repo/blob/main/README.md")

        assert "Project overview" in output
        mock_svc.read_url.assert_called_once_with("https://raw.githubusercontent.com/org/repo/main/README.md")

    def test_not_found_returns_parent_suggestions(self):
        import BatchAgent.tools.web_tools as wt

        missing = _make_record(
            title="",
            url="https://example.com/a/b/missing",
            summary="Client error '404 Not Found'",
            content="URL read failed: 404",
        )
        missing.metadata["error"] = True
        mock_svc = _patch_service(mock_read_record=missing)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            with patch.object(
                wt,
                "_find_available_parent_urls",
                return_value=[("b", "https://example.com/a/b", "parent")],
            ):
                output = wt.fetch_and_parse_url("https://example.com/a/b/missing")

        assert "# URL Not Found" in output
        assert "https://example.com/a/b" in output

    def test_missing_github_file_returns_sitemap(self):
        import BatchAgent.tools.web_tools as wt

        missing = _make_record(
            title="",
            url="https://raw.githubusercontent.com/org/repo/main/missing.md",
            summary="Client error '404 Not Found'",
            content="URL read failed: 404",
        )
        missing.metadata["error"] = True
        mock_svc = _patch_service(mock_read_record=missing)

        with patch.object(wt, "_get_service", return_value=mock_svc):
            with patch.object(
                wt,
                "_github_sitemap_for_missing_url",
                return_value="# GitHub Sitemap\n**Requested URL:** https://github.com/org/repo/blob/main/missing.md",
            ):
                output = wt.fetch_and_parse_url("https://github.com/org/repo/blob/main/missing.md")

        assert "# GitHub Sitemap" in output
        assert "Requested URL" in output


# =====================================================================
# TestDomainResolution
# =====================================================================

class TestDomainResolution:
    """Validates the _resolve_domain() mapping table exhaustively."""

    def test_all_known_mappings(self):
        """
        Every documented category alias must map to its expected service domain.
        This test acts as a regression guard — if the mapping table changes,
        this test catches unintended regressions.
        """
        import BatchAgent.tools.web_tools as wt

        expected = {
            "news":     "news",
            "academic": "academic",
            "research": "academic",
            "paper":    "academic",
            "papers":   "academic",
            "medical":  "medical",
            "health":   "medical",
            "medicine": "medical",
            "biology":  "medical",
            # All below should fall through to "general"
            "code":          "general",
            "programming":   "general",
            "software_eng":  "general",
            "math":          "general",
            "science":       "general",
            "finance":       "general",
            "business":      "general",
            "general":       "general",
            "cooking":       "general",   # unknown → general
        }

        for category, domain in expected.items():
            result = wt._resolve_domain(category)
            assert result == domain, f"_resolve_domain({category!r}) → {result!r}, expected {domain!r}"

    def test_case_insensitive(self):
        """Domain resolution must be case-insensitive."""
        import BatchAgent.tools.web_tools as wt

        assert wt._resolve_domain("NEWS") == "news"
        assert wt._resolve_domain("Academic") == "academic"
        assert wt._resolve_domain("MEDICAL") == "medical"


# =====================================================================
# TestFormatRecords
# =====================================================================

class TestFormatRecords:
    """Unit tests for the _format_records() helper (pure formatting logic)."""

    def test_empty_list_returns_empty(self):
        """An empty record list should produce an empty string."""
        import BatchAgent.tools.web_tools as wt
        assert wt._format_records([], "any query") == ""

    def test_basic_formatting(self):
        """Each record should render its index, title, URL, source, and summary."""
        import BatchAgent.tools.web_tools as wt

        records = [_make_record(
            title="Result One",
            url="https://example.com/1",
            source="SourceA",
            summary="Informative summary text.",
        )]
        output = wt._format_records(records, "test")

        assert "[1] Result One" in output
        assert "https://example.com/1" in output
        assert "SourceA" in output
        assert "Informative summary text." in output

    def test_published_date_for_news_article(self):
        """news_article records with published_at should show a Date line."""
        import BatchAgent.tools.web_tools as wt

        pub = datetime(2024, 3, 15, tzinfo=timezone.utc)
        records = [_make_record(record_type="news_article", published_at=pub)]
        output = wt._format_records(records, "q")

        assert "2024-03-15" in output

    def test_no_date_for_web_page(self):
        """web_page records should NOT show a Date line even if published_at is set."""
        import BatchAgent.tools.web_tools as wt

        pub = datetime(2024, 3, 15, tzinfo=timezone.utc)
        records = [_make_record(record_type="web_page", published_at=pub)]
        output = wt._format_records(records, "q")

        # Date should not appear for generic web pages
        assert "2024-03-15" not in output

    def test_multiple_records_numbered(self):
        """Multiple records should be numbered sequentially [1], [2], …"""
        import BatchAgent.tools.web_tools as wt

        records = [
            _make_record(title=f"Result {i}", url=f"https://example.com/{i}")
            for i in range(1, 4)
        ]
        # Give each a unique id so they're distinct
        for i, r in enumerate(records, 1):
            r.id = f"id_{i}"
            r.title = f"Result {i}"

        output = wt._format_records(records, "multi-test")
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

"""
web_tools.py — Adapter: bridges tool_router.py to online_search_toolkit.

职责:
    • 保持原有公共 API 不变，供 tool_router.py 调用
    • 内部委托给 online_search_toolkit.OnlineSearchService

Public API (unchanged signatures):
    perform_domain_aware_search(query, category, serper_api_key, ...) -> str
    fetch_and_parse_url(url)                                           -> str
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from BatchAgent.online_search_toolkit import (
    OnlineSearchService,
    SearchRecord,
    SearchRequest,
    create_online_search_service,
)
from BatchAgent.online_search_toolkit.config import SearchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level service singleton (lazy-initialized on first use)
# ---------------------------------------------------------------------------
_service: Optional[OnlineSearchService] = None


def _get_service(
    serper_api_key: str = "",
    tavily_api_key: str = "",
) -> OnlineSearchService:
    """Return (or create) the shared OnlineSearchService.

    API keys from environment variables take precedence; explicit arguments
    are used only when the service has not yet been initialized AND the env
    var is not set.
    """
    global _service
    if _service is None:
        cfg = SearchConfig()
        # Supplement env-var values with caller-supplied keys if missing
        if serper_api_key and not cfg.api_keys.serper_api_key:
            cfg.api_keys.serper_api_key = serper_api_key
        if tavily_api_key and not cfg.api_keys.tavily_api_key:
            cfg.api_keys.tavily_api_key = tavily_api_key
        _service = create_online_search_service(cfg)
    return _service


# ---------------------------------------------------------------------------
# Domain/category mapping
# Old web_tools used flat category strings; OnlineSearchService uses typed domains.
# ---------------------------------------------------------------------------
_CATEGORY_TO_DOMAIN: dict[str, str] = {
    "news":          "news",
    "academic":      "academic",
    "research":      "academic",
    "paper":         "academic",
    "papers":        "academic",
    "medical":       "medical",
    "health":        "medical",
    "medicine":      "medical",
    "biology":       "medical",
    # Everything else maps to "general" (Serper + Tavily + Wikimedia)
}


def _resolve_domain(category: str) -> str:
    return _CATEGORY_TO_DOMAIN.get(category.lower().strip(), "general")


# ---------------------------------------------------------------------------
# Format a list of SearchRecord objects into an agent-readable markdown string
# ---------------------------------------------------------------------------
def _format_records(items: List[SearchRecord], query: str) -> str:
    if not items:
        return ""

    lines: list[str] = [f"## Search Results: {query}\n"]
    for i, r in enumerate(items, 1):
        lines.append(f"### [{i}] {r.title}")
        if r.url:
            lines.append(f"**URL:** {r.url}")
        if r.source:
            lines.append(f"**Source:** {r.source}")
        if r.record_type in ("academic_paper", "news_article") and r.published_at:
            lines.append(f"**Date:** {r.published_at.strftime('%Y-%m-%d')}")
        if r.summary:
            lines.append(f"\n{r.summary}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def perform_domain_aware_search(
    query: str,
    category: str,
    serper_api_key: str,
    current_time: str = "",
    enable_youtube: bool = False,
    tavily_api_key: str = "",
    document_search_fn: Optional[Callable[[str, int], str]] = None,
) -> str:
    """Aggregated domain-aware web search.  Returns a formatted markdown string.

    Replaces the old monolithic implementation; delegates to OnlineSearchService
    which handles Serper/Tavily/Wikimedia/RSS/PubMed/arXiv with caching.

    Parameters
    ----------
    query           : search query string
    category        : domain hint (news, academic, medical, code, general, …)
    serper_api_key  : Google Serper key (falls back to SERPER_API_KEY env var)
    current_time    : ISO-8601 reference time for temporal query enhancement
    enable_youtube  : also search YouTube (requires Serper key)
    tavily_api_key  : Tavily fallback key (falls back to TAVILY_API_KEY env var)
    document_search_fn : optional RAG callback (query, top_k) -> str
    """
    service = _get_service(
        serper_api_key=serper_api_key,
        tavily_api_key=tavily_api_key,
    )
    domain = _resolve_domain(category)

    request = SearchRequest(
        query=query,
        domain=domain,         # type: ignore[arg-type]
        category=category or None,
        limit=8,
        enable_youtube=enable_youtube,
        current_time=current_time,
        use_cache=True,
        use_web_search=True,
        use_news_sources=(domain == "news"),
        use_academic_sources=(domain == "academic"),
        use_medical_sources=(domain == "medical"),
    )

    try:
        result = service.search(request)
        items = result.items
    except Exception as exc:
        logger.exception("web_tools: search failed: %s", exc)
        items = []

    parts: list[str] = []
    if items:
        parts.append(_format_records(items, query))
    else:
        parts.append(
            "[Web Search Unavailable] No results returned. "
            "Set SERPER_API_KEY or TAVILY_API_KEY to enable web search."
        )

    # Optional document RAG — skip placeholder messages when no document is loaded
    if document_search_fn:
        try:
            doc = document_search_fn(query, 3) or ""
            if doc and "No document is currently loaded" not in doc:
                parts.append(f"## Document Search\n\n{doc}")
        except Exception:
            pass

    return "\n\n".join(parts)


def fetch_and_parse_url(url: str) -> str:
    """Fetch a URL and return its text content as a markdown string.

    Replaces the old implementation; delegates to OnlineSearchService.read_url()
    which handles HTML, PDF, YouTube, and Playwright fallback.
    """
    service = _get_service()

    try:
        record = service.read_url(url)
        content = record.content or record.summary or ""
        title = record.title or ""

        if not content:
            return f"[URL Read Failed] No content extracted from {url}"

        header = f"# {title}\n**URL:** {record.url}\n\n" if title else ""
        return header + content

    except Exception as exc:
        logger.exception("fetch_and_parse_url failed for %s: %s", url, exc)
        return f"[URL Read Error] Failed to fetch {url}: {exc}"

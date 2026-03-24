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
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import httpx

from BatchAgent.online_search_toolkit import (
    OnlineSearchService,
    SearchRecord,
    SearchRequest,
    create_online_search_service,
)
from BatchAgent.online_search_toolkit.config import SearchConfig

logger = logging.getLogger(__name__)
_DEFAULT_LINK_WINDOW = 8
_MAX_LINK_WINDOW = 50


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


def _normalize_url(url: str) -> str:
    return (url or "").strip()


def _is_not_found_record(record: SearchRecord) -> bool:
    if not record or not record.metadata.get("error"):
        return False
    text = " ".join(
        part for part in [record.summary or "", record.content or ""] if part
    ).lower()
    return "404" in text or "not found" in text


def _record_to_markdown(record: SearchRecord, requested_url: str) -> str:
    content = record.content or record.summary or ""
    title = record.title or ""

    if not content:
        return f"[URL Read Failed] No content extracted from {requested_url}"

    header_parts: List[str] = []
    if title:
        header_parts.append(f"# {title}")
    header_parts.append(f"**URL:** {record.url or requested_url}")
    if requested_url and record.url and requested_url != record.url:
        header_parts.append(f"**Requested URL:** {requested_url}")
    header = "\n".join(header_parts)
    return f"{header}\n\n{content}" if header else content


def _format_link_list(
    heading: str,
    requested_url: str,
    entries: Sequence[Tuple[str, str, str]],
    *,
    intro: Optional[str] = None,
    offset: int = 0,
    limit: int = _DEFAULT_LINK_WINDOW,
    name_contains: str = "",
) -> str:
    filtered_entries = list(entries)
    needle = (name_contains or "").strip().lower()
    if needle:
        filtered_entries = [
            item for item in filtered_entries
            if needle in (item[0] or "").lower() or needle in (item[1] or "").lower()
        ]
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or _DEFAULT_LINK_WINDOW), _MAX_LINK_WINDOW))
    lines = [heading, f"**Requested URL:** {requested_url}"]
    if intro:
        lines.extend(["", intro])
    if needle:
        lines.extend(["", f"**Filter:** name contains `{needle}`"])
    if not filtered_entries:
        lines.extend(["", "No related URLs were found."])
        return "\n".join(lines)
    total = len(filtered_entries)
    if offset >= total:
        offset = max(0, total - limit)
        lines.extend(["", f"**Notice:** Requested offset was beyond the available range. Showing the last available window instead."])
    window = filtered_entries[offset: offset + limit]
    start_idx = offset + 1
    end_idx = min(offset + len(window), total)
    lines.extend(["", f"**Showing:** {start_idx}-{end_idx} of {total} entries"])
    lines.append("")
    for name, url, kind in window:
        label = name or url
        suffix = f" ({kind})" if kind else ""
        lines.append(f"- {label}: {url}{suffix}")
    if end_idx < total:
        lines.extend([
            "",
            f"To continue, call `read_url` again with the same URL and `offset={end_idx}`.",
        ])
    if not needle and total > limit:
        lines.append(
            "To narrow the list, call `read_url` with `name_contains` set to part of a filename or folder name."
        )
    return "\n".join(lines)


def _parse_github_url(url: str) -> Optional[Dict[str, str]]:
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    if host == "github.com":
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1]
        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            return {
                "owner": owner,
                "repo": repo,
                "branch": parts[3],
                "path": "/".join(parts[4:]),
                "kind": parts[2],
                "html_url": normalized,
            }
        return {
            "owner": owner,
            "repo": repo,
            "branch": "main",
            "path": "/".join(parts[2:]),
            "kind": "tree",
            "html_url": normalized,
        }

    if host == "raw.githubusercontent.com":
        if len(parts) < 3:
            return None
        owner, repo, branch = parts[0], parts[1], parts[2]
        path = "/".join(parts[3:])
        html_kind = "blob" if path else "tree"
        html_path = f"/{owner}/{repo}/{html_kind}/{branch}/{path}" if path else f"/{owner}/{repo}"
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "path": path,
            "kind": "raw",
            "html_url": f"https://github.com{html_path}",
        }

    return None


def _github_html_url(owner: str, repo: str, branch: str, path: str, kind: str = "tree") -> str:
    clean = path.strip("/")
    if not clean:
        return f"https://github.com/{owner}/{repo}"
    return f"https://github.com/{owner}/{repo}/{kind}/{branch}/{clean}"


def _github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path.strip('/')}"


def _github_contents_api_url(owner: str, repo: str, branch: str, path: str = "") -> str:
    clean = path.strip("/")
    if clean:
        return f"https://api.github.com/repos/{owner}/{repo}/contents/{clean}?ref={branch}"
    return f"https://api.github.com/repos/{owner}/{repo}/contents?ref={branch}"


def _github_api_headers() -> Dict[str, str]:
    return {"Accept": "application/vnd.github+json"}


def _fetch_github_contents(owner: str, repo: str, branch: str, path: str = "") -> Tuple[int, object]:
    api_url = _github_contents_api_url(owner, repo, branch, path)
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        response = client.get(api_url, headers=_github_api_headers())
    try:
        payload = response.json()
    except Exception:
        payload = None
    return response.status_code, payload


def _github_entries_from_payload(payload: object) -> List[Tuple[str, str, str]]:
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    entries: List[Tuple[str, str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        html_url = str(item.get("html_url") or "").strip()
        kind = str(item.get("type") or "").strip() or "item"
        if html_url:
            entries.append((name, html_url, kind))
    entries.sort(key=lambda item: (0 if item[2] == "dir" else 1, item[0].lower()))
    return entries


def _github_parent_paths(path: str) -> List[str]:
    clean = path.strip("/")
    if not clean:
        return [""]
    parts = clean.split("/")
    parents = ["/".join(parts[:idx]) for idx in range(len(parts) - 1, -1, -1)]
    return list(dict.fromkeys(parents + [""]))


def _github_sitemap_for_missing_url(
    url: str,
    *,
    offset: int = 0,
    limit: int = _DEFAULT_LINK_WINDOW,
    name_contains: str = "",
) -> Optional[str]:
    info = _parse_github_url(url)
    if not info:
        return None
    owner = info["owner"]
    repo = info["repo"]
    branch = info["branch"]
    path = info["path"]

    for parent_path in _github_parent_paths(path):
        status_code, payload = _fetch_github_contents(owner, repo, branch, parent_path)
        if status_code != 200:
            continue
        entries = _github_entries_from_payload(payload)
        heading = "# GitHub Sitemap"
        parent_label = _github_html_url(owner, repo, branch, parent_path, "tree")
        intro = (
            f"The requested GitHub URL was not available. "
            f"Here are nearby entries from {parent_label}."
        )
        return _format_link_list(
            heading,
            url,
            entries,
            intro=intro,
            offset=offset,
            limit=limit,
            name_contains=name_contains,
        )
    return None


def _render_github_listing(
    url: str,
    *,
    offset: int = 0,
    limit: int = _DEFAULT_LINK_WINDOW,
    name_contains: str = "",
) -> Optional[str]:
    info = _parse_github_url(url)
    if not info:
        return None
    if info["kind"] in {"blob", "raw"} and info["path"]:
        return None

    status_code, payload = _fetch_github_contents(
        info["owner"],
        info["repo"],
        info["branch"],
        info["path"],
    )
    if status_code != 200:
        return _github_sitemap_for_missing_url(url)

    entries = _github_entries_from_payload(payload)
    heading = "# GitHub Folder Listing"
    intro = "This GitHub URL points to a repository or folder. Choose one of these entries to continue reading."
    return _format_link_list(
        heading,
        url,
        entries,
        intro=intro,
        offset=offset,
        limit=limit,
        name_contains=name_contains,
    )


def _candidate_parent_urls(url: str) -> List[str]:
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        return []
    parts = [p for p in (parsed.path or "").split("/") if p]
    candidates: List[str] = []
    for idx in range(len(parts) - 1, -1, -1):
        parent_path = "/" + "/".join(parts[:idx]) if idx else ""
        candidate = urlunparse((scheme, netloc, parent_path or "/", "", "", ""))
        candidates.append(candidate)
    root = urlunparse((scheme, netloc, "/", "", "", ""))
    if root not in candidates:
        candidates.append(root)
    return list(dict.fromkeys(candidates))


def _find_available_parent_urls(service: OnlineSearchService, url: str) -> List[Tuple[str, str, str]]:
    suggestions: List[Tuple[str, str, str]] = []
    for candidate in _candidate_parent_urls(url):
        try:
            record = service.read_url(candidate, persist=False, force_refresh=True)
        except Exception:
            continue
        if record.metadata.get("error"):
            continue
        label = candidate.rstrip("/").rsplit("/", 1)[-1] or urlparse(candidate).netloc
        kind = "root" if candidate.rstrip("/") == f"{urlparse(candidate).scheme}://{urlparse(candidate).netloc}" else "parent"
        suggestions.append((label, candidate, kind))
        if len(suggestions) >= _DEFAULT_LINK_WINDOW:
            break
    return suggestions


def _handle_missing_url(
    service: OnlineSearchService,
    url: str,
    record: SearchRecord,
    *,
    offset: int = 0,
    limit: int = _DEFAULT_LINK_WINDOW,
    name_contains: str = "",
) -> str:
    github_map = _github_sitemap_for_missing_url(
        url,
        offset=offset,
        limit=limit,
        name_contains=name_contains,
    )
    if github_map:
        return github_map

    suggestions = _find_available_parent_urls(service, url)
    intro = (
        f"The requested URL was not available. "
        f"Error: {record.summary or '404 Not Found'} "
        f"Here are nearby URLs that were reachable."
    )
    return _format_link_list(
        "# URL Not Found",
        url,
        suggestions,
        intro=intro,
        offset=offset,
        limit=limit,
        name_contains=name_contains,
    )


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


def fetch_and_parse_url(
    url: str,
    offset: int = 0,
    limit: int = _DEFAULT_LINK_WINDOW,
    name_contains: str = "",
) -> str:
    """Fetch a URL and return its text content as a markdown string.

    Replaces the old implementation; delegates to OnlineSearchService.read_url()
    which handles HTML, PDF, YouTube, and Playwright fallback.
    """
    normalized_url = _normalize_url(url)
    if not normalized_url:
        return "[URL Read Error] Empty URL"
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or _DEFAULT_LINK_WINDOW), _MAX_LINK_WINDOW))
    name_contains = (name_contains or "").strip()
    service = _get_service()

    try:
        github_listing = _render_github_listing(
            normalized_url,
            offset=offset,
            limit=limit,
            name_contains=name_contains,
        )
        if github_listing:
            return github_listing

        github_info = _parse_github_url(normalized_url)
        target_url = normalized_url
        if github_info and github_info["path"] and github_info["kind"] in {"blob", "raw"}:
            target_url = _github_raw_url(
                github_info["owner"],
                github_info["repo"],
                github_info["branch"],
                github_info["path"],
            )

        record = service.read_url(target_url)
        if _is_not_found_record(record):
            return _handle_missing_url(
                service,
                normalized_url,
                record,
                offset=offset,
                limit=limit,
                name_contains=name_contains,
            )
        if record.metadata.get("error"):
            return f"[URL Read Failed] Failed to fetch {normalized_url}: {record.summary or 'unknown error'}"

        return _record_to_markdown(record, normalized_url)

    except Exception as exc:
        logger.exception("fetch_and_parse_url failed for %s: %s", normalized_url, exc)
        return f"[URL Read Error] Failed to fetch {normalized_url}: {exc}"

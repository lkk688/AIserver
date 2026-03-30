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
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

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
    embedding_base_url: str = "",
    embedding_api_key: str = "",
    embedding_model: str = "",
    reranker_base_url: str = "",
    reranker_api_key: str = "",
    reranker_model: str = "",
) -> OnlineSearchService:
    """Return (or create) the shared OnlineSearchService.

    API keys from environment variables take precedence; explicit arguments
    are used only when the service has not yet been initialized AND the env
    var is not set.
    """
    global _service
    if _service is None:
        cfg = SearchConfig()
        # Search API keys
        if serper_api_key and not cfg.api_keys.serper_api_key:
            cfg.api_keys.serper_api_key = serper_api_key
        if tavily_api_key and not cfg.api_keys.tavily_api_key:
            cfg.api_keys.tavily_api_key = tavily_api_key
        # Embedding service
        if embedding_base_url:
            cfg.embedding.enabled = True
            cfg.embedding.api_url = embedding_base_url
        if embedding_api_key:
            cfg.embedding.api_key = embedding_api_key
        if embedding_model:
            cfg.embedding.api_model = embedding_model
        # Reranker service
        if reranker_base_url and reranker_model:
            cfg.rerank.enabled = True
            cfg.rerank.provider = "api"
            cfg.rerank.api_url = reranker_base_url
            cfg.rerank.api_key = reranker_api_key or None
            cfg.rerank.api_model = reranker_model
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
    embedding_base_url: str = "",
    embedding_api_key: str = "",
    embedding_model: str = "",
    reranker_base_url: str = "",
    reranker_api_key: str = "",
    reranker_model: str = "",
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
        embedding_base_url=embedding_base_url,
        embedding_api_key=embedding_api_key,
        embedding_model=embedding_model,
        reranker_base_url=reranker_base_url,
        reranker_api_key=reranker_api_key,
        reranker_model=reranker_model,
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
            "[Web Search Unavailable] Web search is not configured on this system "
            "(no SERPER_API_KEY or TAVILY_API_KEY). "
            "Do NOT retry web_search — it will keep failing. "
            "If the required information cannot be obtained from other available tools, "
            "call finish_task explaining that web search is unavailable."
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


_DEFAULT_URL_WINDOW = 200  # lines per read_url page
_MAX_URL_WINDOW = 1000

# ---------------------------------------------------------------------------
# Image extraction from web pages
# ---------------------------------------------------------------------------

# URL fragments that indicate tracking pixels, ads, or decorative assets
_AD_URL_RE = re.compile(
    r"(pixel|beacon|track(ing|er)?|analytics|doubleclick|googlead|adservice"
    r"|googlesyndication|adnxs|outbrain|taboola|sharethrough|moatads"
    r"|spacer|blank|placeholder|spinner|loading|1x1"
    r"|logo\.|icon\.|avatar\.|badge\.)",
    re.IGNORECASE,
)

# HTML containers that typically hold main article content
_CONTENT_TAG_RE = re.compile(r"^(article|main|section|figure)$", re.IGNORECASE)
_CONTENT_CLASS_RE = re.compile(
    r"\b(article|post|content|story|body|entry|detail|prose|text|news|blog)\b",
    re.IGNORECASE,
)

# HTML containers that typically hold chrome/noise (nav, ads, sidebars)
_NOISE_TAG_RE = re.compile(r"^(header|footer|nav|aside|form)$", re.IGNORECASE)
_NOISE_CLASS_RE = re.compile(
    r"\b(header|footer|nav|sidebar|menu|widget|ad[s_-]|banner|promo"
    r"|comment|related|recommend|subscribe|newsletter|share|social|cookie)\b",
    re.IGNORECASE,
)

_MIN_EXPLICIT_DIM = 80   # pixels — images with explicit w/h < this are icons
_MAX_IMAGES = 10         # cap on images reported per page


def _score_img_element(img) -> int:  # type: ignore[return]
    """
    Walk the ancestor chain of a BeautifulSoup <img> tag and return a
    content-relevance score.

    Positive: inside article/main/figure/content-class containers
    Negative: inside nav/footer/sidebar/ad-class containers
    Zero:     neutral (no strong signal)
    """
    score = 0
    for ancestor in img.parents:
        tag = getattr(ancestor, "name", None)
        if tag is None:
            continue
        classes = " ".join(ancestor.get("class") or [])
        if _CONTENT_TAG_RE.match(tag or ""):
            score += 3
        if _CONTENT_CLASS_RE.search(classes):
            score += 2
        if tag == "figure":
            score += 4          # <figure> almost always wraps content images
        if _NOISE_TAG_RE.match(tag or ""):
            score -= 4
        if _NOISE_CLASS_RE.search(classes):
            score -= 3
    return score


def _img_description(img) -> str:
    """
    Return the best human-readable description for an <img> element.

    Priority: <figcaption> sibling → alt → title → aria-label → ''
    """
    # <figcaption> sibling inside the same <figure>
    parent = img.find_parent("figure")
    if parent:
        cap = parent.find("figcaption")
        if cap:
            text = cap.get_text(" ", strip=True)
            if text:
                return text

    for attr in ("alt", "title", "aria-label"):
        val = (img.get(attr) or "").strip()
        if val and val.lower() not in {"image", "photo", "picture", "img", "."}:
            return val

    # Enclosing <a> title
    a = img.find_parent("a")
    if a:
        val = (a.get("title") or a.get("aria-label") or "").strip()
        if val:
            return val

    return ""


def _resolve_img_src(img, base_url: str) -> str:
    """
    Return the best absolute URL for an <img>, checking lazy-load attrs too.
    """
    for attr in ("src", "data-src", "data-original", "data-lazy-src", "data-srcset"):
        val = (img.get(attr) or "").strip()
        if val and not val.startswith("data:"):
            # srcset may have "url 2x" — take just the first token
            val = val.split()[0]
            return urljoin(base_url, val)
    return ""


def _is_likely_content_image(src: str, img) -> bool:
    """Return False for images that are very likely icons, pixels, or ads."""
    if not src:
        return False

    # Explicit tiny dimensions in attributes
    for attr in ("width", "height"):
        val = img.get(attr, "")
        try:
            if int(str(val).replace("px", "").strip()) < _MIN_EXPLICIT_DIM:
                return False
        except (ValueError, TypeError):
            pass

    # Known ad/tracker URL patterns
    if _AD_URL_RE.search(src):
        return False

    # SVG inline images used as icons (tiny data URIs already excluded above)
    if src.endswith(".svg") and _AD_URL_RE.search(src):
        return False

    return True


def _extract_page_images(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    Parse *html* and return a scored, deduplicated list of content images.

    Each entry: {"src": absolute_url, "description": text, "score": int}
    Sorted by score descending; capped at _MAX_IMAGES.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")

    seen: set = set()
    candidates: List[Dict] = []

    for img in soup.find_all("img"):
        src = _resolve_img_src(img, base_url)
        if not src or src in seen:
            continue
        if not _is_likely_content_image(src, img):
            continue
        score = _score_img_element(img)
        if score < -2:          # strongly noise-classified → skip
            continue
        seen.add(src)
        candidates.append({
            "src": src,
            "description": _img_description(img),
            "score": score,
        })

    # Sort: content images first, then by order of appearance (stable sort)
    candidates.sort(key=lambda x: -x["score"])
    return candidates[:_MAX_IMAGES]


_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)\s]+)\)", re.IGNORECASE)
_MD_IMAGE_NOISE_RE = re.compile(
    r"(pixel|beacon|track|analytics|doubleclick|googlead|spacer|1x1"
    r"|/icons?/|/static/images/|/mobile/|/copyright/"
    r"|[-_](\d{1,2})px[-_.]"   # e.g. -25px, _20px — tiny sized images in URL
    r"|icon\.|logo\.|badge\.|spinner\.|wordmark\.|tagline\.)",
    re.IGNORECASE,
)

_IMAGE_EXT_RE = re.compile(
    r"\.(jpe?g|png|gif|webp|avif|tiff?|bmp|svg)(\?[^)]*)?$", re.IGNORECASE
)

def _extract_images_from_markdown(markdown: str) -> List[Dict[str, str]]:
    """
    Extract image URLs from Jina/markdownify output when HTML is unavailable.
    Returns deduplicated list of {"src", "description", "score"} dicts.

    Only includes URLs with recognised image extensions to avoid picking up
    page URLs that appear as <img src> in language-switcher or similar widgets.
    """
    seen: set = set()
    results: List[Dict[str, str]] = []
    for m in _MD_IMAGE_RE.finditer(markdown):
        alt, src = m.group(1).strip(), m.group(2).strip()
        if not src or src in seen:
            continue
        # Must look like an actual image file
        if not _IMAGE_EXT_RE.search(src.split("?")[0]):
            continue
        if _MD_IMAGE_NOISE_RE.search(src):
            continue
        seen.add(src)
        results.append({"src": src, "description": alt, "score": 1})
        if len(results) >= _MAX_IMAGES:
            break
    return results


def _fetch_html_for_images(url: str, timeout: int = 15) -> str:
    """
    Lightweight raw HTML fetch for image extraction.

    Uses urllib (stdlib) as primary transport — passes CDN bot-detection
    that blocks httpx (e.g. Wikipedia).  Returns empty string on failure.
    """
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse as _urlparse

    parsed = _urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": origin + "/",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type", "")
            if "html" in ct:
                return resp.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


def _fetch_wikipedia_images(url: str) -> List[Dict[str, str]]:
    """
    Fetch images for a Wikipedia article via the public REST API.

    Works for any ``*.wikipedia.org/wiki/<Title>`` URL.
    Returns a list of ``{"src": url, "alt": title, "description": ""}``
    dicts, ready for ``_format_images_section``.
    """
    import urllib.request
    import json
    from urllib.parse import urlparse as _urlparse, quote

    parsed = _urlparse(url)
    # Only handle Wikipedia article pages
    if "wikipedia.org" not in parsed.netloc:
        return []
    path_parts = parsed.path.split("/wiki/", 1)
    if len(path_parts) < 2 or not path_parts[1]:
        return []
    article_title = path_parts[1]
    lang = parsed.netloc.split(".")[0]  # e.g. "en"
    api_url = f"https://{lang}.wikipedia.org/api/rest_v1/page/media-list/{quote(article_title)}"

    _SKIP_EXT = {".svg", ".oga", ".ogv", ".ogg", ".webm", ".mp3", ".wav"}
    _SKIP_RE = re.compile(
        r"(icon|logo|flag|button|badge|arrow|stub|edit|portal|commons|wikidata"
        r"|sound|audio|wikimedia-logo|wikipedia-wordmark|enwiki|question_book"
        r"|red_pencil|padlock|globe|wikivoyage|wiktionary|wikisource)",
        re.IGNORECASE,
    )

    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except Exception:
        return []

    results: List[Dict[str, str]] = []
    seen: set = set()
    for item in data.get("items", []):
        title = item.get("title", "").replace("File:", "").replace("_", " ")
        if _SKIP_RE.search(title):
            continue

        # Prefer: largest srcset thumbnail → original source
        # srcset is ordered smallest→largest, so take the last valid entry.
        srcsets = item.get("srcset", [])
        src = ""
        for entry in reversed(srcsets):
            s = entry.get("src", "")
            if s:
                src = ("https:" + s) if s.startswith("//") else s
                break
        if not src:
            orig = item.get("original", {})
            s = orig.get("source", "")
            if s:
                src = ("https:" + s) if s.startswith("//") else s

        if not src or src in seen:
            continue
        # Skip non-photo types based on the FINAL url extension
        url_path = src.split("?")[0]
        ext = re.search(r"\.[a-z0-9]+$", url_path, re.IGNORECASE)
        if ext and ext.group(0).lower() in _SKIP_EXT:
            continue

        seen.add(src)
        results.append({"src": src, "alt": title, "description": ""})
        if len(results) >= 15:
            break
    return results


def _format_images_section(images: List[Dict[str, str]]) -> str:
    """
    Render the image list as a markdown section the model can act on.

    Format: numbered list where each line contains the human-readable title
    and the EXACT URL to pass to view_image, so the model cannot confuse
    the display name with the URL.
    """
    if not images:
        return ""
    lines = [
        "",
        "## Page Images",
        "Call `view_image` with the exact URL shown. Do NOT guess or modify URLs.",
        "",
    ]
    for i, img in enumerate(images, 1):
        src = img["src"]
        alt = (img.get("alt") or img.get("description") or "").strip()
        label = alt if alt else f"Image {i}"
        lines.append(f"{i}. {label}")
        lines.append(f"   URL: {src}")
    return "\n".join(lines)


def _apply_line_window(text: str, offset: int = 0, limit: int = _DEFAULT_URL_WINDOW) -> str:
    """Slice *text* to [offset, offset+limit) lines, appending pagination hints."""
    lines = text.splitlines()
    total = len(lines)
    offset = max(0, offset)
    limit = max(1, min(limit, _MAX_URL_WINDOW))
    window = lines[offset: offset + limit]
    result = "\n".join(window)
    end = offset + len(window)
    if end < total:
        result += f"\n\n[... {total - end} more lines. Call read_url with offset={end} to continue ...]"
    elif offset > 0:
        result += f"\n\n[End of document ({total} lines total)]"
    return result


def _format_url_error(url: str, error_msg: str) -> str:
    """
    Return a structured, agent-actionable error message for a failed URL fetch.

    Distinguishes permanent access failures (4xx) from transient errors so the
    agent knows whether retrying will help.
    """
    msg_lower = error_msg.lower()

    # Permanent access-denial errors — retrying is pointless
    if any(code in msg_lower for code in ("403", "401", "forbidden", "unauthorized", "paywall")):
        domain = url.split("/")[2] if url.count("/") >= 2 else url
        return (
            f"[URL Permanently Blocked] {url}\n"
            f"Error: {error_msg}\n"
            f"This URL is blocked by the server (bot protection / paywall / login required). "
            f"Do NOT retry this URL. "
            f"If web_search is available, try searching for the content title to find an alternative source. "
            f"If no alternative source is available, call finish_task explaining that "
            f"'{domain}' requires authentication or blocks automated access."
        )

    # Resource not found — also permanent
    if any(code in msg_lower for code in ("404", "not found", "410", "gone")):
        return (
            f"[URL Not Found] {url}\n"
            f"Error: {error_msg}\n"
            f"This URL does not exist. Do NOT retry. "
            f"Try a web_search to locate the content, or call finish_task if it cannot be found."
        )

    # Rate limiting — could succeed later but not in this session
    if any(code in msg_lower for code in ("429", "too many requests", "rate limit")):
        return (
            f"[URL Rate Limited] {url}\n"
            f"Error: {error_msg}\n"
            f"The server is rate-limiting requests. Do NOT retry in this session. "
            f"Try web_search to find the information from another source."
        )

    # Generic / transient error
    return (
        f"[URL Read Failed] {url}\n"
        f"Error: {error_msg}\n"
        f"The URL could not be fetched. If this was a transient network error you may retry once. "
        f"If it fails again, try web_search or call finish_task explaining the limitation."
    )


def fetch_and_parse_url(
    url: str,
    offset: int = 0,
    limit: int = _DEFAULT_URL_WINDOW,
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
    limit = max(1, min(int(limit or _DEFAULT_URL_WINDOW), _MAX_URL_WINDOW))
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
        # If the cached record is an error, force a fresh fetch — the site may
        # now be accessible via Jina / alt-headers even if it was blocked before.
        if record.metadata.get("error"):
            record = service.read_url(target_url, force_refresh=True)

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
            return _format_url_error(normalized_url, record.summary or "unknown error")

        full_text = _record_to_markdown(record, normalized_url)
        paged = _apply_line_window(full_text, offset=offset, limit=limit)

        # Append image list only on the first page (offset == 0) so it doesn't
        # repeat on every paginated read_url call.
        if offset == 0:
            images: List[Dict[str, str]] = []
            # 1. Wikipedia REST API — fast, clean, no bot-detection issues.
            wiki_images = _fetch_wikipedia_images(target_url)
            if wiki_images:
                images = wiki_images
            else:
                # 2. Raw HTML fetch (urllib passes most CDN bot checks).
                raw_html = _fetch_html_for_images(target_url)
                if raw_html:
                    images = _extract_page_images(raw_html, target_url)
                else:
                    # 3. Last resort: scan Jina/markdownify output for ![alt](url) patterns.
                    images = _extract_images_from_markdown(full_text)
            img_section = _format_images_section(images)
            if img_section:
                paged = paged + img_section

        return paged

    except Exception as exc:
        logger.exception("fetch_and_parse_url failed for %s: %s", normalized_url, exc)
        return _format_url_error(normalized_url, str(exc))

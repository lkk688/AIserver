from __future__ import annotations

import html
import io
import json
import logging
import re
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
import markdownify
from bs4 import BeautifulSoup

from ..models import SearchRecord
from ..utils import (
    build_embedding_text,
    clean_html_text,
    detect_language,
    make_article_id,
    normalize_url,
    parse_datetime,
    trim_summary,
    utc_now,
)

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "iframe", "svg", "form", "button"
]

_MAX_CONTENT_CHARS = 30000
_PLAYWRIGHT_THRESHOLD = 300

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _truncate_content(text: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n\n... [Content Truncated] ..."


def _extract_pdf_text(raw_bytes: bytes) -> str:
    if not _PYPDF_AVAILABLE:
        raise RuntimeError("pypdf is not installed")

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages: List[str] = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"--- Page {i + 1} ---\n{page_text.strip()}")

    if not pages:
        raise RuntimeError("PDF appears to have no extractable text")

    return _truncate_content("\n\n".join(pages))


def _fetch_via_playwright(url: str) -> str:
    if not _PLAYWRIGHT_AVAILABLE:
        return ""

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_DEFAULT_HEADERS["User-Agent"])
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            text = page.evaluate("() => document.body.innerText")
            browser.close()
        return _truncate_content(text or "")
    except Exception as exc:
        logger.debug("Playwright fallback failed for %s: %s", url, exc)
        return ""


def _extract_youtube_video_id(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""

        if "youtu.be" in host:
            return path.strip("/").split("/")[0]

        if "youtube.com" in host:
            if path == "/watch":
                return (parse_qs(parsed.query).get("v") or [""])[0]
            if path.startswith("/shorts/") or path.startswith("/embed/"):
                parts = [p for p in path.split("/") if p]
                if len(parts) >= 2:
                    return parts[1]
    except Exception:
        return ""
    return ""


def _extract_youtube_description(page_html: str) -> str:
    patterns = [
        r'"shortDescription":"(.*?)","isCrawlable"',
        r'"shortDescription":"(.*?)","allowRatings"',
    ]
    raw = ""
    for pattern in patterns:
        m = re.search(pattern, page_html, flags=re.DOTALL)
        if m:
            raw = m.group(1)
            break

    if not raw:
        return ""

    try:
        return json.loads(f"\"{raw}\"").strip()
    except Exception:
        return raw.replace("\\n", "\n").replace("\\\"", "\"").strip()


def _extract_youtube_title(page_html: str) -> str:
    patterns = [
        r'"title":"(.*?)","lengthSeconds"',
        r'"videoDetails":\{"videoId":"[^"]+","title":"(.*?)"',
    ]
    raw = ""
    for pattern in patterns:
        m = re.search(pattern, page_html, flags=re.DOTALL)
        if m:
            raw = m.group(1)
            break

    if not raw:
        return ""

    try:
        return json.loads(f"\"{raw}\"").strip()
    except Exception:
        return raw.replace("\\n", " ").replace("\\\"", "\"").strip()


def _fetch_youtube_content(url: str) -> Tuple[str, str]:
    headers = dict(_DEFAULT_HEADERS)
    video_id = _extract_youtube_video_id(url)
    if not video_id:
        raise RuntimeError("Could not parse YouTube video ID")

    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    title = ""
    description = ""

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        oembed = client.get(
            "https://www.youtube.com/oembed",
            params={"url": watch_url, "format": "json"},
            headers=headers,
        )
        if oembed.status_code < 400:
            obj = oembed.json()
            title = str(obj.get("title") or "").strip()

        page_resp = client.get(watch_url, headers=headers)
        page_resp.raise_for_status()
        page_html = page_resp.text

        if not title:
            title = _extract_youtube_title(page_html)
        description = _extract_youtube_description(page_html)

    content = "\n".join([
        f"YouTube URL: {watch_url}",
        f"Title: {title or '[Unavailable]'}",
        "",
        "Description:",
        description or "[Unavailable]",
    ])

    return title or "YouTube Video", _truncate_content(content)


class URLReader:
    """
    Unified URL reader for:
    - html
    - pdf
    - plain text
    - local files
    - youtube
    - crawler hook
    """

    def __init__(
        self,
        *,
        max_summary_chars: int = 240,
        crawler_read_fn: Optional[Callable[[str], Dict[str, str]]] = None,
    ) -> None:
        self.max_summary_chars = max_summary_chars
        self.crawler_read_fn = crawler_read_fn

    def _make_record(
        self,
        *,
        url: str,
        title: str,
        summary: str,
        content: str,
        record_type: str,
        source_type: str,
        source: str,
        domain: str,
        category: str = "general",
        published_at=None,
        metadata: Optional[Dict] = None,
    ) -> SearchRecord:
        normalized_url = normalize_url(url)
        clean_title = clean_html_text(title) or normalized_url
        clean_summary = trim_summary(summary or content or "", self.max_summary_chars)
        language = detect_language(clean_title, clean_summary)

        return SearchRecord(
            id=make_article_id(source, clean_title, normalized_url),
            record_type=record_type,
            source_type=source_type,
            title=clean_title,
            summary=clean_summary,
            url=normalized_url,
            content=_truncate_content(content or ""),
            source=source,
            domain=domain,
            language=language,
            category=category,
            published_at=published_at,
            fetched_at=utc_now(),
            metadata=metadata or {},
        )

    def read_url(
        self,
        url: str,
        *,
        domain: str = "general",
        category: str = "general",
        use_crawler: bool = False,
    ) -> SearchRecord:
        headers = dict(_DEFAULT_HEADERS)

        try:
            parsed = urlparse(url)

            if parsed.scheme == "file":
                local_path = Path(parsed.path or "").expanduser()
                if not local_path.exists() or not local_path.is_file():
                    raise FileNotFoundError(f"Local file not found: {local_path}")

                if local_path.suffix.lower() == ".pdf":
                    content = _extract_pdf_text(local_path.read_bytes())
                    return self._make_record(
                        url=url,
                        title=local_path.name,
                        summary=content,
                        content=content,
                        record_type="pdf",
                        source_type="url_fetch",
                        source="local_file",
                        domain=domain,
                        category=category,
                        metadata={"content_type": "application/pdf", "reader": "pypdf"},
                    )

                text = local_path.read_text(encoding="utf-8", errors="replace")
                return self._make_record(
                    url=url,
                    title=local_path.name,
                    summary=text,
                    content=text,
                    record_type="text_file",
                    source_type="url_fetch",
                    source="local_file",
                    domain=domain,
                    category=category,
                    metadata={"content_type": "text/plain", "reader": "local_text"},
                )

            if _extract_youtube_video_id(url):
                title, content = _fetch_youtube_content(url)
                return self._make_record(
                    url=url,
                    title=title,
                    summary=content,
                    content=content,
                    record_type="youtube_video",
                    source_type="youtube",
                    source="youtube",
                    domain=domain,
                    category=category,
                    metadata={"reader": "youtube_parser"},
                )

            if use_crawler and self.crawler_read_fn is not None:
                try:
                    result = self.crawler_read_fn(url)
                    if result and result.get("content"):
                        return self._make_record(
                            url=url,
                            title=result.get("title", url),
                            summary=result.get("summary", result.get("content", "")),
                            content=result.get("content", ""),
                            record_type="crawler_page",
                            source_type="crawler",
                            source=result.get("source", urlparse(url).netloc),
                            domain=domain,
                            category=category,
                            metadata={
                                "reader": "crawler",
                                "crawler_metadata": result.get("metadata", {}),
                            },
                        )
                except Exception as exc:
                    logger.warning("Crawler read failed for %s: %s", url, exc)

            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                raw_bytes = response.content

            url_path = url.split("?")[0].lower()
            is_pdf = (
                url_path.endswith(".pdf")
                or "application/pdf" in content_type
                or "pdf" in content_type
            )
            is_plaintext = any(
                url_path.endswith(ext) for ext in (".txt", ".md", ".csv", ".log", ".rst")
            )

            if is_pdf:
                content = _extract_pdf_text(raw_bytes)
                return self._make_record(
                    url=url,
                    title=Path(url_path).name or url,
                    summary=content,
                    content=content,
                    record_type="pdf",
                    source_type="url_fetch",
                    source=urlparse(url).netloc,
                    domain=domain,
                    category=category,
                    metadata={"content_type": content_type, "reader": "pypdf"},
                )

            if is_plaintext:
                text = raw_bytes.decode("utf-8", errors="replace")
                return self._make_record(
                    url=url,
                    title=Path(url_path).name or url,
                    summary=text,
                    content=text,
                    record_type="text_file",
                    source_type="url_fetch",
                    source=urlparse(url).netloc,
                    domain=domain,
                    category=category,
                    metadata={"content_type": content_type, "reader": "plaintext"},
                )

            html_content = raw_bytes.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")

            title = ""
            if soup.title and soup.title.text:
                title = clean_html_text(soup.title.text)

            for tag in _NOISE_TAGS:
                for el in soup.find_all(tag):
                    el.decompose()

            main_content = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id=re.compile(r"content|main|body|post", re.I))
                or soup.find(class_=re.compile(r"content|main|article|post|entry", re.I))
                or soup.body
            )

            if main_content:
                markdown_text = markdownify.markdownify(
                    str(main_content),
                    heading_style="ATX",
                    strip=["img"],
                )
            else:
                markdown_text = soup.get_text(separator="\n", strip=True)

            markdown_text = _truncate_content(markdown_text)

            if len(markdown_text) < _PLAYWRIGHT_THRESHOLD and _PLAYWRIGHT_AVAILABLE:
                pw_text = _fetch_via_playwright(url)
                if len(pw_text) > len(markdown_text):
                    markdown_text = pw_text

            if not markdown_text.strip():
                raise RuntimeError("Could not extract meaningful page content")

            return self._make_record(
                url=url,
                title=title or url,
                summary=markdown_text,
                content=markdown_text,
                record_type="web_page",
                source_type="url_fetch",
                source=urlparse(url).netloc,
                domain=domain,
                category=category,
                metadata={"content_type": content_type, "reader": "html_parser"},
            )

        except Exception as exc:
            content = f"URL read failed: {exc}\n{traceback.format_exc(limit=2)}"
            return self._make_record(
                url=url,
                title=url,
                summary=str(exc),
                content=content,
                record_type="web_page",
                source_type="url_fetch",
                source=urlparse(url).netloc or "unknown",
                domain=domain,
                category=category,
                metadata={"error": True},
            )
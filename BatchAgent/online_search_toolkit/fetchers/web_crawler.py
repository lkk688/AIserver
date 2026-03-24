"""
Playwright-based async web crawler.

Adapted from sjsu_crawler (https://github.com/ekant1999/Playwright-MCP-Agent/tree/main/sjsu_crawler) for general-purpose site crawling.
Requires: playwright (optional dependency).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urldefrag, urlparse, urlunparse

from ..models import SearchRecord
from ..utils import clean_html_text, make_article_id, normalize_url, trim_summary, utc_now

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import Page, async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _normalize_crawl_url(raw: str) -> str:
    """Normalize a URL for deduplication during crawling."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    url, _ = urldefrag(raw)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() if isinstance(parsed.scheme, str) else ""
    netloc = parsed.netloc.lower() if isinstance(parsed.netloc, str) else ""
    path = parsed.path if isinstance(parsed.path, str) else ""
    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path.rstrip("/") or "/",
    )
    return urlunparse(normalized)


# ---------------------------------------------------------------------------
# Page extraction (runs inside Playwright context)
# ---------------------------------------------------------------------------

async def _extract_title(page: "Page") -> str:
    return (await page.title()).strip()


async def _extract_meta_description(page: "Page") -> str:
    return await page.evaluate(
        """() => {
            const el = document.querySelector('meta[name="description"]');
            return el ? (el.getAttribute('content') || '') : '';
        }"""
    )


async def _extract_full_text(page: "Page") -> str:
    raw = await page.evaluate(
        """() => {
            const main = document.querySelector('[role="main"]')
                || document.querySelector('main')
                || document.querySelector('article')
                || document.body;
            return (main || document.body).innerText || '';
        }"""
    )
    return re.sub(r"\s+", " ", (raw or "")).strip()


async def _extract_headings(page: "Page") -> List[Dict[str, Any]]:
    return await page.evaluate(
        """() => {
            const root = document.querySelector('[role="main"]')
                || document.querySelector('main')
                || document.body;
            const out = [];
            for (const el of root.querySelectorAll('h1, h2, h3, h4')) {
                const level = parseInt(el.tagName[1], 10);
                const text = el.innerText.trim();
                if (text) out.push({level, text});
            }
            return out;
        }"""
    )


async def _extract_links(page: "Page") -> List[str]:
    return await page.evaluate(
        """() => {
            const seen = new Set();
            const out = [];
            for (const a of document.querySelectorAll('a[href]')) {
                const href = a.href;
                if (href && !seen.has(href)) {
                    seen.add(href);
                    out.push(href);
                }
            }
            return out;
        }"""
    )


async def _extract_tables(page: "Page") -> List[Dict[str, Any]]:
    return await page.evaluate(
        """() => {
            const tables = [];
            for (const table of document.querySelectorAll('table')) {
                const headers = [];
                for (const th of table.querySelectorAll('th')) {
                    headers.push(th.innerText.trim());
                }
                const rows = [];
                for (const tr of table.querySelectorAll('tbody tr, tr')) {
                    const cells = [];
                    let hasData = false;
                    for (const td of tr.querySelectorAll('td')) {
                        cells.push(td.innerText.trim());
                        hasData = true;
                    }
                    if (hasData) rows.push(cells);
                }
                tables.push({headers, rows});
            }
            return tables;
        }"""
    )


async def _safe_extract(page: "Page", fn: Callable, label: str):
    """Run an extraction function, returning None on failure."""
    try:
        return await fn(page)
    except Exception:
        logger.warning("extraction step '%s' failed for %s", label, page.url, exc_info=True)
        return None


async def _extract_page_data(
    page: "Page",
    url: str,
    parent_url: Optional[str],
    depth: int,
) -> Dict[str, Any]:
    """Extract structured data from a loaded page."""
    title = await _safe_extract(page, _extract_title, "title") or ""
    meta_desc = await _safe_extract(page, _extract_meta_description, "meta_description") or ""
    full_text = await _safe_extract(page, _extract_full_text, "full_text") or ""
    headings = await _safe_extract(page, _extract_headings, "headings") or []
    links_out = await _safe_extract(page, _extract_links, "links_out") or []
    tables = await _safe_extract(page, _extract_tables, "tables") or []

    return {
        "url": url,
        "parent_url": parent_url,
        "depth": depth,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "meta_description": meta_desc,
        "full_text": full_text,
        "headings": headings,
        "links_out": links_out,
        "tables": tables,
        "status": "ok",
        "error_msg": "",
    }


# ---------------------------------------------------------------------------
# CrawlerFetcher
# ---------------------------------------------------------------------------

class CrawlerFetcher:
    """
    Playwright-based web crawler that yields SearchRecord objects.

    Works as both:
    - A multi-page crawler (crawl → multiple pages)
    - A single-page reader (read_single_page → one page dict)
    """

    def __init__(
        self,
        *,
        max_pages: int = 50,
        max_depth: int = 3,
        polite_delay_ms: int = 500,
        timeout_ms: int = 30000,
        headless: bool = True,
        max_summary_chars: int = 240,
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.polite_delay_ms = polite_delay_ms
        self.timeout_ms = timeout_ms
        self.headless = headless
        self.max_summary_chars = max_summary_chars

    def _page_data_to_record(
        self,
        data: Dict[str, Any],
        *,
        domain: str = "general",
        category: str = "general",
        query: Optional[str] = None,
    ) -> SearchRecord:
        """Convert raw page data dict to a SearchRecord."""
        url = data.get("url", "")
        title = clean_html_text(data.get("title", "")) or url
        full_text = data.get("full_text", "")
        meta_desc = data.get("meta_description", "")
        summary = trim_summary(meta_desc or full_text, self.max_summary_chars)

        # Truncate full_text for storage
        content = full_text[:30000] if full_text else ""

        return SearchRecord(
            id=make_article_id("Crawler", title, normalize_url(url)),
            record_type="crawler_page",
            source_type="crawler",
            title=title,
            summary=summary,
            url=normalize_url(url),
            content=content,
            source=urlparse(url).netloc or "Crawler",
            domain=domain,  # type: ignore[arg-type]
            language="unknown",
            category=category,
            query=query,
            fetched_at=utc_now(),
            metadata={
                "depth": data.get("depth", 0),
                "parent_url": data.get("parent_url"),
                "headings": data.get("headings", []),
                "tables_count": len(data.get("tables", [])),
                "links_out_count": len(data.get("links_out", [])),
                "reader": "playwright_crawler",
            },
        )

    async def _crawl_async(
        self,
        start_url: str,
        *,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        allowed_domains: Optional[List[str]] = None,
        skip_patterns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Core async crawl loop using Playwright."""
        if not _PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright is not installed; crawler unavailable")
            return []

        effective_max_pages = max_pages or self.max_pages
        effective_max_depth = max_depth or self.max_depth

        start_norm = _normalize_crawl_url(start_url)

        # Determine scope by allowed_domains, or default to start URL's domain
        if allowed_domains:
            scope_domains = set(d.lower() for d in allowed_domains)
        else:
            scope_domains = {urlparse(start_norm).netloc.lower()}

        stack: List[tuple] = [(start_norm, None, 0)]
        visited: set = set()
        results: List[Dict[str, Any]] = []
        skip_patterns = skip_patterns or []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=self.headless)
            context = await browser.new_context(ignore_https_errors=True)
            page = await context.new_page()

            try:
                while stack and len(results) < effective_max_pages:
                    url, parent_url, depth = stack.pop()

                    if url in visited:
                        continue
                    visited.add(url)

                    try:
                        await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=self.timeout_ms,
                        )
                        await asyncio.sleep(self.polite_delay_ms / 1000)

                        # Check redirect stayed in scope
                        final_url = page.url
                        final_domain = urlparse(final_url).netloc.lower()
                        if final_domain not in scope_domains:
                            data = {
                                "url": url,
                                "parent_url": parent_url,
                                "depth": depth,
                                "crawled_at": datetime.now(timezone.utc).isoformat(),
                                "status": "error",
                                "error_msg": f"Redirected outside scope to {final_url}",
                                "title": "",
                                "meta_description": "",
                                "full_text": "",
                                "headings": [],
                                "links_out": [],
                                "tables": [],
                            }
                        else:
                            data = await _extract_page_data(page, url, parent_url, depth)

                    except Exception as exc:
                        logger.error("Crawl failed for %s: %s", url, exc)
                        data = {
                            "url": url,
                            "parent_url": parent_url,
                            "depth": depth,
                            "crawled_at": datetime.now(timezone.utc).isoformat(),
                            "status": "error",
                            "error_msg": str(exc),
                            "title": "",
                            "meta_description": "",
                            "full_text": "",
                            "headings": [],
                            "links_out": [],
                            "tables": [],
                        }

                    results.append(data)

                    if data.get("status") != "ok":
                        continue
                    if depth >= effective_max_depth:
                        continue

                    # Enqueue discovered links
                    for link in data.get("links_out", []):
                        norm = _normalize_crawl_url(link)
                        link_domain = urlparse(norm).netloc.lower()
                        if link_domain not in scope_domains:
                            continue
                        if norm in visited:
                            continue
                        if any(s in norm for s in skip_patterns):
                            continue
                        stack.append((norm, url, depth + 1))

            finally:
                await browser.close()

        return results

    def crawl(
        self,
        start_url: str,
        *,
        max_pages: Optional[int] = None,
        max_depth: Optional[int] = None,
        allowed_domains: Optional[List[str]] = None,
        domain: str = "general",
        category: str = "general",
        query: Optional[str] = None,
    ) -> List[SearchRecord]:
        """
        Synchronous entry point: crawl a site and return SearchRecords.

        Uses asyncio.run() under the hood. If an event loop is already
        running, it will attempt nest_asyncio.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                logger.warning(
                    "Event loop already running and nest_asyncio not installed; "
                    "run 'pip install nest_asyncio' or call crawl from a sync context"
                )
                return []

        raw_pages = asyncio.run(
            self._crawl_async(
                start_url,
                max_pages=max_pages,
                max_depth=max_depth,
                allowed_domains=allowed_domains,
            )
        )

        records: List[SearchRecord] = []
        for data in raw_pages:
            if data.get("status") != "ok":
                continue
            records.append(
                self._page_data_to_record(
                    data, domain=domain, category=category, query=query
                )
            )
        return records

    def read_single_page(self, url: str) -> Dict[str, str]:
        """
        Read a single page and return a dict suitable for crawler_read_fn.

        Returns: {"title": ..., "summary": ..., "content": ..., "source": ..., "metadata": ...}
        """
        pages = self.crawl(url, max_pages=1, max_depth=0)
        if not pages:
            return {}

        record = pages[0]
        return {
            "title": record.title,
            "summary": record.summary,
            "content": record.content or "",
            "source": record.source,
            "metadata": record.metadata,
        }

    def search_crawled(
        self,
        query: str,
        limit: int = 8,
        domain: str = "general",
    ) -> List[Dict[str, Any]]:
        """
        Placeholder for searching previously crawled content.
        In production, this would query a database of crawled pages.
        For now, returns empty — the file_store handles search over persisted records.
        """
        return []

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

import feedparser
import requests

from ..models import SearchRecord
from ..utils import (
    clean_html_text,
    detect_language,
    is_recent,
    make_article_id,
    normalize_url,
    parse_datetime,
    trim_summary,
    utc_now,
)

logger = logging.getLogger(__name__)


FEED_REGISTRY: Dict[str, Dict[str, object]] = {
    "breaking_en": {
        "language": "en",
        "category": "breaking",
        "feeds": [
            "https://feeds.npr.org/1039/rss.xml",
            "https://feeds.npr.org/1002/rss.xml",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.theguardian.com/world/rss",
            "https://apnews.com/index.rss",
        ],
    },
    "world_en": {
        "language": "en",
        "category": "world",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://www.theguardian.com/world/rss",
        ],
    },
    "technology_en": {
        "language": "en",
        "category": "technology",
        "feeds": [
            "https://techcrunch.com/feed/",
            "https://news.ycombinator.com/rss",
            "https://www.wired.com/feed/rss",
            "https://feeds.arstechnica.com/arstechnica/index",
        ],
    },
    "ai_en": {
        "language": "en",
        "category": "ai",
        "feeds": [
            "https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=OpenAI+OR+Anthropic+OR+Google+DeepMind+OR+LLM&hl=en-US&gl=US&ceid=US:en",
            "https://www.technologyreview.com/feed/",
        ],
    },
    "business_en": {
        "language": "en",
        "category": "business",
        "feeds": [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            "https://www.theguardian.com/business/rss",
        ],
    },
    "science_en": {
        "language": "en",
        "category": "science",
        "feeds": [
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
            "https://www.theguardian.com/science/rss",
        ],
    },
    "china_zh": {
        "language": "zh",
        "category": "china",
        "feeds": [
            "https://www.chinanews.com.cn/rss/scroll-news.xml",
            "https://www.chinanews.com.cn/rss/china.xml",
            "https://www.chinanews.com.cn/rss/world.xml",
            "http://www.news.cn/world/news_world.xml",
            "http://www.news.cn/tech/news_tech.xml",
        ],
    },
    "breaking_zh": {
        "language": "zh",
        "category": "breaking",
        "feeds": [
            "https://www.chinanews.com.cn/rss/scroll-news.xml",
            "http://www.news.cn/world/news_world.xml",
        ],
    },
}

TOPIC_HINTS = {
    "ai": ["ai_en", "technology_en"],
    "openai": ["ai_en"],
    "anthropic": ["ai_en"],
    "deepmind": ["ai_en"],
    "llm": ["ai_en"],
    "technology": ["technology_en"],
    "tech": ["technology_en"],
    "business": ["business_en"],
    "science": ["science_en"],
    "world": ["world_en", "china_zh"],
    "china": ["china_zh", "breaking_zh"],
    "breaking": ["breaking_en", "breaking_zh"],
}


class NewsSourceFetcher:
    def __init__(
        self,
        max_summary_chars: int = 240,
        timeout_seconds: int = 15,
        gnews_api_key: str = "",
        newsdata_api_key: str = "",
        newsapi_key: str = "",
    ):
        self.max_summary_chars = max_summary_chars
        self.timeout_seconds = timeout_seconds
        self.gnews_api_key = gnews_api_key or os.getenv("GNEWS_API_KEY", "")
        self.newsdata_api_key = newsdata_api_key or os.getenv("NEWSDATA_API_KEY", "")
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY", "")

    def _make_record(
        self,
        *,
        source: str,
        title: str,
        url: str,
        summary: str,
        language: str,
        category: str,
        published_at,
        source_type: str,
        query: str | None = None,
    ) -> SearchRecord:
        normalized_url = normalize_url(url)
        clean_title = clean_html_text(title) or normalized_url
        clean_summary = trim_summary(summary or "", self.max_summary_chars)
        lang = language or detect_language(clean_title, clean_summary)
        is_breaking = (category == "breaking" and is_recent(published_at, 12)) or is_recent(published_at, 6)

        return SearchRecord(
            id=make_article_id(source, clean_title, normalized_url),
            record_type="news_article",
            source_type=source_type,  # type: ignore[arg-type]
            title=clean_title,
            summary=clean_summary,
            url=normalized_url,
            content=None,
            source=source,
            domain="news",
            language=lang,
            category=category,
            query=query,
            published_at=published_at,
            fetched_at=utc_now(),
            is_breaking=is_breaking,
            metadata={},
        )

    # ------------------------------------------------------------------
    # RSS
    # ------------------------------------------------------------------

    def infer_feed_groups(self, query: str) -> List[str]:
        q = query.lower()
        matched: List[str] = []
        for token, groups in TOPIC_HINTS.items():
            if token in q:
                matched.extend(groups)
        if not matched:
            matched = ["breaking_en", "world_en", "technology_en", "china_zh"]

        ordered = []
        seen = set()
        for item in matched:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def fetch_feed_group(self, key: str, max_items_per_feed: int = 10) -> List[SearchRecord]:
        group = FEED_REGISTRY.get(key)
        if not group:
            return []
        return self.fetch_rss_urls(
            group["feeds"],  # type: ignore[arg-type]
            category=str(group["category"]),
            language=str(group["language"]),
            max_items_per_feed=max_items_per_feed,
        )

    def fetch_rss_urls(
        self,
        feed_urls: List[str],
        *,
        category: str,
        language: str,
        max_items_per_feed: int = 10,
    ) -> List[SearchRecord]:
        records: List[SearchRecord] = []

        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                source_name = feed.feed.get("title", urlparse(url).netloc or "Unknown RSS Source")

                for entry in feed.entries[:max_items_per_feed]:
                    raw_summary = (
                        entry.get("summary")
                        or entry.get("description")
                        or entry.get("content", [{}])[0].get("value", "")
                        or ""
                    )
                    published_at = parse_datetime(
                        entry.get("published") or entry.get("updated") or entry.get("pubDate")
                    )

                    records.append(
                        self._make_record(
                            source=source_name,
                            title=entry.get("title", "Untitled"),
                            url=entry.get("link", url),
                            summary=raw_summary,
                            language=language,
                            category=category,
                            published_at=published_at,
                            source_type="rss",
                        )
                    )
            except Exception as exc:
                logger.warning("RSS fetch failed for %s: %s", url, exc)

        dedup = {r.id: r for r in records}
        result = list(dedup.values())
        result.sort(key=lambda x: (1 if x.is_breaking else 0, x.published_at or x.fetched_at), reverse=True)
        return result

    def google_news_rss_search(self, query: str, language: str, limit: int) -> List[SearchRecord]:
        if language == "zh":
            url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        else:
            url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

        items = self.fetch_rss_urls(
            [url],
            category="general",
            language=language,
            max_items_per_feed=limit,
        )
        for item in items:
            item.query = query
        return items[:limit]

    # ------------------------------------------------------------------
    # News APIs (fallback)
    # ------------------------------------------------------------------

    def search_gnews_api(self, query: str, limit: int = 8, language: str = "en") -> List[SearchRecord]:
        """Search GNews API (requires GNEWS_API_KEY)."""
        if not self.gnews_api_key:
            return []
        try:
            params = {
                "q": query,
                "lang": "zh" if language == "zh" else "en",
                "max": limit,
                "apikey": self.gnews_api_key,
            }
            response = requests.get(
                "https://gnews.io/api/v4/search",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            records: List[SearchRecord] = []
            for item in response.json().get("articles", [])[:limit]:
                published_at = parse_datetime(item.get("publishedAt"))
                records.append(
                    self._make_record(
                        source=item.get("source", {}).get("name", "GNews"),
                        title=item.get("title", "Untitled"),
                        url=item.get("url", ""),
                        summary=item.get("description", "") or "",
                        language=language,
                        category="general",
                        published_at=published_at,
                        source_type="news_api",
                        query=query,
                    )
                )
            return records
        except Exception as exc:
            logger.warning("GNews API search failed: %s", exc)
            return []

    def search_newsdata_api(self, query: str, limit: int = 8, language: str = "en") -> List[SearchRecord]:
        """Search NewsData API (requires NEWSDATA_API_KEY)."""
        if not self.newsdata_api_key:
            return []
        try:
            params = {
                "apikey": self.newsdata_api_key,
                "q": query,
                "language": "zh,en" if language == "mixed" else language,
            }
            response = requests.get(
                "https://newsdata.io/api/1/news",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            records: List[SearchRecord] = []
            for item in response.json().get("results", [])[:limit]:
                published_at = parse_datetime(item.get("pubDate"))
                records.append(
                    self._make_record(
                        source=(item.get("source_id") or "NewsData").upper(),
                        title=item.get("title", "Untitled"),
                        url=item.get("link", ""),
                        summary=item.get("description", "") or "",
                        language=language if language != "mixed" else detect_language(
                            item.get("title", ""), item.get("description", "")
                        ),
                        category="general",
                        published_at=published_at,
                        source_type="news_api",
                        query=query,
                    )
                )
            return records
        except Exception as exc:
            logger.warning("NewsData API search failed: %s", exc)
            return []

    def search_newsapi_org(self, query: str, limit: int = 8, language: str = "en") -> List[SearchRecord]:
        """Search NewsAPI.org (requires NEWSAPI_KEY)."""
        if not self.newsapi_key:
            return []
        try:
            params = {
                "q": query,
                "pageSize": limit,
                "language": "zh" if language == "zh" else "en",
                "apiKey": self.newsapi_key,
                "sortBy": "publishedAt",
            }
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            records: List[SearchRecord] = []
            for item in response.json().get("articles", [])[:limit]:
                published_at = parse_datetime(item.get("publishedAt"))
                records.append(
                    self._make_record(
                        source=item.get("source", {}).get("name", "NewsAPI"),
                        title=item.get("title", "Untitled"),
                        url=item.get("url", ""),
                        summary=item.get("description", "") or "",
                        language=language,
                        category="general",
                        published_at=published_at,
                        source_type="news_api",
                        query=query,
                    )
                )
            return records
        except Exception as exc:
            logger.warning("NewsAPI.org search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Unified search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 8,
        language: str = "mixed",
        category: str | None = None,
        use_apis: bool = True,
    ) -> List[SearchRecord]:
        results: List[SearchRecord] = []

        # 1) RSS feeds
        for group_key in self.infer_feed_groups(query):
            group = FEED_REGISTRY.get(group_key)
            if not group:
                continue
            group_lang = str(group["language"])
            if language != "mixed" and group_lang != language:
                continue
            results.extend(self.fetch_feed_group(group_key, max_items_per_feed=max(limit, 5)))

        # 2) Google News RSS
        if language in ("en", "mixed"):
            results.extend(self.google_news_rss_search(query, "en", limit))
        if language in ("zh", "mixed"):
            results.extend(self.google_news_rss_search(query, "zh", limit))

        # 3) API fallback if RSS gave few results
        if use_apis and len(results) < limit:
            api_lang = "en" if language == "mixed" else language
            api_strategies = [
                ("GNews", self.search_gnews_api),
                ("NewsData", self.search_newsdata_api),
                ("NewsAPI", self.search_newsapi_org),
            ]
            for name, fn in api_strategies:
                try:
                    api_results = fn(query, limit, api_lang)
                    if api_results:
                        results.extend(api_results)
                        logger.info("%s API returned %d items", name, len(api_results))
                        break  # stop after first successful API
                except Exception as exc:
                    logger.warning("%s API failed: %s", name, exc)

        if category:
            results = [r for r in results if r.category == category]

        dedup = {r.id: r for r in results}
        final = list(dedup.values())
        final.sort(key=lambda x: (1 if x.is_breaking else 0, x.published_at or x.fetched_at), reverse=True)
        return final[: max(limit * 3, 20)]

    # ------------------------------------------------------------------
    # Breaking news
    # ------------------------------------------------------------------

    def get_breaking_news(self, limit: int = 12, language: str = "mixed") -> List[SearchRecord]:
        records: List[SearchRecord] = []

        if language in ("en", "mixed"):
            records.extend(self.fetch_feed_group("breaking_en", max_items_per_feed=limit))
        if language in ("zh", "mixed"):
            records.extend(self.fetch_feed_group("breaking_zh", max_items_per_feed=limit))

        records = [r for r in records if r.is_breaking]
        dedup = {r.id: r for r in records}
        final = list(dedup.values())
        final.sort(key=lambda x: (1 if x.is_breaking else 0, x.published_at or x.fetched_at), reverse=True)
        return final[:limit]
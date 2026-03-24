import os
import re
import json
import time
import html
import hashlib
import logging
import requests
import feedparser
import schedule

from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse, quote_plus
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from email.utils import parsedate_to_datetime

# ==========================================
# Configuration & Setup
# ==========================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

USER_AGENT = os.getenv(
    "NEWS_TOOL_USER_AGENT",
    "Mozilla/5.0 (compatible; AgentNewsToolkit/1.0; +https://example.com/bot)"
)

REQUEST_TIMEOUT = int(os.getenv("NEWS_REQUEST_TIMEOUT", "12"))
CACHE_DIR = Path(os.getenv("NEWS_CACHE_DIR", "./news_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ARTICLES_JSON = CACHE_DIR / "articles.json"
CACHE_DAILY_DIR = CACHE_DIR / "daily"
CACHE_DAILY_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MARKDOWN_DIR = CACHE_DIR / "markdown"
CACHE_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)

# API keys
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Cache TTLs
BREAKING_CACHE_TTL_MIN = int(os.getenv("BREAKING_CACHE_TTL_MIN", "30"))
SEARCH_CACHE_TTL_MIN = int(os.getenv("SEARCH_CACHE_TTL_MIN", "90"))
FEED_CACHE_TTL_HOURS = int(os.getenv("FEED_CACHE_TTL_HOURS", "6"))
DAILY_ARCHIVE_DAYS = int(os.getenv("DAILY_ARCHIVE_DAYS", "14"))

# ==========================================
# Feed Registry
# ==========================================
# Notes:
# - NPR provides multiple topical feeds.
# - Xinhua and China News publish RSS feed lists.
# - Keep feeds modular so future DB ingestion is easy.

NEWS_FEEDS: Dict[str, Dict[str, Any]] = {
    "breaking_en": {
        "language": "en",
        "category": "breaking",
        "feeds": [
            "https://feeds.npr.org/1039/rss.xml",
            "https://feeds.npr.org/1002/rss.xml",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.theguardian.com/world/rss",
            "https://apnews.com/index.rss"
        ]
    },
    "world_en": {
        "language": "en",
        "category": "world",
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://www.theguardian.com/world/rss",
            "https://feeds.npr.org/1001/rss.xml"
        ]
    },
    "technology_en": {
        "language": "en",
        "category": "technology",
        "feeds": [
            "https://techcrunch.com/feed/",
            "https://news.ycombinator.com/rss",
            "https://www.wired.com/feed/rss",
            "https://feeds.arstechnica.com/arstechnica/index"
        ]
    },
    "ai_en": {
        "language": "en",
        "category": "ai",
        "feeds": [
            "https://news.google.com/rss/search?q=Artificial+Intelligence&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=OpenAI+OR+Anthropic+OR+Google+DeepMind+OR+LLM&hl=en-US&gl=US&ceid=US:en",
            "https://www.technologyreview.com/feed/"
        ]
    },
    "business_en": {
        "language": "en",
        "category": "business",
        "feeds": [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            "https://www.theguardian.com/business/rss"
        ]
    },
    "science_en": {
        "language": "en",
        "category": "science",
        "feeds": [
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
            "https://www.theguardian.com/science/rss"
        ]
    },
    "china_zh": {
        "language": "zh",
        "category": "china",
        "feeds": [
            "https://www.chinanews.com.cn/rss/scroll-news.xml",
            "https://www.chinanews.com.cn/rss/china.xml",
            "https://www.chinanews.com.cn/rss/world.xml",
            "http://www.news.cn/politics/news_politics.xml",
            "http://www.news.cn/world/news_world.xml",
            "http://www.news.cn/tech/news_tech.xml",
            "http://www.news.cn/finance/news_finance.xml"
        ]
    },
    "breaking_zh": {
        "language": "zh",
        "category": "breaking",
        "feeds": [
            "https://www.chinanews.com.cn/rss/scroll-news.xml",
            "http://www.news.cn/world/news_world.xml",
            "http://www.news.cn/politics/news_politics.xml"
        ]
    },
    "tech_zh": {
        "language": "zh",
        "category": "technology",
        "feeds": [
            "http://www.news.cn/tech/news_tech.xml",
            "https://www.chinanews.com.cn/rss/finance.xml"
        ]
    }
}

# Optional domain/topic aliases for agent-friendly routing
QUERY_HINT_TO_FEED_KEYS = {
    "ai": ["ai_en", "technology_en", "tech_zh"],
    "openai": ["ai_en"],
    "anthropic": ["ai_en"],
    "deepmind": ["ai_en"],
    "llm": ["ai_en"],
    "technology": ["technology_en", "tech_zh"],
    "tech": ["technology_en", "tech_zh"],
    "startup": ["technology_en", "business_en"],
    "business": ["business_en"],
    "finance": ["business_en", "china_zh"],
    "world": ["world_en", "china_zh"],
    "china": ["china_zh", "breaking_zh"],
    "breaking": ["breaking_en", "breaking_zh"]
}

# ==========================================
# Utilities
# ==========================================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _iso_now() -> str:
    return _now_utc().isoformat()

def _safe_get(url: str, **kwargs) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers["User-Agent"] = USER_AGENT
    return requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)

def _clean_html_text(raw: str) -> str:
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _trim_summary(text: str, max_len: int = 240) -> str:
    text = _clean_html_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."

def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    # Remove fragment; keep query because some publishers use it meaningfully
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized

def _article_id(source: str, title: str, url: str) -> str:
    key = f"{source}|{title}|{_normalize_url(url)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

def _parse_published(entry: Dict[str, Any]) -> Optional[str]:
    candidates = [
        entry.get("published"),
        entry.get("updated"),
        entry.get("pubDate"),
    ]
    for c in candidates:
        if not c:
            continue
        try:
            return parsedate_to_datetime(c).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        try:
            return datetime.fromisoformat(c).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    return None

def _detect_language_from_text(title: str, summary: str, default: str = "unknown") -> str:
    text = (title or "") + " " + (summary or "")
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return default

def _is_recent_hours(iso_ts: Optional[str], hours: int) -> bool:
    if not iso_ts:
        return False
    try:
        dt = datetime.fromisoformat(iso_ts)
        return (_now_utc() - dt) <= timedelta(hours=hours)
    except Exception:
        return False

def _is_breaking_item(category: str, published_at: Optional[str]) -> bool:
    if category == "breaking" and _is_recent_hours(published_at, 12):
        return True
    return _is_recent_hours(published_at, 6)

def _json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning(f"Failed to load JSON from {path}: {e}")
        return default

def _json_save(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _dedupe_articles(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        aid = item.get("id") or _article_id(item.get("source", ""), item.get("title", ""), item.get("url", ""))
        if aid in seen:
            continue
        seen.add(aid)
        item["id"] = aid
        deduped.append(item)
    return deduped

def _sort_articles(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key_fn(x):
        published = x.get("published_at") or ""
        breaking = 1 if x.get("is_breaking") else 0
        return (breaking, published)
    return sorted(items, key=key_fn, reverse=True)

# ==========================================
# Cache Layer
# ==========================================

class FileNewsCache:
    def __init__(self, base_json_path: Path = CACHE_ARTICLES_JSON):
        self.base_json_path = base_json_path
        self.base_json_path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> List[Dict[str, Any]]:
        data = _json_load(self.base_json_path, {"articles": []})
        return data.get("articles", [])

    def save_all(self, articles: List[Dict[str, Any]]) -> None:
        payload = {
            "updated_at": _iso_now(),
            "articles": _sort_articles(_dedupe_articles(articles))
        }
        _json_save(self.base_json_path, payload)

    def upsert_articles(self, new_articles: List[Dict[str, Any]]) -> None:
        current = self.load_all()
        merged = current + new_articles
        merged = _dedupe_articles(merged)
        self.save_all(merged)

    def search(
        self,
        query: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
        only_recent_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        items = self.load_all()

        def match(item: Dict[str, Any]) -> bool:
            if language and item.get("language") != language:
                return False
            if category and item.get("category") != category:
                return False
            if only_recent_hours is not None and not _is_recent_hours(item.get("published_at"), only_recent_hours):
                return False
            if query:
                blob = " ".join([
                    item.get("title", ""),
                    item.get("summary", ""),
                    item.get("source", ""),
                    item.get("category", "")
                ]).lower()
                for token in query.lower().split():
                    if token not in blob:
                        return False
            return True

        filtered = [x for x in items if match(x)]
        return _sort_articles(filtered)[:limit]

    def archive_daily_snapshot(self) -> Path:
        today = datetime.now().strftime("%Y-%m-%d")
        out = CACHE_DAILY_DIR / f"{today}.json"
        payload = {
            "date": today,
            "archived_at": _iso_now(),
            "articles": self.load_all()
        }
        _json_save(out, payload)
        return out

    def write_markdown_digest(
        self,
        filename: Optional[str] = None,
        limit_per_category: int = 20
    ) -> Path:
        articles = self.load_all()
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for a in articles:
            key = f"{a.get('language', 'unknown')} / {a.get('category', 'general')}"
            grouped.setdefault(key, []).append(a)

        lines = [f"# News Digest", "", f"Generated: {_iso_now()}", ""]
        for group in sorted(grouped.keys()):
            lines.append(f"## {group}")
            lines.append("")
            group_items = _sort_articles(grouped[group])[:limit_per_category]
            for item in group_items:
                lines.append(f"- **{item.get('title', 'Untitled')}**")
                lines.append(f"  - Source: {item.get('source', '')}")
                lines.append(f"  - Published: {item.get('published_at', '')}")
                lines.append(f"  - URL: {item.get('url', '')}")
                lines.append(f"  - Breaking: {item.get('is_breaking', False)}")
                lines.append(f"  - Summary: {item.get('summary', '')}")
                lines.append("")
        if not filename:
            filename = f"news_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        out = CACHE_MARKDOWN_DIR / filename
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def cleanup_old_daily_archives(self, keep_days: int = DAILY_ARCHIVE_DAYS) -> None:
        cutoff = datetime.now() - timedelta(days=keep_days)
        for p in CACHE_DAILY_DIR.glob("*.json"):
            try:
                date_str = p.stem
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                if dt < cutoff:
                    p.unlink(missing_ok=True)
            except Exception:
                continue

cache = FileNewsCache()

# ==========================================
# RSS Discovery
# ==========================================

def discover_rss_feeds(website_url: str) -> List[Dict[str, str]]:
    logging.info(f"Attempting to discover RSS feeds on: {website_url}")
    found_feeds = []
    try:
        response = _safe_get(website_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("link", type=True):
            link_type = link["type"].lower()
            if "application/rss+xml" in link_type or "application/atom+xml" in link_type:
                href = link.get("href")
                if href:
                    found_feeds.append({
                        "title": link.get("title", "Unnamed Feed"),
                        "url": urljoin(website_url, href),
                        "type": link_type
                    })
    except Exception as e:
        logging.error(f"Failed to scan {website_url}: {e}")
    return found_feeds

# ==========================================
# Standardization
# ==========================================

def _standardize_article(
    *,
    source: str,
    title: str,
    url: str,
    summary: str,
    language: Optional[str],
    category: str,
    published_at: Optional[str],
    source_type: str,
    query: Optional[str] = None
) -> Dict[str, Any]:
    clean_title = _clean_html_text(title) or "Untitled"
    clean_summary = _trim_summary(summary or "")
    normalized_url = _normalize_url(url)
    detected_lang = language or _detect_language_from_text(clean_title, clean_summary, default="unknown")
    is_breaking = _is_breaking_item(category, published_at)

    return {
        "id": _article_id(source, clean_title, normalized_url),
        "source": source or "Unknown Source",
        "title": clean_title,
        "url": normalized_url,
        "summary": clean_summary,
        "language": detected_lang,
        "category": category,
        "published_at": published_at,
        "source_type": source_type,   # rss / gnews / newsdata / newsapi
        "query": query,
        "is_breaking": is_breaking,
        "fetched_at": _iso_now()
    }

# ==========================================
# RSS Fetching
# ==========================================

def fetch_rss_articles(
    feed_urls: List[str],
    *,
    category: str = "general",
    language: Optional[str] = None,
    max_items_per_feed: int = 10
) -> List[Dict[str, Any]]:
    logging.info(f"Fetching RSS articles from {len(feed_urls)} feeds...")
    collected: List[Dict[str, Any]] = []

    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get("title", urlparse(url).netloc or "Unknown RSS Source")
            entries = feed.entries[:max_items_per_feed]

            for entry in entries:
                raw_summary = (
                    entry.get("summary")
                    or entry.get("description")
                    or entry.get("content", [{}])[0].get("value", "")
                    or ""
                )
                published_at = _parse_published(entry)

                item = _standardize_article(
                    source=source_name,
                    title=entry.get("title", "Untitled"),
                    url=entry.get("link", url),
                    summary=raw_summary,
                    language=language,
                    category=category,
                    published_at=published_at,
                    source_type="rss"
                )
                collected.append(item)

        except Exception as e:
            logging.error(f"Error fetching RSS {url}: {e}")

    return _sort_articles(_dedupe_articles(collected))

def fetch_feed_group(feed_key: str, max_items_per_feed: int = 10) -> List[Dict[str, Any]]:
    config = NEWS_FEEDS.get(feed_key)
    if not config:
        return []
    return fetch_rss_articles(
        config["feeds"],
        category=config["category"],
        language=config["language"],
        max_items_per_feed=max_items_per_feed
    )

# ==========================================
# Google News RSS Search
# ==========================================

def _google_news_rss_url(query: str, language: str = "en") -> str:
    if language == "zh":
        return (
            "https://news.google.com/rss/search?"
            f"q={quote_plus(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        )
    return (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )

def _search_google_news_rss(query: str, max_items: int, language: str = "en") -> List[Dict[str, Any]]:
    url = _google_news_rss_url(query, language=language)
    category = "general"
    items = fetch_rss_articles([url], category=category, language=language, max_items_per_feed=max_items)
    for item in items:
        item["query"] = query
    return items[:max_items]

# ==========================================
# News APIs
# ==========================================

def _search_gnews_api(query: str, max_items: int, language: str = "en") -> List[Dict[str, Any]]:
    if not GNEWS_API_KEY:
        raise ValueError("Missing GNEWS_API_KEY")

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "zh" if language == "zh" else "en",
        "max": max_items,
        "apikey": GNEWS_API_KEY
    }
    response = _safe_get(url, params=params)
    response.raise_for_status()

    results = []
    for item in response.json().get("articles", [])[:max_items]:
        results.append(_standardize_article(
            source=item.get("source", {}).get("name", "GNews"),
            title=item.get("title", "Untitled"),
            url=item.get("url", ""),
            summary=item.get("description", "") or "",
            language=language,
            category="general",
            published_at=item.get("publishedAt"),
            source_type="gnews",
            query=query
        ))
    return results

def _search_newsdata_api(query: str, max_items: int, language: str = "en") -> List[Dict[str, Any]]:
    if not NEWSDATA_API_KEY:
        raise ValueError("Missing NEWSDATA_API_KEY")

    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "zh,en" if language == "mixed" else language
    }
    response = _safe_get(url, params=params)
    response.raise_for_status()

    results = []
    for item in response.json().get("results", [])[:max_items]:
        published_at = item.get("pubDate")
        results.append(_standardize_article(
            source=(item.get("source_id") or "NewsData").upper(),
            title=item.get("title", "Untitled"),
            url=item.get("link", ""),
            summary=item.get("description", "") or "",
            language=language if language != "mixed" else None,
            category="general",
            published_at=published_at,
            source_type="newsdata",
            query=query
        ))
    return results

def _search_newsapi_org(query: str, max_items: int, language: str = "en") -> List[Dict[str, Any]]:
    if not NEWSAPI_KEY:
        raise ValueError("Missing NEWSAPI_KEY")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": max_items,
        "language": "zh" if language == "zh" else "en",
        "apiKey": NEWSAPI_KEY,
        "sortBy": "publishedAt"
    }
    response = _safe_get(url, params=params)
    response.raise_for_status()

    results = []
    for item in response.json().get("articles", [])[:max_items]:
        results.append(_standardize_article(
            source=item.get("source", {}).get("name", "NewsAPI"),
            title=item.get("title", "Untitled"),
            url=item.get("url", ""),
            summary=item.get("description", "") or "",
            language=language,
            category="general",
            published_at=item.get("publishedAt"),
            source_type="newsapi",
            query=query
        ))
    return results

# ==========================================
# Search Router
# ==========================================

def _infer_feed_keys_from_query(query: str) -> List[str]:
    q = query.lower()
    matched = []
    for hint, feed_keys in QUERY_HINT_TO_FEED_KEYS.items():
        if hint in q:
            matched.extend(feed_keys)
    if not matched:
        matched = ["breaking_en", "world_en", "technology_en", "china_zh"]
    return list(dict.fromkeys(matched))

def _search_cache_first(
    query: str,
    max_items: int,
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    # Try strict recent search first
    items = cache.search(
        query=query,
        language=language,
        limit=max_items,
        only_recent_hours=SEARCH_CACHE_TTL_MIN // 60 + 24
    )
    return items[:max_items]

def _search_relevant_rss_first(
    query: str,
    max_items: int,
    language: str = "mixed"
) -> List[Dict[str, Any]]:
    feed_keys = _infer_feed_keys_from_query(query)
    collected = []

    # 1) Relevant registered feeds
    for feed_key in feed_keys:
        config = NEWS_FEEDS.get(feed_key)
        if not config:
            continue
        if language != "mixed" and config["language"] != language:
            continue
        collected.extend(fetch_feed_group(feed_key, max_items_per_feed=max(3, max_items // 2)))

    # 2) Google News RSS search in both languages when mixed
    if language in ("en", "mixed"):
        collected.extend(_search_google_news_rss(query, max_items=max_items, language="en"))
    if language in ("zh", "mixed"):
        collected.extend(_search_google_news_rss(query, max_items=max_items, language="zh"))

    collected = _sort_articles(_dedupe_articles(collected))

    # cheap relevance ranking
    q_tokens = query.lower().split()

    def score(item: Dict[str, Any]) -> Tuple[int, int, str]:
        blob = " ".join([
            item.get("title", ""),
            item.get("summary", ""),
            item.get("source", ""),
            item.get("category", "")
        ]).lower()
        token_hits = sum(1 for t in q_tokens if t in blob)
        breaking_bonus = 1 if item.get("is_breaking") else 0
        published = item.get("published_at") or ""
        return (token_hits, breaking_bonus, published)

    collected.sort(key=score, reverse=True)
    return collected[:max_items]

def search_news(
    query: str,
    max_items: int = 8,
    language: str = "mixed",
    use_cache: bool = True,
    use_rss: bool = True,
    use_apis: bool = True,
    write_back_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Main tool for AI agents.
    Search order:
    1) Local cache
    2) Relevant RSS feeds + Google News RSS
    3) Paid/free news APIs
    """
    logging.info(f"Search news: query='{query}' language={language} max_items={max_items}")

    # 1) Cache first
    if use_cache:
        cached = _search_cache_first(query, max_items=max_items, language=None if language == "mixed" else language)
        if cached:
            logging.info(f"Cache hit: {len(cached)} items")
            return cached[:max_items]

    # 2) RSS first
    rss_results = []
    if use_rss:
        try:
            rss_results = _search_relevant_rss_first(query, max_items=max_items, language=language)
            if rss_results:
                logging.info(f"RSS search got {len(rss_results)} items")
                if write_back_cache:
                    cache.upsert_articles(rss_results)
                return rss_results[:max_items]
        except Exception as e:
            logging.error(f"RSS search failed: {e}")

    # 3) API fallback
    if use_apis:
        strategies = [
            ("GNews API", _search_gnews_api),
            ("NewsData API", _search_newsdata_api),
            ("NewsAPI.org", _search_newsapi_org),
        ]
        api_results = []
        for source_name, strategy_func in strategies:
            try:
                logging.info(f"Trying API source: {source_name}")
                api_results = strategy_func(query, max_items, "en" if language == "mixed" else language)
                if api_results:
                    logging.info(f"{source_name} success with {len(api_results)} items")
                    if write_back_cache:
                        cache.upsert_articles(api_results)
                    return api_results[:max_items]
            except Exception as e:
                logging.warning(f"{source_name} failed: {e}")

    return []

# ==========================================
# Breaking News
# ==========================================

def get_breaking_news(
    *,
    max_items: int = 12,
    language: str = "mixed",
    use_cache: bool = True,
    write_back_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Returns recent high-priority items.
    """
    if use_cache:
        cached = cache.search(
            category="breaking",
            language=None if language == "mixed" else language,
            limit=max_items,
            only_recent_hours=24
        )
        if cached:
            return cached[:max_items]

    feed_keys = []
    if language in ("en", "mixed"):
        feed_keys.append("breaking_en")
    if language in ("zh", "mixed"):
        feed_keys.append("breaking_zh")

    collected = []
    for key in feed_keys:
        collected.extend(fetch_feed_group(key, max_items_per_feed=max_items))

    collected = [x for x in collected if x.get("is_breaking")]
    collected = _sort_articles(_dedupe_articles(collected))[:max_items]

    if collected and write_back_cache:
        cache.upsert_articles(collected)

    return collected

# ==========================================
# Daily Full Refresh / Scheduled Jobs
# ==========================================

def refresh_daily_cache(
    *,
    max_items_per_feed: int = 20,
    also_query_topics: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """
    Full daily aggregation job.
    - Pulls all configured RSS feeds
    - Optionally enriches with topic searches
    - Writes JSON cache + daily archive + markdown digest
    """
    logging.info("=== Starting daily cache refresh ===")

    all_articles = []

    # 1) Pull all feed groups
    for feed_key, config in NEWS_FEEDS.items():
        try:
            items = fetch_rss_articles(
                config["feeds"],
                category=config["category"],
                language=config["language"],
                max_items_per_feed=max_items_per_feed
            )
            all_articles.extend(items)
            logging.info(f"[{feed_key}] fetched {len(items)} items")
        except Exception as e:
            logging.error(f"Failed feed group {feed_key}: {e}")

    # 2) Optional API/RSS topic enrichment
    default_topics = [
        ("OpenAI", "en"),
        ("Anthropic", "en"),
        ("Google DeepMind", "en"),
        ("人工智能", "zh"),
        ("半导体", "zh"),
        ("机器人", "zh")
    ]
    topic_list = also_query_topics or default_topics

    for query, lang in topic_list:
        try:
            items = search_news(
                query=query,
                max_items=10,
                language=lang,
                use_cache=False,
                use_rss=True,
                use_apis=True,
                write_back_cache=False
            )
            all_articles.extend(items)
        except Exception as e:
            logging.warning(f"Topic enrichment failed for {query}: {e}")

    all_articles = _sort_articles(_dedupe_articles(all_articles))
    cache.save_all(all_articles)

    daily_json = cache.archive_daily_snapshot()
    digest_md = cache.write_markdown_digest()
    cache.cleanup_old_daily_archives()

    result = {
        "article_count": len(all_articles),
        "daily_json": str(daily_json),
        "digest_md": str(digest_md),
        "updated_at": _iso_now()
    }
    logging.info(f"=== Daily refresh complete: {result} ===")
    return result

def breaking_news_job():
    logging.info("=== Starting breaking news job ===")
    items = get_breaking_news(max_items=20, language="mixed", use_cache=False, write_back_cache=True)
    logging.info(f"Breaking news job collected {len(items)} items")

def daily_refresh_job():
    logging.info("=== Starting daily refresh job ===")
    result = refresh_daily_cache(max_items_per_feed=20)
    logging.info(f"Daily refresh result: {result}")

# ==========================================
# Agent-Friendly Helpers
# ==========================================

def summarize_for_agent(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lightweight schema specifically for agent context.
    """
    out = []
    for x in items:
        out.append({
            "title": x.get("title"),
            "summary": x.get("summary"),
            "url": x.get("url"),
            "source": x.get("source"),
            "language": x.get("language"),
            "category": x.get("category"),
            "published_at": x.get("published_at"),
            "is_breaking": x.get("is_breaking", False)
        })
    return out

def search_news_for_agent(query: str, max_items: int = 6, language: str = "mixed") -> Dict[str, Any]:
    items = search_news(query=query, max_items=max_items, language=language)
    return {
        "query": query,
        "count": len(items),
        "items": summarize_for_agent(items)
    }

def get_breaking_news_for_agent(max_items: int = 10, language: str = "mixed") -> Dict[str, Any]:
    items = get_breaking_news(max_items=max_items, language=language)
    return {
        "count": len(items),
        "items": summarize_for_agent(items)
    }

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Agent News Toolkit (Enhanced)")
    print("=" * 60 + "\n")

    # 1) Discovery test
    print("Testing RSS discovery...")
    feeds = discover_rss_feeds("https://www.theverge.com")
    for f in feeds[:5]:
        print(f"  -> {f['title']} | {f['url']}")

    print("\n" + "-" * 60)

    # 2) Breaking news test
    print("\nTesting breaking news...")
    breaking = get_breaking_news(max_items=5, language="mixed", use_cache=False)
    for i, art in enumerate(breaking, 1):
        print(f"[{i}] [{art['language']}] {art['source']} | {art['title']}")
        print(f"    {art['url']}")
        print(f"    {art['summary']}")

    print("\n" + "-" * 60)

    # 3) Topic search test
    print("\nTesting unified search...")
    articles = search_news("OpenAI agents", max_items=5, language="mixed")
    for i, art in enumerate(articles, 1):
        print(f"[{i}] [{art['language']}] {art['source']} | {art['title']}")
        print(f"    {art['url']}")
        print(f"    {art['summary']}")

    print("\n" + "-" * 60)

    # 4) Daily refresh now
    print("\nRunning daily refresh once...")
    result = refresh_daily_cache(max_items_per_feed=12)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n" + "-" * 60)

    # 5) Scheduler
    print("\nStarting scheduler...")
    # Breaking news every 30 min
    schedule.every(30).minutes.do(breaking_news_job)
    # Full daily sync twice a day
    schedule.every().day.at("07:00").do(daily_refresh_job)
    schedule.every().day.at("19:00").do(daily_refresh_job)

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from .config import SearchConfig
from .fetchers.academic_sources import AcademicSourceFetcher
from .fetchers.news_sources import NewsSourceFetcher
from .fetchers.search_apis import SearchAPIFetcher
from .models import SearchRecord
from .service import OnlineSearchService

logger = logging.getLogger(__name__)


def build_news_search_fn(config: SearchConfig):
    fetcher = NewsSourceFetcher(
        max_summary_chars=config.max_summary_chars,
        timeout_seconds=config.request_timeout_seconds,
        gnews_api_key=config.api_keys.gnews_api_key,
        newsdata_api_key=config.api_keys.newsdata_api_key,
        newsapi_key=config.api_keys.newsapi_key,
    )

    def _search_news(
        query: str,
        limit: int = 8,
        language: str = "mixed",
        category: str | None = None,
        **kwargs,
    ) -> List[SearchRecord]:
        return fetcher.search(
            query=query,
            limit=limit,
            language=language,
            category=category,
        )

    return _search_news


def build_web_search_fn(config: SearchConfig):
    fetcher = SearchAPIFetcher(
        max_summary_chars=config.max_summary_chars,
        serper_api_key=config.api_keys.serper_api_key,
        tavily_api_key=config.api_keys.tavily_api_key,
    )

    def _search_web(
        query: str,
        domain: str = "general",
        limit: int = 8,
        language: str = "mixed",
        category: str | None = None,
        enable_youtube: bool = False,
        **kwargs,
    ) -> List[SearchRecord]:
        return fetcher.search(
            query=query,
            domain=domain,
            limit=limit,
            use_serper=True,
            use_tavily=True,
            use_wikimedia=True,
            enable_youtube=enable_youtube,
        )

    return _search_web


def build_academic_search_fn(
    config: SearchConfig,
    crawler_search_fn: Optional[Callable[..., List[Dict]]] = None,
):
    fetcher = AcademicSourceFetcher(
        max_summary_chars=config.max_summary_chars,
        timeout_seconds=config.request_timeout_seconds,
        crawler_search_fn=crawler_search_fn,
    )

    def _search_academic(
        query: str,
        limit: int = 8,
        language: str = "en",
        category: str | None = None,
        use_pubmed: bool = True,
        use_medlineplus: bool = True,
        use_nimh: bool = False,
        use_crawler: bool = False,
        use_semantic_scholar: bool = True,
        use_arxiv: bool = True,
        **kwargs,
    ) -> List[SearchRecord]:
        return fetcher.search(
            query=query,
            limit=limit,
            language=language,
            category=category,
            use_pubmed=use_pubmed,
            use_medlineplus=use_medlineplus,
            use_nimh=use_nimh,
            use_crawler=use_crawler,
            use_semantic_scholar=use_semantic_scholar,
            use_arxiv=use_arxiv,
        )

    return _search_academic


def build_crawler_read_fn(
    crawler_read_impl: Optional[Callable[[str], Dict[str, str]]] = None,
    config: Optional[SearchConfig] = None,
):
    """
    Build a crawler_read_fn. If no explicit implementation is provided,
    attempt to create one from CrawlerFetcher (requires playwright).
    """
    if crawler_read_impl is not None:
        def _read(url: str) -> Dict[str, str]:
            return crawler_read_impl(url)
        return _read

    # Try to build from CrawlerFetcher
    if config is not None:
        try:
            from .fetchers.web_crawler import CrawlerFetcher
            fetcher = CrawlerFetcher(
                max_pages=1,
                max_depth=0,
                headless=config.crawler.headless,
                polite_delay_ms=config.crawler.polite_delay_ms,
                timeout_ms=config.crawler.timeout_ms,
                max_summary_chars=config.max_summary_chars,
            )

            def _crawler_read(url: str) -> Dict[str, str]:
                return fetcher.read_single_page(url)

            logger.info("Playwright CrawlerFetcher initialized for crawler_read_fn")
            return _crawler_read
        except ImportError:
            logger.debug("Playwright not available; crawler_read_fn disabled")
        except Exception as exc:
            logger.warning("Failed to initialize CrawlerFetcher: %s", exc)

    return None


def build_crawler_search_fn(
    crawler_search_impl: Optional[Callable[..., List[Dict]]] = None,
):
    if crawler_search_impl is not None:
        def _search(**kwargs) -> List[Dict]:
            return crawler_search_impl(**kwargs)
        return _search
    return None


def create_online_search_service(
    config: Optional[SearchConfig] = None,
    *,
    crawler_read_impl: Optional[Callable[[str], Dict[str, str]]] = None,
    crawler_search_impl: Optional[Callable[..., List[Dict]]] = None,
) -> OnlineSearchService:
    cfg = config or SearchConfig()

    news_search_fn = build_news_search_fn(cfg)
    web_search_fn = build_web_search_fn(cfg)
    academic_search_fn = build_academic_search_fn(cfg, crawler_search_fn=crawler_search_impl)
    crawler_read_fn = build_crawler_read_fn(crawler_read_impl, config=cfg)

    service = OnlineSearchService(
        config=cfg,
        news_search_fn=news_search_fn,
        web_search_fn=web_search_fn,
        academic_search_fn=academic_search_fn,
        crawler_read_fn=crawler_read_fn,
    )
    service.initialize()
    return service
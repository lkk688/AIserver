"""
online_search_toolkit - Unified search toolkit for AI agents.

Provides web search, news search, academic search, web crawling, and
URL reading through a single modular service.
"""
from .bootstrap import create_online_search_service
from .config import SearchConfig, CrawlerConfig, APIKeysConfig
from .models import SearchRecord, SearchRequest, SearchResult, SearchPlan, URLReadResult
from .service import OnlineSearchService

__all__ = [
    "create_online_search_service",
    "SearchConfig",
    "CrawlerConfig",
    "APIKeysConfig",
    "SearchRecord",
    "SearchRequest",
    "SearchResult",
    "SearchPlan",
    "URLReadResult",
    "OnlineSearchService",
]
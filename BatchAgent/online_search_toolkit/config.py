from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class EmbeddingConfig(BaseModel):
    enabled: bool = Field(default=os.getenv("EMBEDDING_ENABLED", "true").lower() == "true")
    api_url: str = Field(default=os.getenv("EMBEDDING_API_URL", "http://localhost:8081/v1"))
    api_key: str = Field(default=os.getenv("EMBEDDING_API_KEY", "EMPTY"))
    api_model: str = Field(default=os.getenv("EMBEDDING_API_MODEL", "text-embeddings-inference"))
    batch_size: int = Field(default=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    timeout_seconds: int = Field(default=int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")))


class RerankConfig(BaseModel):
    enabled: bool = Field(default=os.getenv("RERANK_ENABLED", "false").lower() == "true")
    provider: str = Field(default=os.getenv("RERANK_PROVIDER", "none"))
    api_url: Optional[str] = Field(default=os.getenv("RERANK_API_URL"))
    api_key: Optional[str] = Field(default=os.getenv("RERANK_API_KEY"))
    api_model: Optional[str] = Field(default=os.getenv("RERANK_API_MODEL"))


class StoreConfig(BaseModel):
    file_cache_dir: Path = Field(default=Path(os.getenv("SEARCH_FILE_CACHE_DIR", "./search_cache")))
    postgres_enabled: bool = Field(default=os.getenv("SEARCH_POSTGRES_ENABLED", "false").lower() == "true")
    postgres_dsn: str = Field(default=os.getenv("SEARCH_POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/searchdb"))
    pgvector_enabled: bool = Field(default=os.getenv("SEARCH_PGVECTOR_ENABLED", "true").lower() == "true")
    embedding_dim: int = Field(default=int(os.getenv("SEARCH_EMBEDDING_DIM", "1024")))


class SchedulerConfig(BaseModel):
    timezone: str = Field(default=os.getenv("SEARCH_SCHEDULER_TIMEZONE", "America/Los_Angeles"))
    breaking_interval_minutes: int = Field(default=int(os.getenv("SEARCH_BREAKING_INTERVAL_MINUTES", "30")))
    daily_refresh_hour_1: int = Field(default=int(os.getenv("SEARCH_DAILY_REFRESH_HOUR_1", "7")))
    daily_refresh_hour_2: int = Field(default=int(os.getenv("SEARCH_DAILY_REFRESH_HOUR_2", "19")))


class CrawlerConfig(BaseModel):
    """Configuration for the Playwright-based web crawler."""
    headless: bool = Field(default=True)
    max_pages: int = Field(default=int(os.getenv("CRAWLER_MAX_PAGES", "50")))
    max_depth: int = Field(default=int(os.getenv("CRAWLER_MAX_DEPTH", "3")))
    polite_delay_ms: int = Field(default=int(os.getenv("CRAWLER_POLITE_DELAY_MS", "500")))
    timeout_ms: int = Field(default=int(os.getenv("CRAWLER_TIMEOUT_MS", "30000")))


class APIKeysConfig(BaseModel):
    """Centralized API keys for search providers."""
    serper_api_key: str = Field(default=os.getenv("SERPER_API_KEY", ""))
    tavily_api_key: str = Field(default=os.getenv("TAVILY_API_KEY", ""))
    gnews_api_key: str = Field(default=os.getenv("GNEWS_API_KEY", ""))
    newsdata_api_key: str = Field(default=os.getenv("NEWSDATA_API_KEY", ""))
    newsapi_key: str = Field(default=os.getenv("NEWSAPI_KEY", ""))


class SearchConfig(BaseModel):
    user_agent: str = Field(default=os.getenv("SEARCH_USER_AGENT", "OnlineSearchToolkit/1.0 (+https://example.com)"))
    request_timeout_seconds: int = Field(default=int(os.getenv("SEARCH_REQUEST_TIMEOUT_SECONDS", "15")))
    breaking_cache_hours: int = Field(default=int(os.getenv("SEARCH_BREAKING_CACHE_HOURS", "24")))
    general_cache_hours: int = Field(default=int(os.getenv("SEARCH_GENERAL_CACHE_HOURS", "48")))
    daily_archive_days: int = Field(default=int(os.getenv("SEARCH_DAILY_ARCHIVE_DAYS", "14")))
    max_summary_chars: int = Field(default=int(os.getenv("SEARCH_MAX_SUMMARY_CHARS", "240")))
    default_search_limit: int = Field(default=int(os.getenv("SEARCH_DEFAULT_SEARCH_LIMIT", "8")))

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
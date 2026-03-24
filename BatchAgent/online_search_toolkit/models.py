from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RecordType = Literal[
    "news_article",
    "web_page",
    "pdf",
    "youtube_video",
    "wiki_entry",
    "academic_paper",
    "medical_article",
    "text_file",
    "crawler_page",
]

SourceType = Literal[
    "rss",
    "search_api",
    "url_fetch",
    "youtube",
    "wikimedia",
    "news_api",
    "pubmed",
    "medlineplus",
    "nimh",
    "crawler",
]

DomainType = Literal[
    "general",
    "news",
    "academic",
    "medical",
    "research",
    "programming",
    "finance",
]


class SearchRecord(BaseModel):
    id: str
    record_type: RecordType
    source_type: SourceType

    title: str
    summary: str = ""
    url: str
    content: Optional[str] = None

    source: str = ""
    domain: DomainType = "general"
    language: str = "unknown"
    category: str = "general"

    query: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime

    is_breaking: bool = False

    embedding_model: Optional[str] = None
    embedding: Optional[List[float]] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    domain: DomainType = "general"
    limit: int = 8
    language: str = "mixed"
    category: Optional[str] = None
    use_cache: bool = True
    use_web_search: bool = True
    use_news_sources: bool = True
    use_academic_sources: bool = True
    use_medical_sources: bool = True
    enable_youtube: bool = False
    current_time: str = ""


class SearchResult(BaseModel):
    query: str
    domain: DomainType = "general"
    count: int
    items: List[SearchRecord]


class SearchPlan(BaseModel):
    normalized_query: str
    domain: DomainType = "general"

    use_serper: bool = True
    use_tavily: bool = True
    use_wikimedia: bool = True
    use_youtube: bool = False

    use_news_rss: bool = False
    use_news_api: bool = False

    use_pubmed: bool = False
    use_medlineplus: bool = False
    use_nimh: bool = False

    use_crawler: bool = False

    search_filter: str = ""
    candidate_feed_groups: List[str] = Field(default_factory=list)
    recent_hours: Optional[int] = 48
    category: Optional[str] = None


class URLReadResult(BaseModel):
    record: SearchRecord
    from_cache: bool = False


class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 10
    max_depth: int = 2
    allowed_domains: List[str] = Field(default_factory=list)
    query: Optional[str] = None
    domain: DomainType = "general"


class AcademicSearchSeed(BaseModel):
    query: str
    source: Literal["pubmed", "medlineplus", "nimh", "mixed"] = "mixed"
    language: str = "en"


class RefreshSummary(BaseModel):
    domain: DomainType
    record_count: int
    source_count: int
    query_count: int = 0
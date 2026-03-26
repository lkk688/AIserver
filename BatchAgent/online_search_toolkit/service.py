from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from .config import SearchConfig
from .embedding_client import EmbeddingClient
from .fetchers.url_reader import URLReader
from .models import SearchPlan, SearchRecord, SearchRequest, SearchResult
from .reranker import build_reranker
from .search.hybrid import HybridSearcher
from .store.file_store import FileSearchStore

try:
    from .store.pg_store import PostgresSearchStore
except Exception:  # optional import during migration
    PostgresSearchStore = None  # type: ignore

from .utils import dedupe_preserve_order

logger = logging.getLogger(__name__)


class OnlineSearchService:
    """
    Unified online search service:
    - news
    - general web
    - academic / medical
    - URL reading
    - crawler-backed extraction
    - file cache + optional pgvector store
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        *,
        news_search_fn: Optional[Callable[..., List[SearchRecord]]] = None,
        web_search_fn: Optional[Callable[..., List[SearchRecord]]] = None,
        academic_search_fn: Optional[Callable[..., List[SearchRecord]]] = None,
        crawler_read_fn: Optional[Callable[[str], Dict[str, str]]] = None,
    ):
        self.config = config or SearchConfig()

        self.file_store = FileSearchStore(self.config)
        self.pg_store = (
            PostgresSearchStore(self.config)
            if self.config.store.postgres_enabled and PostgresSearchStore is not None
            else None
        )

        self.embedding_client = (
            EmbeddingClient(self.config.embedding)
            if self.config.embedding.enabled
            else None
        )

        self.reranker = build_reranker(self.config.rerank)

        self.searcher = HybridSearcher(
            file_store=self.file_store,
            pg_store=self.pg_store,
            embedding_client=self.embedding_client,
            reranker=self.reranker,
        )

        self.url_reader = URLReader(
            max_summary_chars=self.config.max_summary_chars,
            crawler_read_fn=crawler_read_fn,
        )

        self.news_search_fn = news_search_fn
        self.web_search_fn = web_search_fn
        self.academic_search_fn = academic_search_fn

    def initialize(self) -> None:
        self.file_store.initialize()
        if self.pg_store:
            self.pg_store.initialize()

    def _persist_records(self, records: List[SearchRecord]) -> None:
        if not records:
            return

        deduped = dedupe_preserve_order(records, key_fn=lambda x: x.id)
        enriched = self.searcher.enrich_embeddings(deduped)

        self.file_store.upsert_records(enriched)
        if self.pg_store:
            self.pg_store.upsert_records(enriched)

    def build_search_plan(self, request: SearchRequest) -> SearchPlan:
        plan = SearchPlan(
            normalized_query=request.query.strip(),
            domain=request.domain,
            category=request.category,
            use_youtube=request.enable_youtube,
        )

        if request.domain == "news":
            plan.use_news_rss = True
            plan.use_news_api = True
            plan.use_wikimedia = False
            plan.recent_hours = self.config.general_cache_hours

        elif request.domain in ("academic", "medical", "medical_academic", "research"):
            plan.use_pubmed = request.use_academic_sources or request.use_medical_sources
            plan.use_medlineplus = request.use_medical_sources
            plan.use_nimh = False  # consistently 403; disabled
            plan.use_crawler = True
            plan.use_news_rss = False
            plan.use_news_api = False
            plan.use_wikimedia = True
            plan.recent_hours = self.config.general_cache_hours
            if request.domain == "medical":
                # Health info: PubMed + MedlinePlus consumer pages + Europe PMC
                plan.use_arxiv = False
                plan.use_semantic_scholar = False
                plan.use_cdc = False  # JS-rendered, no usable API
                plan.use_who = False  # iris.who.int requires auth (403)
                plan.use_europe_pmc = True
            elif request.domain == "medical_academic":
                # Research papers only: PubMed + Europe PMC; no consumer health pages
                plan.use_pubmed = True
                plan.use_medlineplus = False
                plan.use_arxiv = False
                plan.use_semantic_scholar = False
                plan.use_cdc = False
                plan.use_who = False
                plan.use_europe_pmc = True
            else:
                # academic / research: arXiv + Semantic Scholar
                plan.use_arxiv = True
                plan.use_semantic_scholar = True

        else:
            plan.use_serper = True
            plan.use_tavily = True
            plan.use_wikimedia = True
            plan.use_crawler = True
            plan.recent_hours = self.config.general_cache_hours

        return plan

    def _search_cache(
        self,
        *,
        query: str,
        domain: str,
        limit: int,
        language: str,
        category: Optional[str],
    ) -> List[SearchRecord]:
        return self.searcher.search(
            query=query,
            limit=limit,
            language=language,
            domain=domain,
            category=category,
            recent_hours=self.config.general_cache_hours,
        )

    def _fetch_online(self, request: SearchRequest, plan: SearchPlan) -> List[SearchRecord]:
        fetched: List[SearchRecord] = []

        if request.domain == "news" and self.news_search_fn is not None and request.use_news_sources:
            try:
                fetched.extend(
                    self.news_search_fn(
                        query=plan.normalized_query,
                        limit=request.limit,
                        language=request.language,
                        category=request.category,
                    )
                )
            except Exception:
                logger.exception("news_search_fn failed for query=%s", request.query)

        if (
            request.domain in ("academic", "medical", "medical_academic", "research")
            and self.academic_search_fn is not None
            and (request.use_academic_sources or request.use_medical_sources)
        ):
            try:
                fetched.extend(
                    self.academic_search_fn(
                        query=plan.normalized_query,
                        limit=request.limit,
                        language=request.language,
                        category=request.category,
                        use_pubmed=plan.use_pubmed,
                        use_medlineplus=plan.use_medlineplus,
                        use_nimh=plan.use_nimh,
                        use_cdc=plan.use_cdc,
                        use_who=plan.use_who,
                        use_europe_pmc=plan.use_europe_pmc,
                        use_crawler=plan.use_crawler,
                        use_semantic_scholar=plan.use_semantic_scholar,
                        use_arxiv=plan.use_arxiv,
                        target_domain=request.domain,
                    )
                )
            except Exception:
                logger.exception("academic_search_fn failed for query=%s", request.query)

        if self.web_search_fn is not None and request.use_web_search:
            try:
                fetched.extend(
                    self.web_search_fn(
                        query=plan.normalized_query,
                        domain=request.domain,
                        limit=request.limit,
                        language=request.language,
                        category=request.category,
                        enable_youtube=request.enable_youtube,
                    )
                )
            except Exception:
                logger.exception("web_search_fn failed for query=%s", request.query)

        return dedupe_preserve_order(fetched, key_fn=lambda x: x.id)

    def search(self, request: SearchRequest) -> SearchResult:
        if request.use_cache:
            cached = self._search_cache(
                query=request.query,
                domain=request.domain,
                limit=request.limit,
                language=request.language,
                category=request.category,
            )
            if cached:
                sources = {}
                for item in cached:
                    sources[item.source] = sources.get(item.source, 0) + 1
                return SearchResult(
                    query=request.query,
                    domain=request.domain,
                    count=len(cached),
                    items=cached,
                    metadata={
                        "cached_count": len(cached),
                        "sources_count": sources,
                        "all_from_cache": True,
                        "fetched_ids": [],
                    },
                )

        plan = self.build_search_plan(request)
        fetched = self._fetch_online(request, plan)
        self._persist_records(fetched)

        refreshed = self._search_cache(
            query=request.query,
            domain=request.domain,
            limit=request.limit,
            language=request.language,
            category=request.category,
        )

        final_items = refreshed if refreshed else fetched[:request.limit]
        fetched_ids = {x.id for x in fetched}
        cached_count = sum(1 for x in final_items if x.id not in fetched_ids)
        sources = {}
        for item in final_items:
            sources[item.source] = sources.get(item.source, 0) + 1

        return SearchResult(
            query=request.query,
            domain=request.domain,
            count=len(final_items),
            items=final_items,
            metadata={
                "cached_count": cached_count,
                "sources_count": sources,
                "all_from_cache": False,
                "fetched_ids": list(fetched_ids),
            },
        )

    def search_news(
        self,
        query: str,
        limit: int = 8,
        language: str = "mixed",
        category: Optional[str] = None,
    ) -> SearchResult:
        return self.search(
            SearchRequest(
                query=query,
                domain="news",
                limit=limit,
                language=language,
                category=category,
            )
        )

    def search_web(
        self,
        query: str,
        limit: int = 8,
        language: str = "mixed",
        category: Optional[str] = None,
        enable_youtube: bool = False,
    ) -> SearchResult:
        return self.search(
            SearchRequest(
                query=query,
                domain="general",
                limit=limit,
                language=language,
                category=category,
                enable_youtube=enable_youtube,
            )
        )

    def search_academic(
        self,
        query: str,
        limit: int = 8,
        language: str = "en",
        category: Optional[str] = None,
    ) -> SearchResult:
        return self.search(
            SearchRequest(
                query=query,
                domain="academic",
                limit=limit,
                language=language,
                category=category,
            )
        )

    def search_medical(
        self,
        query: str,
        limit: int = 8,
        language: str = "en",
        category: Optional[str] = None,
    ) -> SearchResult:
        return self.search(
            SearchRequest(
                query=query,
                domain="medical",
                limit=limit,
                language=language,
                category=category,
            )
        )

    def read_url(
        self,
        url: str,
        *,
        domain: str = "general",
        category: str = "general",
        persist: bool = True,
        force_refresh: bool = False,
        use_crawler: bool = False,
    ) -> SearchRecord:
        normalized_url = url.strip()

        if not force_refresh:
            cached = self.file_store.get_by_url(normalized_url)
            if cached:
                return cached

        record = self.url_reader.read_url(
            normalized_url,
            domain=domain,
            category=category,
            use_crawler=use_crawler,
        )

        if persist:
            try:
                self._persist_records([record])
            except Exception as exc:
                logger.warning(
                    "read_url persistence skipped for %s due to persistence/enrichment failure: %s",
                    normalized_url,
                    exc,
                )

        return record

    def refresh_domain_cache(
        self,
        *,
        domain: str,
        queries: List[str],
        limit: int = 10,
        language: str = "mixed",
    ) -> Dict[str, object]:
        all_records: List[SearchRecord] = []

        for query in queries:
            try:
                result = self.search(
                    SearchRequest(
                        query=query,
                        domain=domain,  # type: ignore[arg-type]
                        limit=limit,
                        language=language,
                        use_cache=False,
                    )
                )
                all_records.extend(result.items)
            except Exception:
                logger.exception("refresh failed for domain=%s query=%s", domain, query)

        deduped = dedupe_preserve_order(all_records, key_fn=lambda x: x.id)
        self._persist_records(deduped)
        self.file_store.archive_daily_snapshot()

        return {
            "domain": domain,
            "query_count": len(queries),
            "record_count": len(deduped),
        }

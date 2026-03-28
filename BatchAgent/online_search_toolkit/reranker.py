from __future__ import annotations

import logging
from typing import List

from .config import RerankConfig
from .models import SearchRecord

logger = logging.getLogger(__name__)


class BaseReranker:
    def rerank(self, query: str, items: List[SearchRecord]) -> List[SearchRecord]:
        return items


class NoOpReranker(BaseReranker):
    pass


class APIReranker(BaseReranker):
    """
    Reranker that calls a TEI-compatible /v1/rerank endpoint.

    Compatible with Text Embeddings Inference (TEI) and any server that
    implements the standard reranking API:
        POST {api_url}/rerank
        { "model": "...", "query": "...", "documents": [...], "top_n": N }
    Response: { "results": [{"index": int, "relevance_score": float}, ...] }
    """

    def __init__(self, config: RerankConfig) -> None:
        self.config = config
        self._base_url = (config.api_url or "").rstrip("/")
        self._model = config.api_model or ""
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

    def rerank(self, query: str, items: List[SearchRecord]) -> List[SearchRecord]:
        if not items or not query.strip():
            return items

        # Build document texts — use summary + title for relevance scoring
        docs = []
        for r in items:
            parts = []
            if r.title:
                parts.append(r.title)
            if r.summary:
                parts.append(r.summary[:400])
            docs.append(" ".join(parts) if parts else (r.content or "")[:400])

        payload = {
            "model": self._model,
            "query": query,
            "documents": docs,
            "top_n": len(items),
            "return_documents": False,
        }

        try:
            import httpx
            resp = httpx.post(
                f"{self._base_url}/rerank",
                json=payload,
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # Sort by relevance_score descending, map back to original items
            ranked = sorted(results, key=lambda r: r.get("relevance_score", 0.0), reverse=True)
            reranked = [items[r["index"]] for r in ranked if r.get("index", -1) < len(items)]
            # Append any items that didn't appear in the response (safety fallback)
            seen = {r.get("index") for r in ranked}
            for i, item in enumerate(items):
                if i not in seen:
                    reranked.append(item)
            return reranked
        except Exception as exc:
            logger.warning("Reranker API call failed (%s). Returning original order.", exc)
            return items


def build_reranker(config: RerankConfig) -> BaseReranker:
    if config.enabled and config.provider == "api" and config.api_url and config.api_model:
        logger.info(
            "Reranker enabled: provider=api model=%s url=%s",
            config.api_model, config.api_url,
        )
        return APIReranker(config)
    return NoOpReranker()

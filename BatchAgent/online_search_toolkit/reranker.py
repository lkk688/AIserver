from __future__ import annotations

from typing import List

from .config import RerankConfig
from .models import SearchRecord


class BaseReranker:
    def rerank(self, query: str, items: List[SearchRecord]) -> List[SearchRecord]:
        return items


class NoOpReranker(BaseReranker):
    pass


def build_reranker(config: RerankConfig) -> BaseReranker:
    return NoOpReranker()
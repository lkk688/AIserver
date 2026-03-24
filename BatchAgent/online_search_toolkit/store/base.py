from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import SearchRecord


class SearchStore(ABC):
    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert_records(self, records: List[SearchRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_by_url(self, url: str) -> Optional[SearchRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_recent_records(
        self,
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
    ) -> List[SearchRecord]:
        raise NotImplementedError

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        raise NotImplementedError

    @abstractmethod
    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        raise NotImplementedError

    @abstractmethod
    def archive_daily_snapshot(self) -> None:
        raise NotImplementedError
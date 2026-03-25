from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..config import SearchConfig
from ..models import SearchRecord
from ..utils import cosine_similarity, normalize_url, utc_now
from .base import SearchStore


class FileSearchStore(SearchStore):
    def __init__(self, config: SearchConfig):
        self.config = config
        self.base_dir: Path = config.store.file_cache_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.records_path = self.base_dir / "records.json"
        self.daily_dir = self.base_dir / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        if not self.records_path.exists():
            self._write_payload([])

    def _read_records(self) -> List[SearchRecord]:
        if not self.records_path.exists():
            return []
        payload = json.loads(self.records_path.read_text(encoding="utf-8"))
        return [SearchRecord.model_validate(item) for item in payload.get("records", [])]

    def _write_payload(self, records: List[SearchRecord]) -> None:
        payload = {
            "updated_at": utc_now().isoformat(),
            "records": [r.model_dump(mode="json") for r in records],
        }
        self.records_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert_records(self, records: List[SearchRecord]) -> None:
        current = {r.id: r for r in self._read_records()}
        for record in records:
            current[record.id] = record

        merged = list(current.values())
        merged.sort(
            key=lambda x: (
                1 if x.is_breaking else 0,
                x.published_at or x.fetched_at or utc_now(),
            ),
            reverse=True,
        )
        self._write_payload(merged)

    def get_by_url(self, url: str) -> Optional[SearchRecord]:
        normalized = normalize_url(url)
        for record in self._read_records():
            if normalize_url(record.url) == normalized:
                return record
        return None

    def _filter(
        self,
        items: List[SearchRecord],
        language: Optional[str],
        domain: Optional[str],
        category: Optional[str],
        recent_hours: Optional[int],
        record_types: Optional[List[str]] = None,
    ) -> List[SearchRecord]:
        out = []
        for item in items:
            if language and item.language != language:
                continue
            if domain:
                if domain in ("academic", "medical", "research"):
                    if item.domain not in ("academic", "medical", "research"):
                        continue
                elif item.domain != domain:
                    continue
            if category and item.category != category:
                continue
            if record_types and item.record_type not in record_types:
                continue
            if recent_hours is not None:
                ts = item.published_at or item.fetched_at
                if ts is None:
                    continue
                if (utc_now() - ts).total_seconds() > recent_hours * 3600:
                    continue
            out.append(item)
        return out

    def get_recent_records(
        self,
        limit: int,
        language: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        recent_hours: Optional[int] = None,
    ) -> List[SearchRecord]:
        items = self._filter(
            self._read_records(),
            language=language,
            domain=domain,
            category=category,
            recent_hours=recent_hours,
        )
        items.sort(
            key=lambda x: (
                1 if x.is_breaking else 0,
                x.published_at or x.fetched_at or utc_now(),
            ),
            reverse=True,
        )
        return items[:limit]

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
        q_lower = query.lower()
        items = self._filter(
            self._read_records(),
            language=language,
            domain=domain,
            category=category,
            recent_hours=recent_hours,
            record_types=record_types,
        )

        scored = []
        for item in items:
            blob = " ".join([
                item.title,
                item.summary,
                item.content or "",
                item.source,
                item.category,
                item.domain,
            ]).lower()
            if q_lower in blob or (item.query and item.query.lower() == q_lower):
                scored.append((
                    1,
                    1 if item.is_breaking else 0,
                    item.published_at or item.fetched_at or utc_now(),
                    item,
                ))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [x[3] for x in scored[:limit]]

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
        items = self._filter(
            self._read_records(),
            language=language,
            domain=domain,
            category=category,
            recent_hours=recent_hours,
            record_types=record_types,
        )

        scored = []
        for item in items:
            if not item.embedding:
                continue
            score = cosine_similarity(query_embedding, item.embedding)
            if score < 0.65:
                continue
            scored.append((
                score,
                1 if item.is_breaking else 0,
                item.published_at or item.fetched_at or utc_now(),
                item,
            ))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [x[3] for x in scored[:limit]]

    def archive_daily_snapshot(self) -> None:
        stamp = utc_now().strftime("%Y-%m-%d")
        out = self.daily_dir / f"{stamp}.json"

        records = self._read_records()
        payload = {
            "date": stamp,
            "archived_at": utc_now().isoformat(),
            "records": [r.model_dump(mode="json") for r in records],
        }

        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
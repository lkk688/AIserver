from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..config import SearchConfig
from ..models import SearchRecord
from ..utils import cosine_similarity, normalize_url, utc_now
from .base import SearchStore

# Map each domain to its cache file name
_DOMAIN_FILE: Dict[str, str] = {
    "news":        "news.json",
    "academic":    "academic.json",
    "research":    "academic.json",
    "medical":     "medical.json",
    "general":     "general.json",
    "programming": "general.json",
    "finance":     "general.json",
}
_ALL_FILES = ["news.json", "academic.json", "medical.json", "general.json"]


def _domain_file(domain: str) -> str:
    return _DOMAIN_FILE.get(domain, "general.json")


class FileSearchStore(SearchStore):
    def __init__(self, config: SearchConfig):
        self.config = config
        self.base_dir: Path = config.store.file_cache_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.daily_dir = self.base_dir / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)

        # Legacy single-file path — only used for migration
        self._legacy_path = self.base_dir / "records.json"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _path(self, filename: str) -> Path:
        return self.base_dir / filename

    def _read_file(self, filename: str) -> List[SearchRecord]:
        p = self._path(filename)
        if not p.exists():
            return []
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return [SearchRecord.model_validate(item) for item in payload.get("records", [])]
        except Exception:
            return []

    def _write_file(self, filename: str, records: List[SearchRecord]) -> None:
        records_sorted = sorted(
            records,
            key=lambda x: (1 if x.is_breaking else 0, x.published_at or x.fetched_at or utc_now()),
            reverse=True,
        )
        payload = {
            "updated_at": utc_now().isoformat(),
            "category": filename.replace(".json", ""),
            "records": [r.model_dump(mode="json") for r in records_sorted],
        }
        self._path(filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_domain(self, domain: Optional[str]) -> List[SearchRecord]:
        """Read records for a specific domain (single file) or all files."""
        if domain:
            return self._read_file(_domain_file(domain))
        # Read all category files
        seen: Dict[str, SearchRecord] = {}
        for fname in _ALL_FILES:
            for r in self._read_file(fname):
                seen[r.id] = r
        return list(seen.values())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create empty category files and migrate legacy records.json if present."""
        for fname in _ALL_FILES:
            p = self._path(fname)
            if not p.exists():
                self._write_file(fname, [])

        # One-time migration from old single records.json
        if self._legacy_path.exists():
            try:
                payload = json.loads(self._legacy_path.read_text(encoding="utf-8"))
                old = [SearchRecord.model_validate(item) for item in payload.get("records", [])]
                if old:
                    self.upsert_records(old)
                    self._legacy_path.rename(self._legacy_path.with_suffix(".json.migrated"))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_records(self, records: List[SearchRecord]) -> None:
        # Group incoming records by target file
        by_file: Dict[str, List[SearchRecord]] = defaultdict(list)
        for r in records:
            by_file[_domain_file(r.domain)].append(r)

        for fname, new_records in by_file.items():
            current = {r.id: r for r in self._read_file(fname)}
            for r in new_records:
                current[r.id] = r
            self._write_file(fname, list(current.values()))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_url(self, url: str) -> Optional[SearchRecord]:
        normalized = normalize_url(url)
        for fname in _ALL_FILES:
            for record in self._read_file(fname):
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
                if domain == "medical":
                    if item.domain != "medical":
                        continue
                elif domain in ("academic", "research"):
                    if item.domain not in ("academic", "research"):
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
            self._read_domain(domain),
            language=language,
            domain=domain,
            category=category,
            recent_hours=recent_hours,
        )
        items.sort(
            key=lambda x: (1 if x.is_breaking else 0, x.published_at or x.fetched_at or utc_now()),
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
            self._read_domain(domain),
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
            self._read_domain(domain),
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
        snapshot_dir = self.daily_dir / stamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for fname in _ALL_FILES:
            records = self._read_file(fname)
            if not records:
                continue
            out = snapshot_dir / fname
            payload = {
                "date": stamp,
                "archived_at": utc_now().isoformat(),
                "category": fname.replace(".json", ""),
                "records": [r.model_dump(mode="json") for r in records],
            }
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

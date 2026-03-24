from __future__ import annotations

import hashlib
import html
import math
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_html_text(raw: str) -> str:
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def trim_summary(text: str, max_len: int) -> str:
    text = clean_html_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    out = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        out += f"?{parsed.query}"
    return out


def make_article_id(source: str, title: str, url: str) -> str:
    key = f"{source}|{title}|{normalize_url(url)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def detect_language(title: str, summary: str) -> str:
    blob = f"{title} {summary}"
    if re.search(r"[\u4e00-\u9fff]", blob):
        return "zh"
    if re.search(r"[A-Za-z]", blob):
        return "en"
    return "unknown"


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def is_recent(dt: Optional[datetime], hours: int) -> bool:
    if not dt:
        return False
    return (utc_now() - dt) <= timedelta(hours=hours)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def build_embedding_text(title: str, source: str, category: str, summary: str, content: str = "") -> str:
    text = (
        f"Title: {title}\n"
        f"Source: {source}\n"
        f"Category: {category}\n"
        f"Summary: {summary}"
    )
    if content:
        text += f"\nContent: {content[:4000]}"
    return text


def dedupe_preserve_order(items: Iterable, key_fn):
    seen = set()
    out = []
    for item in items:
        k = key_fn(item)
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out
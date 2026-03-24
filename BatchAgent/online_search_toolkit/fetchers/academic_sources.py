from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from ..models import SearchRecord
from ..utils import clean_html_text, make_article_id, normalize_url, trim_summary, utc_now

logger = logging.getLogger(__name__)


class AcademicSourceFetcher:
    def __init__(
        self,
        max_summary_chars: int = 240,
        timeout_seconds: int = 20,
        crawler_search_fn=None,
    ):
        self.max_summary_chars = max_summary_chars
        self.timeout_seconds = timeout_seconds
        self.crawler_search_fn = crawler_search_fn

    def _make_record(
        self,
        *,
        title: str,
        summary: str,
        url: str,
        source: str,
        source_type: str,
        record_type: str,
        domain: str,
        category: str = "general",
        query: Optional[str] = None,
        published_at=None,
        metadata: Optional[Dict] = None,
    ) -> SearchRecord:
        normalized_url = normalize_url(url)
        clean_title = clean_html_text(title) or normalized_url
        clean_summary = trim_summary(summary or "", self.max_summary_chars)

        return SearchRecord(
            id=make_article_id(source, clean_title, normalized_url),
            record_type=record_type,  # type: ignore[arg-type]
            source_type=source_type,  # type: ignore[arg-type]
            title=clean_title,
            summary=clean_summary,
            url=normalized_url,
            content=None,
            source=source,
            domain=domain,  # type: ignore[arg-type]
            language="en",
            category=category,
            query=query,
            published_at=published_at,
            fetched_at=utc_now(),
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # PubMed
    # ------------------------------------------------------------------

    def search_pubmed(self, query: str, limit: int = 8) -> List[SearchRecord]:
        try:
            esearch_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&retmode=json&retmax={limit}&term={quote_plus(query)}"
            )
            esearch = requests.get(esearch_url, timeout=self.timeout_seconds)
            esearch.raise_for_status()
            ids = esearch.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            esummary_url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                f"?db=pubmed&retmode=json&id={','.join(ids)}"
            )
            esummary = requests.get(esummary_url, timeout=self.timeout_seconds)
            esummary.raise_for_status()
            result = esummary.json().get("result", {})

            records: List[SearchRecord] = []
            for pmid in ids:
                item = result.get(pmid, {})
                title = item.get("title", f"PubMed {pmid}")
                pubdate = item.get("pubdate")
                source = item.get("source", "PubMed")
                authors = ", ".join(a.get("name", "") for a in item.get("authors", [])[:5])
                summary = f"{source}. {pubdate or ''}. {authors}".strip()
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                records.append(
                    self._make_record(
                        title=title,
                        summary=summary,
                        url=url,
                        source="PubMed",
                        source_type="pubmed",
                        record_type="academic_paper",
                        domain="academic",
                        category="medical_research",
                        query=query,
                        metadata={"pmid": pmid},
                    )
                )
            return records
        except Exception as exc:
            logger.warning("PubMed search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # MedlinePlus
    # ------------------------------------------------------------------

    def search_medlineplus(self, query: str, limit: int = 8) -> List[SearchRecord]:
        try:
            url = f"https://medlineplus.gov/search/?query={quote_plus(query)}"
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            records: List[SearchRecord] = []

            for a in soup.select("a")[: limit * 5]:
                href = a.get("href") or ""
                title = clean_html_text(a.get_text(" ", strip=True))
                if not href or not title:
                    continue
                if "medlineplus.gov" not in href and not href.startswith("/"):
                    continue

                full_url = href if href.startswith("http") else f"https://medlineplus.gov{href}"
                parent_text = clean_html_text(a.parent.get_text(" ", strip=True)) if a.parent else ""
                summary = trim_summary(parent_text, self.max_summary_chars)

                records.append(
                    self._make_record(
                        title=title,
                        summary=summary,
                        url=full_url,
                        source="MedlinePlus",
                        source_type="medlineplus",
                        record_type="medical_article",
                        domain="medical",
                        category="consumer_health",
                        query=query,
                    )
                )

            dedup = {r.id: r for r in records}
            return list(dedup.values())[:limit]
        except Exception as exc:
            logger.warning("MedlinePlus search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # NIMH
    # ------------------------------------------------------------------

    def search_nimh(self, query: str, limit: int = 8) -> List[SearchRecord]:
        try:
            url = f"https://www.nimh.nih.gov/search?query={quote_plus(query)}"
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            records: List[SearchRecord] = []

            for a in soup.select("a")[: limit * 5]:
                href = a.get("href") or ""
                title = clean_html_text(a.get_text(" ", strip=True))
                if not href or not title:
                    continue

                full_url = href if href.startswith("http") else f"https://www.nimh.nih.gov{href}"
                if "nimh.nih.gov" not in full_url:
                    continue

                parent_text = clean_html_text(a.parent.get_text(" ", strip=True)) if a.parent else ""
                if len(parent_text) < 20:
                    continue

                records.append(
                    self._make_record(
                        title=title,
                        summary=parent_text,
                        url=full_url,
                        source="NIMH",
                        source_type="nimh",
                        record_type="medical_article",
                        domain="medical",
                        category="mental_health",
                        query=query,
                    )
                )

            dedup = {r.id: r for r in records}
            return list(dedup.values())[:limit]
        except Exception as exc:
            logger.warning("NIMH search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Semantic Scholar (free API, no key required)
    # ------------------------------------------------------------------

    def search_semantic_scholar(self, query: str, limit: int = 8) -> List[SearchRecord]:
        """Search Semantic Scholar API (free, no key required)."""
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": min(limit, 100),
                "fields": "title,abstract,url,year,authors,externalIds,citationCount",
            }
            response = requests.get(api_url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()

            records: List[SearchRecord] = []
            for paper in data.get("data", []):
                title = paper.get("title", "")
                abstract = paper.get("abstract") or ""
                paper_url = paper.get("url") or ""
                year = paper.get("year")
                authors = ", ".join(
                    a.get("name", "") for a in (paper.get("authors") or [])[:5]
                )
                citation_count = paper.get("citationCount", 0)
                external_ids = paper.get("externalIds") or {}

                summary_parts = []
                if authors:
                    summary_parts.append(authors)
                if year:
                    summary_parts.append(str(year))
                if citation_count:
                    summary_parts.append(f"{citation_count} citations")
                if abstract:
                    summary_parts.append(abstract)

                records.append(
                    self._make_record(
                        title=title,
                        summary=". ".join(summary_parts),
                        url=paper_url,
                        source="Semantic Scholar",
                        source_type="search_api",
                        record_type="academic_paper",
                        domain="academic",
                        category="research",
                        query=query,
                        metadata={
                            "year": year,
                            "citation_count": citation_count,
                            "external_ids": external_ids,
                        },
                    )
                )
            return records[:limit]
        except Exception as exc:
            logger.warning("Semantic Scholar search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # arXiv (Atom API, free, no key required)
    # ------------------------------------------------------------------

    def search_arxiv(self, query: str, limit: int = 8) -> List[SearchRecord]:
        """Search arXiv via their Atom API."""
        try:
            api_url = (
                f"http://export.arxiv.org/api/query"
                f"?search_query=all:{quote_plus(query)}"
                f"&start=0&max_results={limit}"
                f"&sortBy=relevance&sortOrder=descending"
            )
            response = requests.get(api_url, timeout=self.timeout_seconds)
            response.raise_for_status()

            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(response.text)

            records: List[SearchRecord] = []
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""

                summary_el = entry.find("atom:summary", ns)
                abstract = re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else ""

                paper_url = ""
                pdf_url = ""
                for link_el in entry.findall("atom:link", ns):
                    rel = link_el.get("rel", "")
                    href = link_el.get("href", "")
                    link_type = link_el.get("type", "")
                    if rel == "alternate":
                        paper_url = href
                    elif "pdf" in link_type or href.endswith(".pdf"):
                        pdf_url = href

                authors = []
                for author_el in entry.findall("atom:author", ns):
                    name_el = author_el.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append(name_el.text.strip())

                published_el = entry.find("atom:published", ns)
                published = (published_el.text or "").strip() if published_el is not None else ""

                arxiv_id = ""
                if paper_url:
                    parts = paper_url.rstrip("/").split("/")
                    if parts:
                        arxiv_id = parts[-1]

                summary_parts = []
                if authors:
                    summary_parts.append(", ".join(authors[:5]))
                if published:
                    summary_parts.append(published[:10])
                if abstract:
                    summary_parts.append(abstract)

                records.append(
                    self._make_record(
                        title=title,
                        summary=". ".join(summary_parts),
                        url=paper_url or "",
                        source="arXiv",
                        source_type="search_api",
                        record_type="academic_paper",
                        domain="academic",
                        category="research",
                        query=query,
                        metadata={
                            "arxiv_id": arxiv_id,
                            "pdf_url": pdf_url,
                            "published": published,
                        },
                    )
                )
            return records[:limit]
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Crawler
    # ------------------------------------------------------------------

    def search_crawler(self, query: str, limit: int = 8, domain: str = "academic") -> List[SearchRecord]:
        if self.crawler_search_fn is None:
            return []
        try:
            raw_items = self.crawler_search_fn(query=query, limit=limit, domain=domain) or []
            records: List[SearchRecord] = []

            for item in raw_items:
                records.append(
                    self._make_record(
                        title=item.get("title", ""),
                        summary=item.get("summary", "") or item.get("content", ""),
                        url=item.get("url", ""),
                        source=item.get("source", "Crawler"),
                        source_type="crawler",
                        record_type="crawler_page",
                        domain=domain,
                        category=item.get("category", "general"),
                        query=query,
                        metadata=item.get("metadata", {}),
                    )
                )
            return records
        except Exception as exc:
            logger.warning("Crawler search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Unified search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 8,
        language: str = "en",
        category: str | None = None,
        use_pubmed: bool = True,
        use_medlineplus: bool = True,
        use_nimh: bool = False,
        use_crawler: bool = False,
        use_semantic_scholar: bool = True,
        use_arxiv: bool = True,
    ) -> List[SearchRecord]:
        results: List[SearchRecord] = []

        if use_pubmed:
            results.extend(self.search_pubmed(query, limit))
        if use_semantic_scholar:
            results.extend(self.search_semantic_scholar(query, limit))
        if use_arxiv:
            results.extend(self.search_arxiv(query, limit))
        if use_medlineplus:
            results.extend(self.search_medlineplus(query, limit))
        if use_nimh:
            results.extend(self.search_nimh(query, limit))
        if use_crawler:
            results.extend(self.search_crawler(query, limit, domain="medical" if use_nimh else "academic"))

        if category:
            results = [r for r in results if r.category == category]

        dedup = {r.id: r for r in results}
        final = list(dedup.values())
        final.sort(key=lambda x: x.fetched_at, reverse=True)
        return final[: max(limit * 3, 20)]
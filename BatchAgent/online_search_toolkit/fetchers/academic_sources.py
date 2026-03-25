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

    def search_pubmed(self, query: str, limit: int = 8, target_domain: str = "medical") -> List[SearchRecord]:
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
                        domain=target_domain,
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
            # MedlinePlus healthTopics DB works best with short queries (≤4 words);
            # drop org-name prefixes (AAP, CDC, WHO, NIH) and take leading keywords.
            _stop = {"aap", "cdc", "who", "nih", "fda", "usda", "acsm", "aha"}
            words = [w for w in query.split() if w.lower() not in _stop]
            short_query = " ".join(words[:4])
            url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={quote_plus(short_query)}&retmax={limit}"
            response = requests.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()

            root = ET.fromstring(response.text)
            records: List[SearchRecord] = []

            for doc in root.findall(".//document")[:limit]:
                doc_url = doc.get("url", "")
                title = ""
                summary = ""
                for content in doc.findall("content"):
                    name = content.get("name")
                    if name == "title":
                        title = content.text or ""
                    elif name == "FullSummary":
                        summary = clean_html_text(content.text or "")
                    elif name == "snippet" and not summary:
                        summary = clean_html_text(content.text or "")

                if not doc_url or not title:
                    continue

                records.append(
                    self._make_record(
                        title=title,
                        summary=summary,
                        url=doc_url,
                        source="MedlinePlus",
                        source_type="medlineplus",
                        record_type="medical_article",
                        domain="medical",
                        category="consumer_health",
                        query=query,
                    )
                )

            return records
        except Exception as exc:
            logger.warning("MedlinePlus search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # NIMH
    # ------------------------------------------------------------------

    def search_nimh(self, query: str, limit: int = 8) -> List[SearchRecord]:
        try:
            url = f"https://www.nimh.nih.gov/search?query={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
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
            if response.status_code == 429:
                logger.debug("Semantic Scholar rate limited (429 HTTP Error). Skipping.")
                return []
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
    # Europe PMC  (free API, covers PubMed + PMC + WHO guidelines + preprints)
    # ------------------------------------------------------------------

    def search_europe_pmc(self, query: str, limit: int = 8, target_domain: str = "medical") -> List[SearchRecord]:
        """Search Europe PMC REST API (free, no key). Covers PubMed, PMC, WHO docs."""
        try:
            params = {
                "query": query,
                "format": "json",
                "resulttype": "lite",
                "pageSize": min(limit, 25),
            }
            response = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            records: List[SearchRecord] = []
            for item in data.get("resultList", {}).get("result", [])[:limit]:
                title = item.get("title", "")
                if not title:
                    continue
                pmid = item.get("pmid", "")
                pmcid = item.get("pmcid", "")
                doi = item.get("doi", "")
                abstract = item.get("abstractText") or ""
                authors = item.get("authorString", "")
                year = str(item.get("pubYear", ""))
                journal = item.get("journalTitle", "")
                source_code = item.get("source", "")
                citations = item.get("citedByCount", 0)

                if pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                elif pmcid:
                    url = f"https://europepmc.org/article/PMC/{pmcid.replace('PMC', '')}"
                elif doi:
                    url = f"https://doi.org/{doi}"
                else:
                    continue

                summary_parts = [p for p in [authors[:120], year, journal, abstract] if p]

                records.append(
                    self._make_record(
                        title=title,
                        summary=". ".join(summary_parts),
                        url=url,
                        source="Europe PMC",
                        source_type="search_api",
                        record_type="academic_paper",
                        domain=target_domain,
                        category="medical_research",
                        query=query,
                        metadata={
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "doi": doi,
                            "pub_year": year,
                            "epmc_source": source_code,
                            "citation_count": citations,
                        },
                    )
                )
            return records[:limit]
        except Exception as exc:
            logger.warning("Europe PMC search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # CDC (Centers for Disease Control and Prevention) — HTML search page
    # ------------------------------------------------------------------

    def search_cdc(self, query: str, limit: int = 8) -> List[SearchRecord]:
        """Search CDC via their main HTML search page (search.cdc.gov/search/)."""
        try:
            url = f"https://search.cdc.gov/search/?query={quote_plus(query)}&affiliate=cdc-main"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = requests.get(url, headers=headers, timeout=self.timeout_seconds)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            records: List[SearchRecord] = []

            # CDC search results use various selectors across redesigns — try all
            result_blocks = (
                soup.select("li.content_item")
                or soup.select("li.search-result-item")
                or soup.select("div.result")
                or soup.select("li[data-module='search-result']")
            )

            for block in result_blocks[: limit * 3]:
                a = block.find("a", href=True)
                if not a:
                    continue
                title = clean_html_text(a.get_text(" ", strip=True))
                href = a["href"]
                full_url = href if href.startswith("http") else f"https://www.cdc.gov{href}"
                if "cdc.gov" not in full_url:
                    continue
                desc_el = block.find("p") or block.find(class_=re.compile(r"desc|snippet|summary"))
                description = clean_html_text(desc_el.get_text(" ", strip=True)) if desc_el else ""
                if not title:
                    continue
                records.append(
                    self._make_record(
                        title=title,
                        summary=description,
                        url=full_url,
                        source="CDC",
                        source_type="search_api",
                        record_type="medical_article",
                        domain="medical",
                        category="health_guidelines",
                        query=query,
                    )
                )

            dedup = {r.id: r for r in records}
            return list(dedup.values())[:limit]
        except Exception as exc:
            logger.warning("CDC search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # WHO (World Health Organization) — publications search API
    # ------------------------------------------------------------------

    def search_who(self, query: str, limit: int = 8) -> List[SearchRecord]:
        """Search WHO publications via their IRIS/publications search."""
        try:
            # WHO IRIS (Institutional Repository for Information Sharing) — free REST API
            params = {
                "query": query,
                "rows": min(limit, 20),
                "wt": "json",
            }
            response = requests.get(
                "https://iris.who.int/rest/search",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            records: List[SearchRecord] = []
            docs = data.get("response", {}).get("docs", [])
            for doc in docs[:limit]:
                title = doc.get("dc.title", [""])[0] if isinstance(doc.get("dc.title"), list) else doc.get("dc.title", "")
                if not title:
                    continue
                handle = doc.get("handle", "")
                item_url = f"https://iris.who.int/handle/{handle}" if handle else ""
                if not item_url:
                    continue
                desc_raw = doc.get("dc.description", doc.get("dc.description.abstract", ""))
                if isinstance(desc_raw, list):
                    desc_raw = " ".join(desc_raw)
                summary = clean_html_text(str(desc_raw))
                date = doc.get("dc.date.issued", [""])[0] if isinstance(doc.get("dc.date.issued"), list) else ""
                authors_raw = doc.get("dc.contributor.author", [])
                if isinstance(authors_raw, list):
                    authors_str = ", ".join(authors_raw[:5])
                else:
                    authors_str = str(authors_raw)

                records.append(
                    self._make_record(
                        title=title,
                        summary=f"{authors_str}. {date}. {summary}".strip(". "),
                        url=item_url,
                        source="WHO",
                        source_type="search_api",
                        record_type="medical_article",
                        domain="medical",
                        category="health_guidelines",
                        query=query,
                        metadata={"handle": handle, "date": date},
                    )
                )

            return records[:limit]
        except Exception as exc:
            logger.warning("WHO search failed: %s", exc)
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
        use_cdc: bool = False,
        use_who: bool = False,
        use_europe_pmc: bool = False,
        use_crawler: bool = False,
        use_semantic_scholar: bool = True,
        use_arxiv: bool = True,
        target_domain: str = "academic",
    ) -> List[SearchRecord]:
        results: List[SearchRecord] = []

        if use_pubmed:
            results.extend(self.search_pubmed(query, limit, target_domain=target_domain))
        if use_europe_pmc:
            results.extend(self.search_europe_pmc(query, limit, target_domain=target_domain))
        if use_semantic_scholar:
            results.extend(self.search_semantic_scholar(query, limit))
        if use_arxiv:
            results.extend(self.search_arxiv(query, limit))
        if use_medlineplus:
            results.extend(self.search_medlineplus(query, limit))
        if use_nimh:
            results.extend(self.search_nimh(query, limit))
        if use_cdc:
            results.extend(self.search_cdc(query, limit))
        if use_who:
            results.extend(self.search_who(query, limit))
        if use_crawler:
            results.extend(self.search_crawler(query, limit, domain="medical" if use_nimh else "academic"))

        if category:
            results = [r for r in results if r.category == category]

        dedup = {r.id: r for r in results}
        final = list(dedup.values())
        final.sort(key=lambda x: x.fetched_at, reverse=True)
        return final[: max(limit * 3, 20)]
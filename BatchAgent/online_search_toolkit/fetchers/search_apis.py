from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from urllib.parse import quote

from ..models import SearchRecord
from ..utils import clean_html_text, make_article_id, normalize_url, trim_summary, utc_now


class SearchAPIFetcher:
    def __init__(
        self,
        max_summary_chars: int = 240,
        serper_api_key: str = "",
        tavily_api_key: str = "",
    ):
        self.max_summary_chars = max_summary_chars
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY", "")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY", "")

    def _make_record(
        self,
        *,
        title: str,
        summary: str,
        url: str,
        source: str,
        domain: str,
        category: str = "general",
        query: Optional[str] = None,
        record_type: str = "web_page",
        source_type: str = "search_api",
    ) -> SearchRecord:
        normalized_url = normalize_url(url)
        clean_title = clean_html_text(title) or normalized_url
        clean_summary = trim_summary(summary or "", self.max_summary_chars)

        return SearchRecord(
            id=make_article_id(source, clean_title, normalized_url),
            record_type=record_type,      # type: ignore[arg-type]
            source_type=source_type,      # type: ignore[arg-type]
            title=clean_title,
            summary=clean_summary,
            url=normalized_url,
            content=None,
            source=source,
            domain=domain,                # type: ignore[arg-type]
            language="unknown",
            category=category,
            query=query,
            fetched_at=utc_now(),
            metadata={},
        )

    # Canonical domain → site-filter string
    _DOMAIN_FILTER_MAP: Dict[str, str] = {
        "news":             "site:reuters.com OR site:apnews.com OR site:bbc.com OR site:theguardian.com OR site:bloomberg.com",
        "academic":         "site:arxiv.org OR site:scholar.google.com OR site:semanticscholar.org OR site:pubmed.ncbi.nlm.nih.gov",
        "medical":          "site:pubmed.ncbi.nlm.nih.gov OR site:medlineplus.gov OR site:nih.gov OR site:mayoclinic.org OR site:webmd.com",
        "medical_academic": "site:pubmed.ncbi.nlm.nih.gov OR site:europepmc.org OR site:ncbi.nlm.nih.gov",
        "research":         "site:arxiv.org OR site:semanticscholar.org OR site:scholar.google.com OR site:jstor.org",
        "programming":      "site:stackoverflow.com OR site:github.com OR site:docs.python.org OR site:pypi.org",
        "code":             "site:stackoverflow.com OR site:github.com OR site:docs.python.org OR site:pypi.org",
        "software_eng":     "site:stackoverflow.com OR site:github.com OR site:dev.to OR site:pypi.org OR site:docs.python.org",
        "math":             "site:artofproblemsolving.com OR site:math.stackexchange.com OR site:mathworld.wolfram.com OR site:khanacademy.org",
        "science":          "site:nature.com OR site:sciencedirect.com OR site:phys.org OR site:wolframalpha.com OR site:pnas.org",
        "language":         "site:wiktionary.org OR site:languagelearning.stackexchange.com OR site:bbc.co.uk/languages",
        "business":         "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com OR site:wsj.com",
        "finance":          "site:sec.gov OR site:finance.yahoo.com OR site:bloomberg.com OR site:investopedia.com OR site:wsj.com",
        "assistant":        "site:superuser.com OR site:askubuntu.com OR site:serverfault.com OR site:apple.stackexchange.com",
        "sales_support":    "site:zendesk.com OR site:hubspot.com OR site:salesforce.com OR site:freshdesk.com",
        "health":           "site:pubmed.ncbi.nlm.nih.gov OR site:medlineplus.gov OR site:nih.gov OR site:mayoclinic.org",
        "general":          "",
    }

    def _domain_filter(self, domain: str) -> str:
        return self._DOMAIN_FILTER_MAP.get(domain, "")

    def search_serper(self, query: str, domain: str, limit: int = 8) -> List[SearchRecord]:
        if not self.serper_api_key or self.serper_api_key == "EMPTY":
            return []

        final_query = query
        filt = self._domain_filter(domain)
        if filt:
            final_query = f"{query} {filt}"

        req = urllib.request.Request("https://google.serper.dev/search", method="POST")
        req.add_header("X-API-KEY", self.serper_api_key)
        req.add_header("Content-Type", "application/json")

        fetch_limit = max(limit, 20)
        payload = json.dumps({"q": final_query, "num": fetch_limit}).encode("utf-8")

        try:
            with urllib.request.urlopen(req, data=payload, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        out: List[SearchRecord] = []

        answer_box = data.get("answerBox", {})
        if answer_box and answer_box.get("snippet"):
            out.append(
                self._make_record(
                    title=f"Direct answer for: {query}",
                    summary=answer_box.get("snippet", ""),
                    url=answer_box.get("link") or "",
                    source="Serper AnswerBox",
                    domain=domain,
                    query=query,
                )
            )

        knowledge_graph = data.get("knowledgeGraph", {})
        if knowledge_graph and knowledge_graph.get("description"):
            out.append(
                self._make_record(
                    title=knowledge_graph.get("title") or f"Knowledge panel for: {query}",
                    summary=knowledge_graph.get("description", ""),
                    url=knowledge_graph.get("website") or "",
                    source="Serper KnowledgeGraph",
                    domain=domain,
                    query=query,
                )
            )

        for item in data.get("organic", []):
            out.append(
                self._make_record(
                    title=item.get("title", ""),
                    summary=item.get("snippet", ""),
                    url=item.get("link", ""),
                    source="Serper",
                    domain=domain,
                    query=query,
                )
            )

        return out

    def search_tavily(self, query: str, domain: str, limit: int = 5) -> List[SearchRecord]:
        if not self.tavily_api_key or self.tavily_api_key == "EMPTY":
            return []

        payload = json.dumps({
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": limit,
            "include_answer": True,
        }).encode("utf-8")

        req = urllib.request.Request("https://api.tavily.com/search", method="POST")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, data=payload, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        out: List[SearchRecord] = []

        if data.get("answer"):
            out.append(
                self._make_record(
                    title=f"Tavily answer for: {query}",
                    summary=data["answer"],
                    url="",
                    source="Tavily Answer",
                    domain=domain,
                    query=query,
                )
            )

        for item in data.get("results", [])[:limit]:
            out.append(
                self._make_record(
                    title=item.get("title", ""),
                    summary=(item.get("content") or item.get("snippet") or ""),
                    url=item.get("url", ""),
                    source="Tavily",
                    domain=domain,
                    query=query,
                )
            )

        return out

    def search_wikimedia(self, query: str, limit: int = 3) -> List[SearchRecord]:
        req = urllib.request.Request(
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={quote(query)}&srlimit={limit}&format=json&utf8=1",
            method="GET",
        )
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "OnlineSearchToolkit/1.0")

        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        items = payload.get("query", {}).get("search", []) or []
        out: List[SearchRecord] = []

        for item in items:
            title = str(item.get("title") or "").strip()
            if not title:
                continue

            snippet_html = str(item.get("snippet") or "").strip()
            description = html.unescape(re.sub(r"<[^>]+>", "", snippet_html)).strip()
            wiki_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

            out.append(
                self._make_record(
                    title=title,
                    summary=description,
                    url=wiki_url,
                    source="Wikimedia",
                    domain="general",
                    query=query,
                    record_type="wiki_entry",
                    source_type="wikimedia",
                )
            )

        return out

    def search_youtube(self, query: str, limit: int = 5) -> List[SearchRecord]:
        """Search YouTube via Serper's videos endpoint."""
        if not self.serper_api_key or self.serper_api_key == "EMPTY":
            return []

        req = urllib.request.Request("https://google.serper.dev/videos", method="POST")
        req.add_header("X-API-KEY", self.serper_api_key)
        req.add_header("Content-Type", "application/json")

        fetch_limit = max(limit, 20)
        payload = json.dumps({"q": query, "num": fetch_limit}).encode("utf-8")

        try:
            with urllib.request.urlopen(req, data=payload, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        out: List[SearchRecord] = []
        for item in data.get("videos", []):
            url = item.get("link", "")
            if not url:
                continue

            title = item.get("title", "")
            snippet = item.get("snippet", "")
            channel = item.get("channel", "")
            duration = item.get("duration", "")

            summary_parts = []
            if channel:
                summary_parts.append(f"Channel: {channel}")
            if duration:
                summary_parts.append(f"Duration: {duration}")
            if snippet:
                summary_parts.append(snippet)

            out.append(
                self._make_record(
                    title=title,
                    summary=" | ".join(summary_parts),
                    url=url,
                    source="YouTube",
                    domain="general",
                    query=query,
                    record_type="youtube_video",
                    source_type="youtube",
                )
            )

        return out

    def search(
        self,
        query: str,
        domain: str = "general",
        limit: int = 8,
        use_serper: bool = True,
        use_tavily: bool = True,
        use_wikimedia: bool = True,
        enable_youtube: bool = False,
    ) -> List[SearchRecord]:
        results: List[SearchRecord] = []

        if use_serper:
            results.extend(self.search_serper(query=query, domain=domain, limit=limit))

        # Run Tavily as a supplement when serper returns fewer than requested results,
        # or as the sole source when serper is disabled/unavailable.
        if use_tavily and len(results) < limit:
            results.extend(self.search_tavily(query=query, domain=domain, limit=limit))

        if use_wikimedia:
            results.extend(self.search_wikimedia(query=query, limit=min(limit, 3)))

        if enable_youtube:
            results.extend(self.search_youtube(query=query, limit=min(limit, 5)))

        dedup = {}
        for item in results:
            dedup[item.id] = item
        return list(dedup.values())
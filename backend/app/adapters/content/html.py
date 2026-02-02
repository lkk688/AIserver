from bs4 import BeautifulSoup
import requests
from backend.app.domain.ports import ContentExtractor
from backend.app.domain.models import ExtractedContent
from backend.app.domain.errors import ExtractionError
from backend.app.config.schema import AppConfig

class HTMLExtractor(ContentExtractor):
    def __init__(self, config: AppConfig):
        self.enabled = config.web_fetch.enabled
        self.timeout = config.web_fetch.timeout_sec
        self.user_agent = config.web_fetch.user_agent

    def extract(self, document_uri: str) -> ExtractedContent:
        if not self.enabled and (document_uri.startswith("http://") or document_uri.startswith("https://")):
             raise ExtractionError("Web fetch is disabled in configuration")

        try:
            if document_uri.startswith("http://") or document_uri.startswith("https://"):
                headers = {"User-Agent": self.user_agent}
                response = requests.get(document_uri, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                html_content = response.text
                content_type = response.headers.get("Content-Type", "text/html")
            else:
                # Local file
                path = document_uri.replace("file://", "")
                with open(path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                content_type = "text/html"

            soup = BeautifulSoup(html_content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()

            heading_nodes = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            heading_sequence = []
            for node in heading_nodes:
                title_text = node.get_text(strip=True)
                if not title_text:
                    continue
                name = node.name.lower()
                level = 1
                if len(name) == 2 and name[1].isdigit():
                    level = int(name[1])
                heading_sequence.append((title_text, level))

            raw_text = soup.get_text(separator="\n")
            lines = (line.strip() for line in raw_text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned = "\n".join(chunk for chunk in chunks if chunk)

            line_items = cleaned.splitlines()
            paragraphs = []
            current = []
            for line in line_items:
                stripped = line.strip()
                if not stripped:
                    if current:
                        paragraphs.append(" ".join(current))
                        current = []
                else:
                    current.append(stripped)
            if current:
                paragraphs.append(" ".join(current))

            normalized_text = "\n\n".join(paragraphs)

            sections = []
            search_start = 0
            for title_text, level in heading_sequence:
                idx = normalized_text.find(title_text, search_start)
                if idx == -1:
                    continue
                sections.append(
                    {
                        "title": title_text,
                        "level": level,
                        "start": idx,
                        "end": idx + len(title_text),
                    }
                )
                search_start = idx + len(title_text)

            if sections:
                sections_sorted = sorted(sections, key=lambda s: s["start"])
                for i, s in enumerate(sections_sorted):
                    if i + 1 < len(sections_sorted):
                        s["end"] = sections_sorted[i + 1]["start"]
                    else:
                        s["end"] = None
                sections = sections_sorted

            title = soup.title.string if soup.title else None

            extra = {}
            if sections:
                extra["sections"] = sections

            return ExtractedContent(
                text=normalized_text,
                title=title,
                mime_type=content_type,
                extra=extra
            )

        except Exception as e:
            raise ExtractionError(f"Failed to extract HTML content from {document_uri}: {str(e)}")

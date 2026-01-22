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
            
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            title = soup.title.string if soup.title else None

            return ExtractedContent(
                text=text,
                title=title,
                mime_type=content_type,
                extra={}
            )

        except Exception as e:
            raise ExtractionError(f"Failed to extract HTML content from {document_uri}: {str(e)}")

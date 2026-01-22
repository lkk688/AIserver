import re
from urllib.parse import urlparse
from backend.app.domain.ports import ContentExtractor
from backend.app.domain.models import ExtractedContent
from backend.app.domain.errors import ExtractionError
import requests

class MarkdownExtractor(ContentExtractor):
    def extract(self, document_uri: str) -> ExtractedContent:
        try:
            parsed = urlparse(document_uri)
            content = ""
            
            if parsed.scheme in ('http', 'https'):
                response = requests.get(document_uri, timeout=10)
                response.raise_for_status()
                content = response.text
            else:
                path = document_uri
                if parsed.scheme == 'file':
                    path = parsed.path
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Parse frontmatter if present
            # Simple regex for YAML frontmatter
            frontmatter_regex = r"^---\s+(.*?)\s+---\s+(.*)$"
            match = re.search(frontmatter_regex, content, re.DOTALL | re.MULTILINE)
            
            title = None
            text = content
            
            if match:
                frontmatter = match.group(1)
                text = match.group(2)
                
                # Try to extract title from frontmatter
                title_match = re.search(r"^title:\s+(.+)$", frontmatter, re.MULTILINE)
                if title_match:
                    title = title_match.group(1).strip().strip('"').strip("'")
            
            # If no title in frontmatter, try first h1
            if not title:
                h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                if h1_match:
                    title = h1_match.group(1).strip()

            return ExtractedContent(
                text=text.strip(),
                title=title,
                mime_type="text/markdown",
                extra={}
            )

        except Exception as e:
            raise ExtractionError(f"Failed to extract Markdown content from {document_uri}: {str(e)}")

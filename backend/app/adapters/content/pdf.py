from pypdf import PdfReader
from backend.app.domain.ports import ContentExtractor
from backend.app.domain.models import ExtractedContent
from backend.app.domain.errors import ExtractionError
import io
import requests
from urllib.parse import urlparse

class PDFExtractor(ContentExtractor):
    def extract(self, document_uri: str) -> ExtractedContent:
        try:
            # Handle local file or URL
            parsed = urlparse(document_uri)
            if parsed.scheme in ('http', 'https'):
                response = requests.get(document_uri, timeout=10)
                response.raise_for_status()
                f = io.BytesIO(response.content)
            else:
                # Assume local path if no scheme or file scheme
                path = document_uri
                if parsed.scheme == 'file':
                    path = parsed.path
                f = open(path, 'rb')

            try:
                reader = PdfReader(f)
                text = ""
                page_count = len(reader.pages)
                
                # Extract text from all pages
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                # Extract metadata
                info = reader.metadata
                title = info.title if info and info.title else None
                
                return ExtractedContent(
                    text=text.strip(),
                    title=title,
                    mime_type="application/pdf",
                    extra={"page_count": page_count}
                )
            finally:
                if not isinstance(f, io.BytesIO):
                    f.close()
                    
        except Exception as e:
            raise ExtractionError(f"Failed to extract PDF content from {document_uri}: {str(e)}")

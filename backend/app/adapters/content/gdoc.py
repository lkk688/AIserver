from backend.app.domain.ports import ContentExtractor
from backend.app.domain.models import ExtractedContent
from backend.app.domain.errors import ExtractionError
from backend.app.adapters.content.html import HTMLExtractor
from backend.app.config.schema import AppConfig

class GoogleDocExtractor(ContentExtractor):
    def __init__(self, config: AppConfig):
        # Re-use HTML extractor for MVP since GDocs can be viewed as web pages
        self.html_extractor = HTMLExtractor(config)

    def extract(self, document_uri: str) -> ExtractedContent:
        # Check if it's a GDoc URL
        if "docs.google.com" not in document_uri:
             raise ExtractionError(f"Not a recognized Google Doc URL: {document_uri}")
        
        # For MVP, we treat it as HTML extraction. 
        # Ideally, we would detect if it's a /edit URL and try to convert to /export?format=pdf or docx
        # but that requires auth or public access.
        # If public, /export?format=txt or html might be better.
        
        try:
            return self.html_extractor.extract(document_uri)
        except ExtractionError as e:
            raise ExtractionError(f"Failed to extract Google Doc: {str(e)}")

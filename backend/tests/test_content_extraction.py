import pytest
from pathlib import Path
from backend.app.adapters.content.pdf import PDFExtractor
from backend.app.adapters.content.markdown import MarkdownExtractor
from backend.app.adapters.content.html import HTMLExtractor
from backend.app.config.schema import AppConfig, WebFetchConfig, MetadataBackend, LexicalBackend, VectorBackend, StorageConfig, IngestionConfig, BookmarksConfig, EmbeddingConfig

@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def mock_config(tmp_path):
    return AppConfig(
        metadata_backend=MetadataBackend.SQLITE,
        lexical_backend=LexicalBackend.FTS5,
        vector_backend=VectorBackend.FAISS,
        storage=StorageConfig(data_dir=tmp_path, sqlite_path=tmp_path/"db", faiss_dir=tmp_path),
        ingestion=IngestionConfig(chunk_size_tokens=100, chunk_overlap_tokens=0, max_file_mb=10),
        bookmarks=BookmarksConfig(),
        web_fetch=WebFetchConfig(enabled=True),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=10)
    )

def test_pdf_extraction(fixtures_dir):
    extractor = PDFExtractor()
    pdf_path = fixtures_dir / "sample.pdf"
    
    result = extractor.extract(str(pdf_path))
    
    assert "Hello PDF World" in result.text
    assert result.mime_type == "application/pdf"
    assert result.extra["page_count"] == 1

def test_markdown_extraction(fixtures_dir):
    extractor = MarkdownExtractor()
    md_path = fixtures_dir / "sample.md"
    
    result = extractor.extract(str(md_path))
    
    assert "Sample Markdown" == result.title
    assert "This is a sample markdown file." in result.text
    assert "Heading 1" in result.text
    # Frontmatter should be stripped from text
    assert "---" not in result.text

def test_html_extraction(fixtures_dir, mock_config):
    extractor = HTMLExtractor(mock_config)
    html_path = fixtures_dir / "sample.html"
    
    result = extractor.extract(f"file://{html_path}")
    
    assert "Sample HTML" == result.title
    assert "Heading 1" in result.text
    assert "sample html page" in result.text
    # Scripts should be removed
    assert "console.log" not in result.text

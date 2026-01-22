import pytest
from pathlib import Path
from backend.app.services.ingestion import IngestionService
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.domain import models

@pytest.fixture
def ingestion_service(tmp_path):
    db_path = tmp_path / "metadata.db"
    config = AppConfig(
        metadata_backend=MetadataBackend.SQLITE,
        lexical_backend=LexicalBackend.FTS5,
        vector_backend=VectorBackend.FAISS,
        storage=StorageConfig(data_dir=tmp_path, sqlite_path=db_path, faiss_dir=tmp_path),
        ingestion=IngestionConfig(chunk_size_tokens=100, chunk_overlap_tokens=0, max_file_mb=10),
        bookmarks=BookmarksConfig(),
        web_fetch=WebFetchConfig(),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=10)
    )
    metadata = SQLiteMetadataStore(config)
    return IngestionService(metadata)

def test_bookmarks_ingestion(ingestion_service, tmp_path):
    fixtures_dir = Path(__file__).parent / "fixtures"
    bookmarks_path = fixtures_dir / "bookmarks.json"
    
    if not bookmarks_path.exists():
        pytest.skip("Bookmarks fixture not found")
        
    source = models.Source(name="bookmarks", path=str(bookmarks_path))
    
    candidates = list(ingestion_service.scan_bookmarks(source))
    
    assert len(candidates) == 2
    
    titles = {d.title for d in candidates}
    urls = {d.uri for d in candidates}
    
    assert "Google" in titles
    assert "Example" in titles
    assert "https://www.google.com/" in urls
    assert "https://example.com/" in urls
    
    # Check document properties
    doc = candidates[0]
    assert doc.mime_type == "text/html"
    assert doc.status == "scanned"

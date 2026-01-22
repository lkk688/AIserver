import pytest
import os
from pathlib import Path
from uuid import uuid4
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.adapters.lexical.fts5 import FTS5LexicalIndex
from backend.app.adapters.vector.faiss import FAISSVectorStore
from backend.app.adapters.embedding.litellm import LiteLLMEmbeddingProvider
from backend.app.services.indexing import IndexingService
from backend.app.domain import models

# Mock Embedding Provider to avoid calling real API
class MockEmbeddingProvider(LiteLLMEmbeddingProvider):
    def __init__(self):
        self.dim = 4
        
    def embed_texts(self, texts):
        # Return dummy vectors
        return [[0.1] * 4 for _ in texts]

@pytest.fixture
def test_pipeline(tmp_path):
    # Setup config
    db_path = tmp_path / "metadata.db"
    faiss_dir = tmp_path / "faiss_idx"
    config = AppConfig(
        metadata_backend=MetadataBackend.SQLITE,
        lexical_backend=LexicalBackend.FTS5,
        vector_backend=VectorBackend.FAISS,
        storage=StorageConfig(data_dir=tmp_path, sqlite_path=db_path, faiss_dir=faiss_dir),
        ingestion=IngestionConfig(chunk_size_tokens=100, chunk_overlap_tokens=0, max_file_mb=10),
        bookmarks=BookmarksConfig(),
        web_fetch=WebFetchConfig(enabled=True),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=4)
    )
    
    metadata = SQLiteMetadataStore(config)
    lexical = FTS5LexicalIndex(config)
    vector = FAISSVectorStore(config)
    embedding = MockEmbeddingProvider()
    
    service = IndexingService(config, metadata, lexical, vector, embedding)
    
    return service, metadata, lexical, vector

def test_end_to_end_indexing(test_pipeline, tmp_path):
    service, metadata, lexical, vector = test_pipeline
    
    # 1. Create a source pointing to fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    # Ensure fixtures exist (they were created in previous step)
    if not (fixtures_dir / "sample.md").exists():
        pytest.skip("Fixtures not found")
        
    source = models.Source(name="fixtures", path=str(fixtures_dir))
    saved_source = metadata.upsert_source(source)
    
    # 2. Run Scan
    job = service.scan_source(saved_source.id)
    
    assert job.status == models.JobStatus.DONE
    
    # 3. Verify Documents
    docs = metadata.list_documents_by_source(saved_source.id)
    assert len(docs) >= 3 # pdf, md, html
    
    # Find markdown doc
    md_doc = next((d for d in docs if d.uri.endswith("sample.md")), None)
    assert md_doc is not None
    assert md_doc.status == "indexed"
    assert "Sample Markdown" in (md_doc.title or "")
    
    # 4. Verify Chunks
    chunks = metadata.list_chunks(md_doc.id)
    assert len(chunks) > 0
    assert "Heading 1" in chunks[0].text
    
    # 5. Verify FTS
    results = lexical.search("markdown", top_k=5)
    assert len(results) > 0
    # Check if one of the results belongs to our doc
    chunk_ids = {c.id for c in chunks}
    found = any(res[0] in chunk_ids for res in results)
    assert found
    
    # 6. Verify Vector
    # Query with dummy vector
    vec_results = vector.query([0.1]*4, top_k=5)
    assert len(vec_results) > 0
    found_vec = any(res[0] in chunk_ids for res in vec_results)
    assert found_vec

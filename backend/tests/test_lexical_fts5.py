import pytest
import os
from uuid import uuid4
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.adapters.lexical.fts5 import FTS5LexicalIndex
from backend.app.domain import models

@pytest.fixture
def test_env(tmp_path):
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
    lexical = FTS5LexicalIndex(config)
    return metadata, lexical

def test_fts5_indexing_and_search(test_env):
    metadata, lexical = test_env
    
    # Setup data
    source = models.Source(name="src", path="/tmp")
    metadata.upsert_source(source)
    
    doc1 = models.Document(source_id=source.id, uri="doc1", title="Python Guide")
    metadata.upsert_document(doc1)
    
    chunk1 = models.Chunk(
        doc_id=doc1.id, 
        chunk_index=0, 
        text="Python is a programming language", 
        start_offset=0, 
        end_offset=30, 
        chunk_hash="c1"
    )
    metadata.upsert_chunk(chunk1)
    
    doc2 = models.Document(source_id=source.id, uri="doc2", title="Rust Guide")
    metadata.upsert_document(doc2)
    
    chunk2 = models.Chunk(
        doc_id=doc2.id, 
        chunk_index=0, 
        text="Rust is a systems programming language", 
        start_offset=0, 
        end_offset=35, 
        chunk_hash="c2"
    )
    metadata.upsert_chunk(chunk2)
    
    # Index chunks
    lexical.upsert_chunks([chunk1, chunk2])
    
    # Search "Python"
    results = lexical.search("Python", top_k=10)
    assert len(results) == 1
    assert results[0][0] == chunk1.id
    
    # Search "programming" (should hit both)
    results = lexical.search("programming", top_k=10)
    assert len(results) == 2
    ids = {r[0] for r in results}
    assert chunk1.id in ids
    assert chunk2.id in ids
    
    # Test delete doc
    lexical.delete_doc(doc1.id)
    results = lexical.search("Python", top_k=10)
    assert len(results) == 0
    
    # doc2 should still be there
    results = lexical.search("Rust", top_k=10)
    assert len(results) == 1

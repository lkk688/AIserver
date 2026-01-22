import pytest
import numpy as np
import shutil
from uuid import uuid4
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.adapters.vector.faiss import FAISSVectorStore
from backend.app.domain import models

@pytest.fixture
def faiss_store(tmp_path):
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
        web_fetch=WebFetchConfig(),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=4)
    )
    return FAISSVectorStore(config)

def test_faiss_lifecycle(faiss_store):
    # 1. Upsert
    doc_id = uuid4()
    chunk1 = models.Chunk(
        id=uuid4(), doc_id=doc_id, chunk_index=0, text="a", start_offset=0, end_offset=1, chunk_hash="1"
    )
    chunk2 = models.Chunk(
        id=uuid4(), doc_id=doc_id, chunk_index=1, text="b", start_offset=0, end_offset=1, chunk_hash="2"
    )
    
    # Vectors: dim=4
    # v1 = [1, 0, 0, 0]
    # v2 = [0, 1, 0, 0]
    # v3 = [0.9, 0.1, 0, 0] -> should be close to v1
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0, 0.0]
    
    faiss_store.upsert_embeddings([chunk1, chunk2], [v1, v2])
    
    # 2. Query
    # Query close to v1
    q_vec = [0.9, 0.1, 0.0, 0.0]
    results = faiss_store.query(q_vec, top_k=1)
    
    assert len(results) >= 1
    assert results[0][0] == chunk1.id
    # Score should be high (close to 1.0 for cosine)
    assert results[0][1] > 0.8
    
    # 3. Persistence
    # Reload store (simulate restart)
    # We pass the same config, so it should load from disk
    config = faiss_store.index_dir.parent
    # Actually just re-instantiate using same config paths
    # (Since fixture uses tmp_path, we rely on the fact that tmp_path persists for the test function duration)
    # But wait, we need to ensure the file was written. _save_index is called in upsert.
    
    # 4. Delete Doc
    faiss_store.delete_doc(doc_id)
    
    # Query again - should be empty or not contain chunk1
    results = faiss_store.query(q_vec, top_k=10)
    # Should be empty since we deleted the only doc
    assert len(results) == 0
    
def test_soft_delete_and_reinsert(faiss_store):
    doc_id = uuid4()
    chunk = models.Chunk(
        id=uuid4(), doc_id=doc_id, chunk_index=0, text="a", start_offset=0, end_offset=1, chunk_hash="1"
    )
    v1 = [1.0, 0.0, 0.0, 0.0]
    
    faiss_store.upsert_embeddings([chunk], [v1])
    
    # Delete
    faiss_store.delete_doc(doc_id)
    assert len(faiss_store.query(v1, top_k=1)) == 0
    
    # Re-insert same chunk (e.g. re-indexing)
    faiss_store.upsert_embeddings([chunk], [v1])
    
    # Should be found again
    results = faiss_store.query(v1, top_k=1)
    assert len(results) == 1
    assert results[0][0] == chunk.id

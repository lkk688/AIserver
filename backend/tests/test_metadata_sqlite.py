import pytest
import os
from uuid import uuid4
from datetime import datetime
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.domain import models

@pytest.fixture
def sqlite_store(tmp_path):
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
    return SQLiteMetadataStore(config)

def test_sqlite_metadata_lifecycle(sqlite_store):
    # 1. Create Source
    source = models.Source(name="test_source", path="/tmp/test")
    saved_source = sqlite_store.upsert_source(source)
    assert saved_source.id == source.id
    assert saved_source.name == "test_source"
    
    # 2. Upsert Document
    doc = models.Document(
        source_id=source.id,
        uri="file:///tmp/test/doc1.txt",
        title="Doc 1",
        status="new"
    )
    saved_doc = sqlite_store.upsert_document(doc)
    assert saved_doc.id == doc.id
    assert saved_doc.source_id == source.id
    
    # 3. Insert Chunks
    chunk1 = models.Chunk(
        doc_id=doc.id,
        chunk_index=0,
        text="Hello world",
        start_offset=0,
        end_offset=11,
        chunk_hash="h1"
    )
    saved_chunk = sqlite_store.upsert_chunk(chunk1)
    assert saved_chunk.id == chunk1.id
    
    # 4. Create Job
    job = models.Job(type=models.JobType.SCAN_SOURCE)
    saved_job = sqlite_store.upsert_job(job)
    assert saved_job.status == models.JobStatus.PENDING
    
    # Update Job
    saved_job.status = models.JobStatus.RUNNING
    saved_job.progress = 0.5
    updated_job = sqlite_store.upsert_job(saved_job)
    assert updated_job.status == models.JobStatus.RUNNING
    
    # 5. List/Get
    docs = sqlite_store.list_documents_by_source(source.id)
    assert len(docs) == 1
    assert docs[0].id == doc.id
    
    chunks = sqlite_store.list_chunks(doc.id)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"
    
    fetched_job = sqlite_store.get_job(job.id)
    assert fetched_job.progress == 0.5

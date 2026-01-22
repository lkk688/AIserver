import pytest
import time
from uuid import uuid4
from pathlib import Path
from backend.app.domain import models
from backend.app.services.jobs import JobRunner
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.config.schema import AppConfig, MetadataBackend, LexicalBackend, VectorBackend, StorageConfig, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig

@pytest.fixture
def test_config(tmp_path):
    db_path = tmp_path / "metadata.db"
    return AppConfig(
        metadata_backend=MetadataBackend.SQLITE,
        lexical_backend=LexicalBackend.FTS5,
        vector_backend=VectorBackend.FAISS,
        storage=StorageConfig(data_dir=tmp_path, sqlite_path=db_path, faiss_dir=tmp_path),
        ingestion=IngestionConfig(chunk_size_tokens=100, chunk_overlap_tokens=0, max_file_mb=10),
        bookmarks=BookmarksConfig(),
        web_fetch=WebFetchConfig(),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=10)
    )

@pytest.fixture
def metadata_store(test_config):
    return SQLiteMetadataStore(test_config)

@pytest.fixture
def mock_indexing_service(metadata_store):
    class MockIndexingService:
        def __init__(self, store):
            self.store = store
            self.scanned_sources = []
            
        def scan_source(self, source_id, job=None):
            self.scanned_sources.append((source_id, job))
            if job:
                job.status = models.JobStatus.DONE
                job.progress = 1.0
                self.store.upsert_job(job)
                
    return MockIndexingService(metadata_store)

@pytest.fixture
def job_runner(metadata_store, mock_indexing_service):
    return JobRunner(metadata_store, mock_indexing_service)

def test_job_runner_lifecycle(job_runner, metadata_store, mock_indexing_service):
    # 1. Enqueue Job
    source_id = uuid4()
    job = job_runner.enqueue_job(
        type=models.JobType.SCAN_SOURCE,
        payload={"source_id": str(source_id)}
    )
    
    assert job.status == models.JobStatus.PENDING
    assert job.payload["source_id"] == str(source_id)
    
    # Verify in DB
    pending = metadata_store.get_pending_jobs()
    assert len(pending) == 1
    assert pending[0].id == job.id
    
    # 2. Run Loop (Manual invocation for deterministic testing)
    # We don't start the thread, we just call _worker_loop logic once
    
    # To test single loop iteration, we can't easily call _worker_loop since it loops.
    # But we can call _process_job directly if we fetch it, OR modify _worker_loop to support running once.
    # Alternatively, start thread, wait, stop.
    
    job_runner.start()
    
    # Wait for job to be processed
    max_retries = 10
    for _ in range(max_retries):
        updated_job = metadata_store.get_job(job.id)
        if updated_job.status == models.JobStatus.DONE:
            break
        time.sleep(0.1)
        
    job_runner.stop()
    
    updated_job = metadata_store.get_job(job.id)
    assert updated_job.status == models.JobStatus.DONE
    assert len(mock_indexing_service.scanned_sources) == 1
    assert mock_indexing_service.scanned_sources[0][0] == source_id

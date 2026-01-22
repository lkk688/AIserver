from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.dependencies import get_config
from backend.app.config.schema import AppConfig, MetadataBackend, LexicalBackend, VectorBackend, StorageConfig, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
import pytest
from pathlib import Path

@pytest.fixture
def test_client(tmp_path, monkeypatch):
    # Reset job runner to ensure it uses the new config/DB
    from backend.app.dependencies import reset_job_runner
    reset_job_runner()
    
    # Override config to use tmp_path
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
    
    # Monkeypatch get_config in dependencies to return our test config
    # This ensures manual calls (like in lifespan) get the test config
    import backend.app.dependencies
    monkeypatch.setattr(backend.app.dependencies, "get_config", lambda: config)
    
    # Also keep dependency_overrides for routes just in case
    app.dependency_overrides[get_config] = lambda: config
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()
    # Ensure runner is stopped and cleared after test too
    reset_job_runner()

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_source_lifecycle(test_client):
    # 1. Create Source
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    response = test_client.post("/api/v1/sources", json={
        "name": "api_test_source",
        "path": str(fixtures_dir),
        "config": {}
    })
    assert response.status_code == 200, response.text
    source = response.json()
    source_id = source["id"]
    
    # 2. List Sources
    response = test_client.get("/api/v1/sources")
    assert response.status_code == 200
    assert len(response.json()) > 0
    
    # 3. Scan Source
    # Call the endpoint to enqueue the job
    response = test_client.post(f"/api/v1/sources/{source_id}/scan")
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "pending"
    assert job["type"] == "scan_source"
    assert job["payload"]["source_id"] == source_id

    # Note: We don't wait for completion here because the JobRunner thread 
    # might process it asynchronously, but TestClient context might close 
    # before it finishes or we'd need to sleep.
    # The unit test test_jobs.py covers the runner logic.

@pytest.fixture
def mock_embedding_provider(monkeypatch):
    from backend.app.adapters.embedding.litellm import LiteLLMEmbeddingProvider
    def mock_embed(self, texts):
        # Return 10-dim vectors to match test config
        return [[0.1] * 10 for _ in texts]
    monkeypatch.setattr(LiteLLMEmbeddingProvider, "embed_texts", mock_embed)

def test_search_endpoint(test_client, mock_embedding_provider):
    # Just check if endpoint is up, result might be empty
    response = test_client.post("/api/v1/search", json={"query": "test", "top_k": 5})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

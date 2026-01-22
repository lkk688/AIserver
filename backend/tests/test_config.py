import pytest
from pathlib import Path
import yaml
import os
from pydantic import ValidationError
from backend.app.config.loader import load_config
from backend.app.config.schema import AppConfig

@pytest.fixture
def valid_config_data():
    return {
        "metadata_backend": "sqlite",
        "lexical_backend": "fts5",
        "vector_backend": "faiss",
        "storage": {
            "data_dir": "./data",
            "sqlite_path": "./data/metadata.db",
            "faiss_dir": "./data/faiss_index"
        },
        "ingestion": {
            "chunk_size_tokens": 512,
            "chunk_overlap_tokens": 50,
            "max_file_mb": 10
        },
        "bookmarks": {
            "chrome_bookmarks_path": "/tmp/bookmarks"
        },
        "web_fetch": {
            "enabled": True,
            "timeout_sec": 10,
            "user_agent": "TestAgent"
        },
        "embedding": {
            "provider": "test",
            "model_name": "test-model",
            "dim": 128
        }
    }

def test_load_valid_config(tmp_path, valid_config_data):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
    
    config = load_config(str(config_file))
    assert isinstance(config, AppConfig)
    assert config.metadata_backend == "sqlite"
    # Note: path normalization happens, so we check if it resolves to an absolute path
    assert config.storage.data_dir.is_absolute()

def test_missing_required_field(tmp_path, valid_config_data):
    del valid_config_data["metadata_backend"]
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
    
    with pytest.raises(ValueError, match="validation failed"):
        load_config(str(config_file))

def test_invalid_enum(tmp_path, valid_config_data):
    valid_config_data["metadata_backend"] = "invalid_backend"
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
        
    with pytest.raises(ValueError, match="validation failed"):
        load_config(str(config_file))

def test_env_override(tmp_path, valid_config_data):
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
    
    os.environ["APP_METADATA_BACKEND"] = "postgres"
    os.environ["APP_INGESTION_CHUNK_SIZE_TOKENS"] = "1024"
    
    try:
        config = load_config(str(config_file))
        assert config.metadata_backend == "postgres"
        assert config.ingestion.chunk_size_tokens == 1024
    finally:
        del os.environ["APP_METADATA_BACKEND"]
        del os.environ["APP_INGESTION_CHUNK_SIZE_TOKENS"]

def test_path_normalization(tmp_path, valid_config_data):
    # Use a mock home dir path for testing normalization
    valid_config_data["storage"]["data_dir"] = "/tmp/test_data"
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(valid_config_data, f)
        
    config = load_config(str(config_file))
    expected = Path("/tmp/test_data").resolve()
    assert config.storage.data_dir == expected

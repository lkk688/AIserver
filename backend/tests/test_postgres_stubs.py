import pytest
from backend.app.config.schema import AppConfig, StorageConfig, MetadataBackend, LexicalBackend, VectorBackend, IngestionConfig, BookmarksConfig, WebFetchConfig, EmbeddingConfig
from backend.app.adapters.metadata.postgres import PostgresMetadataStore
from backend.app.adapters.vector.pgvector import PgVectorStore
from backend.app.adapters.lexical.pg_fts import PgFTSIndex

@pytest.fixture
def mock_config(tmp_path):
    return AppConfig(
        metadata_backend=MetadataBackend.POSTGRES,
        lexical_backend=LexicalBackend.PG_FTS,
        vector_backend=VectorBackend.PGVECTOR,
        storage=StorageConfig(data_dir=tmp_path, sqlite_path=tmp_path/"db", faiss_dir=tmp_path),
        ingestion=IngestionConfig(chunk_size_tokens=100, chunk_overlap_tokens=0, max_file_mb=10),
        bookmarks=BookmarksConfig(),
        web_fetch=WebFetchConfig(),
        embedding=EmbeddingConfig(provider="test", model_name="test", dim=10)
    )

def test_postgres_metadata_stub(mock_config):
    store = PostgresMetadataStore(mock_config)
    with pytest.raises(NotImplementedError):
        store.list_sources()

def test_pgvector_stub(mock_config):
    store = PgVectorStore(mock_config)
    with pytest.raises(NotImplementedError):
        store.query([0.1]*10, 5)

def test_pgfts_stub(mock_config):
    index = PgFTSIndex(mock_config)
    with pytest.raises(NotImplementedError):
        index.search("test", 5)

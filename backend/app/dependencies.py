from functools import lru_cache
from typing import Annotated, Optional
from fastapi import Depends
from backend.app.config.loader import load_config
from backend.app.config.schema import AppConfig, MetadataBackend, LexicalBackend, VectorBackend
from backend.app.domain.ports import MetadataStore, LexicalIndex, VectorStore, EmbeddingProvider
from backend.app.adapters.metadata.sqlite import SQLiteMetadataStore
from backend.app.adapters.metadata.postgres import PostgresMetadataStore
from backend.app.adapters.lexical.fts5 import FTS5LexicalIndex
from backend.app.adapters.lexical.pg_fts import PgFTSIndex
from backend.app.adapters.vector.faiss import FAISSVectorStore
from backend.app.adapters.vector.pgvector import PgVectorStore
from backend.app.adapters.embedding.litellm import LiteLLMEmbeddingProvider
from backend.app.services.indexing import IndexingService
from backend.app.services.search import SearchService
from backend.app.services.jobs import JobRunner
import os

@lru_cache
def get_config() -> AppConfig:
    config_path = os.getenv("APP_CONFIG_PATH", "backend/config.yaml")
    return load_config(config_path)

def get_metadata_store(config: Annotated[AppConfig, Depends(get_config)] = None) -> MetadataStore:
    if config is None: config = get_config()
    if config.metadata_backend == MetadataBackend.SQLITE:
        return SQLiteMetadataStore(config)
    elif config.metadata_backend == MetadataBackend.POSTGRES:
        return PostgresMetadataStore(config)
    raise ValueError(f"Unknown metadata backend: {config.metadata_backend}")

def get_lexical_index(config: Annotated[AppConfig, Depends(get_config)] = None) -> LexicalIndex:
    if config is None: config = get_config()
    if config.lexical_backend == LexicalBackend.FTS5:
        return FTS5LexicalIndex(config)
    elif config.lexical_backend == LexicalBackend.PG_FTS:
        return PgFTSIndex(config)
    raise ValueError(f"Unknown lexical backend: {config.lexical_backend}")

def get_vector_store(config: Annotated[AppConfig, Depends(get_config)] = None) -> VectorStore:
    if config is None: config = get_config()
    if config.vector_backend == VectorBackend.FAISS:
        return FAISSVectorStore(config)
    elif config.vector_backend == VectorBackend.PGVECTOR:
        return PgVectorStore(config)
    raise ValueError(f"Unknown vector backend: {config.vector_backend}")

def get_embedding_provider(config: Annotated[AppConfig, Depends(get_config)] = None) -> EmbeddingProvider:
    if config is None: config = get_config()
    return LiteLLMEmbeddingProvider(config)

def get_indexing_service(
    config: Annotated[AppConfig, Depends(get_config)] = None,
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)] = None,
    lexical_index: Annotated[LexicalIndex, Depends(get_lexical_index)] = None,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)] = None,
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)] = None
) -> IndexingService:
    if config is None: config = get_config()
    if metadata_store is None: metadata_store = get_metadata_store(config)
    if lexical_index is None: lexical_index = get_lexical_index(config)
    if vector_store is None: vector_store = get_vector_store(config)
    if embedding_provider is None: embedding_provider = get_embedding_provider(config)
    
    return IndexingService(
        config=config,
        metadata_store=metadata_store,
        lexical_index=lexical_index,
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )

def get_search_service(
    config: Annotated[AppConfig, Depends(get_config)] = None,
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)] = None,
    lexical_index: Annotated[LexicalIndex, Depends(get_lexical_index)] = None,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)] = None,
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)] = None
) -> SearchService:
    if config is None: config = get_config()
    if metadata_store is None: metadata_store = get_metadata_store(config)
    if lexical_index is None: lexical_index = get_lexical_index(config)
    if vector_store is None: vector_store = get_vector_store(config)
    if embedding_provider is None: embedding_provider = get_embedding_provider(config)

    return SearchService(
        config=config,
        metadata_store=metadata_store,
        lexical_index=lexical_index,
        vector_store=vector_store,
        embedding_provider=embedding_provider
    )

_job_runner_instance: Optional[JobRunner] = None

def get_job_runner(
    metadata_store: Annotated[MetadataStore, Depends(get_metadata_store)] = None,
    indexing_service: Annotated[IndexingService, Depends(get_indexing_service)] = None
) -> JobRunner:
    # Singleton-ish for the runner, though dependencies are per-request usually.
    # But JobRunner must be long-lived.
    # In FastAPI, we can use app.state or a global variable.
    # Ideally, we initialize it in lifespan and store it.
    # For dependency injection here, we can return the global instance if set, or create one.
    # Since dependencies (metadata_store) might be scoped, we need to be careful.
    # However, our stores are effectively singletons (connection pools).
    
    global _job_runner_instance
    if _job_runner_instance is None:
        # If calling from test or script without lifespan
        if metadata_store is None: metadata_store = get_metadata_store()
        if indexing_service is None: indexing_service = get_indexing_service()
        _job_runner_instance = JobRunner(metadata_store, indexing_service)
    
    return _job_runner_instance

def reset_job_runner():
    global _job_runner_instance
    if _job_runner_instance:
        _job_runner_instance.stop()
    _job_runner_instance = None

# Local Search RAG - Backend Implementation Summary

This document summarizes the implementation of the backend for the Local Search RAG system, covering the bootstrap, domain modeling, and adapter implementations (Steps 1-6).

## 1. Project Structure

The project follows a **Clean Architecture** (Ports & Adapters) pattern to ensure separation of concerns and modularity.

```
backend/
├── app/
│   ├── config/                 # Configuration management
│   │   ├── loader.py           # Config loader with env overrides
│   │   └── schema.py           # Pydantic configuration models
│   ├── domain/                 # Core domain logic (inner layer)
│   │   ├── models.py           # Domain entities (Source, Document, Chunk, Job)
│   │   ├── ports.py            # Interfaces/Ports (ABCs) for adapters
│   │   └── errors.py           # Domain exceptions
│   ├── adapters/               # Infrastructure implementations (outer layer)
│   │   ├── metadata/           # Metadata store adapters
│   │   │   ├── sqlite.py       # SQLite implementation (SQLAlchemy)
│   │   │   └── postgres.py     # Postgres stub
│   │   ├── lexical/            # Lexical search adapters
│   │   │   ├── fts5.py         # SQLite FTS5 implementation
│   │   │   └── pg_fts.py       # Postgres FTS stub
│   │   └── vector/             # Vector store adapters
│   │       ├── faiss.py        # FAISS implementation (local disk)
│   │       └── pgvector.py     # Postgres pgvector stub
│   │   ├── content/            # Content extraction adapters
│   │   │   ├── pdf.py          # PDF extractor (pypdf)
│   │   │   ├── markdown.py     # Markdown extractor (frontmatter regex)
│   │   │   ├── html.py         # HTML extractor (BeautifulSoup)
│   │   │   └── gdoc.py         # Google Doc wrapper
│   │   └── embedding/          # Embedding adapters
│   │       └── litellm.py      # LiteLLM embedding provider with caching
│   ├── services/               # Application services (orchestration)
│   │   ├── ingestion.py        # File scanning service
│   │   └── indexing.py         # Main indexing pipeline service
│   ├── util/                   # Shared utilities
│   │   ├── chunking.py         # Token-aware chunking (tiktoken)
│   │   └── hashing.py          # Stable hashing (SHA-256)
│   └── main.py                 # FastAPI application entry point
├── tests/                      # Unit and integration tests
│   ├── test_config.py          # Config loading tests
│   ├── test_domain.py          # Domain model tests
│   ├── test_metadata_sqlite.py # SQLite metadata adapter tests
│   ├── test_lexical_fts5.py    # SQLite FTS5 adapter tests
│   ├── test_vector_faiss.py    # FAISS vector adapter tests
│   ├── test_postgres_stubs.py  # Postgres stub tests
│   ├── test_content_extraction.py # Content extraction tests
│   ├── test_chunking.py        # Chunking utility tests
│   └── test_indexing_service.py # End-to-end indexing pipeline tests
├── tests/fixtures/             # Test fixture files
│   ├── sample.pdf
│   ├── sample.md
│   └── sample.html
└── config.yaml.example         # Example configuration file
```

## 2. Implementation Steps Summary

### Step 1: Backend Bootstrap
- **Goal**: Set up FastAPI app and configuration management.
- **Tech**: FastAPI, Pydantic v2, PyYAML.
- **Key Components**:
  - `backend/app/main.py`: App initialization.
  - `backend/app/config/schema.py`: Typed config definition.
  - `backend/app/config/loader.py`: Loads YAML and overrides with environment variables.

### Step 2: Domain Models & Ports
- **Goal**: Define the core business logic and interfaces.
- **Key Components**:
  - `models.py`: Pydantic models for `Source`, `Document` (with metadata), `Chunk` (text segments), and `Job` (async tasks).
  - `ports.py`: Abstract Base Classes (ABCs) for `MetadataStore`, `LexicalIndex`, `VectorStore`, `ContentExtractor`, and `EmbeddingProvider`.

### Step 3: SQLite Metadata Store
- **Goal**: Implement persistent metadata storage.
- **Tech**: SQLAlchemy 2.0, SQLite.
- **Implementation**:
  - Maps domain models to SQLAlchemy ORM models (`SourceORM`, `DocumentORM`, etc.).
  - Handles CRUD operations and schema creation via `Base.metadata.create_all()`.

### Step 4: Lexical Index (SQLite FTS5)
- **Goal**: Implement keyword search.
- **Tech**: SQLite FTS5 extension.
- **Implementation**:
  - Uses a virtual table `chunks_fts` within the same SQLite database as metadata.
  - Supports BM25 ranking via `bm25()` function.
  - Handles upserts by deleting old entries for a chunk before inserting.

### Step 5: Vector Store (FAISS)
- **Goal**: Implement semantic search.
- **Tech**: `faiss-cpu`, `numpy`.
- **Implementation**:
  - Uses `faiss.IndexIDMap` + `IndexFlatIP` (Inner Product) for vector storage with persistent IDs.
  - Maintains a sidecar SQLite table `chunk_vectors` to map FAISS internal IDs (int64) to Domain Chunk IDs (UUID) and handle soft deletions.
  - Persists index to disk (`index.faiss`).

### Step 6: Postgres Adapters (Stubs)
- **Goal**: Prepare for future Postgres support.
- **Implementation**:
  - Created stub classes for `PostgresMetadataStore`, `PgVectorStore`, and `PgFTSIndex`.
  - These classes raise `NotImplementedError` but validate that the architecture supports swapping backends via config.

### Step 7: Content Extraction
- **Goal**: Parse and extract text from various file formats.
- **Tech**: `pypdf`, `BeautifulSoup4`, `markdown`, `regex`.
- **Implementation**:
  - Implemented adapters for PDF, Markdown, HTML, and Google Docs.
  - **ExtractedContent**: Added new domain model to standardize output.

### Step 8: Chunking Utilities
- **Goal**: Token-aware text splitting.
- **Tech**: `tiktoken` (cl100k_base).
- **Implementation**:
  - `chunk_text`: Splits text into chunks with overlap, handling fallback to char-based splitting if needed.
  - `hashing`: Stable SHA-256 hashing for text content.

### Step 9: Embedding Provider
- **Goal**: Interface with external embedding API.
- **Tech**: `httpx` (client), `json` (local cache).
- **Implementation**:
  - `LiteLLMEmbeddingProvider`: Calls local OpenAI-compatible embedding service.
  - **Caching**: JSON file-based caching keyed by hash(text + model).

### Step 10: Service Orchestration
- **Goal**: Coordinate ingestion, indexing, and search workflows.
- **Implementation**:
  - `IndexingService`: Orchestrates scanning, content extraction, chunking, and updating metadata/vector/lexical stores.
  - `SearchService`: Orchestrates hybrid search (keyword + vector) and reranking (future).

### Step 11: Bookmarks Ingestion
- **Goal**: Ingest URLs from browser bookmarks (JSON export).
- **Implementation**:
  - `BookmarksConfig`: Configuration for bookmarks file path and tag filtering.
  - `IngestionService`: Extended to parse Chrome-style bookmarks JSON structure.
  - Added integration tests for bookmarks parsing.

### Step 13: REST API
- **Goal**: Expose functionality via HTTP API.
- **Tech**: FastAPI APIRouter, Dependency Injection.
- **Implementation**:
  - **Endpoints & Curl Examples**:
    > Note: All API endpoints are prefixed with `/api/v1` except `/health`.

    1.  **Health Check**
        - `GET /health`: Check server status.
        ```bash
        curl -X GET http://localhost:8000/health
        ```

    2.  **Sources**
        - `POST /api/v1/sources`: Register a new data source (local folder).
        ```bash
        curl -X POST http://localhost:8000/api/v1/sources \
             -H "Content-Type: application/json" \
             -d '{"name": "Sample Docs", "path": "/absolute/path/to/docs", "config": {}}'
        ```
        - `GET /api/v1/sources`: List all registered sources.
        ```bash
        curl -X GET http://localhost:8000/api/v1/sources
        ```
        - `POST /api/v1/sources/{id}/scan`: Trigger an async background scan job for a source.
        ```bash
        # Replace {source_id} with the UUID returned from GET /sources
        curl -X POST http://localhost:8000/api/v1/sources/{source_id}/scan
        ```

    3.  **Jobs**
        - `GET /api/v1/jobs`: List all background jobs (scans).
        ```bash
        curl -X GET http://localhost:8000/api/v1/jobs
        ```
        - `GET /api/v1/jobs/{id}`: Get status of a specific job.
        ```bash
        curl -X GET http://localhost:8000/api/v1/jobs/{job_id}
        ```

    4.  **Documents**
        - `GET /api/v1/documents`: List documents (required `source_id` query param).
        ```bash
        curl -X GET "http://localhost:8000/api/v1/documents?source_id={source_id}"
        ```
        - `GET /api/v1/documents/{id}`: Get metadata for a specific document.
        ```bash
        curl -X GET http://localhost:8000/api/v1/documents/{doc_id}
        ```
        - `GET /api/v1/documents/{id}/chunks`: Get the text chunks for a document.
        ```bash
        curl -X GET http://localhost:8000/api/v1/documents/{doc_id}/chunks
        ```

    5.  **Search**
        - `POST /api/v1/search`: Perform hybrid search (Keyword + Vector).
        ```bash
        curl -X POST http://localhost:8000/api/v1/search \
             -H "Content-Type: application/json" \
             -d '{"query": "machine learning", "top_k": 5}'
        ```

  - **Architecture**:
    - Uses `Depends` for DI of services and stores based on config.
    - Pydantic models for stable Request/Response schemas.
    - Comprehensive `TestClient` integration tests.

### Step 14: Background Job Runner
- **Goal**: Asynchronous job execution for long-running tasks (scanning).
- **Implementation**:
  - `JobRunner`: Thread-based worker loop that polls `pending` jobs from MetadataStore.
  - **Wiring**: Integrated into FastAPI `lifespan` to start/stop automatically.
  - **Endpoints**: `POST /sources/{id}/scan` now enqueues a job instead of blocking.
  - **Tests**: `test_jobs.py` verifies job lifecycle (enqueue -> pending -> running -> done).

## 3. Test Scripts

The project includes a comprehensive test suite using `pytest`.

| Test File | Description | Command to Run |
|-----------|-------------|----------------|
| `tests/test_config.py` | Verifies config loading, validation, and env overrides. | `pytest backend/tests/test_config.py` |
| `tests/test_domain.py` | Verifies domain model instantiation and constraints. | `pytest backend/tests/test_domain.py` |
| `tests/test_metadata_sqlite.py` | Integration test for SQLite metadata CRUD lifecycle. | `pytest backend/tests/test_metadata_sqlite.py` |
| `tests/test_lexical_fts5.py` | Integration test for FTS5 indexing, search, and ranking. | `pytest backend/tests/test_lexical_fts5.py` |
| `tests/test_vector_faiss.py` | Integration test for FAISS vector upsert, query, and persistence. | `pytest backend/tests/test_vector_faiss.py` |
| `tests/test_postgres_stubs.py` | Verifies that Postgres stubs exist and raise correct errors. | `pytest backend/tests/test_postgres_stubs.py` |
| `tests/test_content_extraction.py` | Verifies text/metadata extraction from PDF, MD, HTML. | `pytest backend/tests/test_content_extraction.py` |
| `tests/test_chunking.py` | Verifies chunking logic and hash stability. | `pytest backend/tests/test_chunking.py` |
| `tests/test_api.py` | Integration tests for REST API endpoints. | `pytest backend/tests/test_api.py` |
| `tests/test_jobs.py` | Integration tests for background job runner. | `pytest backend/tests/test_jobs.py` |

### Running All Tests
To run all tests at once:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m pytest -v backend/tests
```

## 4. How to Run the App

1.  **Install Dependencies**:
    ```bash
    pip install fastapi uvicorn pydantic pyyaml pytest httpx sqlalchemy faiss-cpu numpy
    ```

2.  **Configuration**:
    Copy the example config:
    ```bash
    cp backend/config.yaml.example backend/config.yaml
    ```

3.  **Start Server**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    uvicorn backend.app.main:app --port 8000 --reload
    ```
    Health check: http://localhost:8000/health

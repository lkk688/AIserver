from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from uuid import UUID
from backend.app.domain.models import Source, Document, Chunk, Job, ExtractedContent

class MetadataStore(ABC):
    @abstractmethod
    def upsert_source(self, source: Source) -> Source: ...
    
    @abstractmethod
    def get_source(self, source_id: UUID) -> Optional[Source]: ...
    
    @abstractmethod
    def list_sources(self) -> List[Source]: ...

    @abstractmethod
    def upsert_document(self, doc: Document) -> Document: ...
    
    @abstractmethod
    def get_document(self, doc_id: UUID) -> Optional[Document]: ...
    
    @abstractmethod
    def list_documents_by_source(self, source_id: UUID) -> List[Document]: ...
    
    @abstractmethod
    def mark_document_deleted(self, doc_id: UUID) -> None: ...

    @abstractmethod
    def upsert_chunk(self, chunk: Chunk) -> Chunk: ...
    
    @abstractmethod
    def list_chunks(self, doc_id: UUID) -> List[Chunk]: ...

    @abstractmethod
    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]: ...

    @abstractmethod
    def upsert_job(self, job: Job) -> Job: ...
    
    @abstractmethod
    def get_job(self, job_id: UUID) -> Optional[Job]: ...

    @abstractmethod
    def list_jobs(self) -> List[Job]: ...

    @abstractmethod
    def get_pending_jobs(self, limit: int = 10) -> List[Job]: ...

class LexicalIndex(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]) -> None: ...
    
    @abstractmethod
    def delete_doc(self, doc_id: UUID) -> None: ...
    
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[UUID, float]]:
        """Returns list of (doc_id or chunk_id, score)"""
        ...

class VectorStore(ABC):
    @abstractmethod
    def upsert_embeddings(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None: ...
    
    @abstractmethod
    def delete_doc(self, doc_id: UUID) -> None: ...
    
    @abstractmethod
    def query(self, vector: List[float], top_k: int) -> List[Tuple[UUID, float]]:
        """Returns list of (chunk_id, score)"""
        ...

class ContentExtractor(ABC):
    @abstractmethod
    def extract(self, document_uri: str) -> Tuple[str, dict]:
        """Returns (extracted_text, metadata_dict)"""
        ...

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Tuple[Chunk, float]]:
        """Returns chunks sorted by relevance score"""
        ...

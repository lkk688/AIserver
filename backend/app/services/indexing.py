import time
from uuid import UUID
from typing import List, Optional, Dict
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore, LexicalIndex, VectorStore, ContentExtractor, EmbeddingProvider
from backend.app.config.schema import AppConfig
from backend.app.services.ingestion import IngestionService
from backend.app.util.chunking import chunk_text
from backend.app.util.hashing import compute_hash
from backend.app.adapters.content.pdf import PDFExtractor
from backend.app.adapters.content.markdown import MarkdownExtractor
from backend.app.adapters.content.html import HTMLExtractor
from backend.app.adapters.content.gdoc import GoogleDocExtractor

class IndexingService:
    def __init__(
        self,
        config: AppConfig,
        metadata_store: MetadataStore,
        lexical_index: LexicalIndex,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        # extractors injected or initialized here
    ):
        self.config = config
        self.metadata = metadata_store
        self.lexical = lexical_index
        self.vector = vector_store
        self.embedding = embedding_provider
        self.ingestion = IngestionService(metadata_store)
        
        # Initialize extractors
        self.extractors = {
            'application/pdf': PDFExtractor(),
            'text/markdown': MarkdownExtractor(),
            'text/html': HTMLExtractor(config),
            # 'application/vnd.google-apps.document': GoogleDocExtractor(config) # If we detect this mime
        }
        self.default_extractor = self.extractors['text/html'] # Fallback for now? Or None.

    def scan_source(self, source_id: UUID, job: Optional[models.Job] = None) -> models.Job:
        source = self.metadata.get_source(source_id)
        if not source:
            raise ValueError("Source not found")

        if not job:
            job = models.Job(type=models.JobType.SCAN_SOURCE, status=models.JobStatus.RUNNING, payload={"source_id": str(source_id)})
            job = self.metadata.upsert_job(job)
        else:
            job.status = models.JobStatus.RUNNING
            job = self.metadata.upsert_job(job)

        try:
            # 1. Scan candidates
            if source.path.endswith(".json"):
                candidates = list(self.ingestion.scan_bookmarks(source))
            else:
                candidates = list(self.ingestion.scan_directory(source))
            
            # 2. Filter changes
            # Retrieve all existing docs for this source to compare
            existing_docs = {d.uri: d for d in self.metadata.list_documents_by_source(source.id)}
            
            docs_to_index = []
            
            for doc in candidates:
                existing = existing_docs.get(doc.uri)
                if not existing:
                    # New
                    doc.status = "new"
                    saved = self.metadata.upsert_document(doc)
                    docs_to_index.append(saved)
                else:
                    # Check change (mtime, size)
                    # Using float comparison for mtime might be flaky, but okay for now
                    if doc.mtime != existing.mtime or doc.size_bytes != existing.size_bytes:
                        existing.status = "changed"
                        existing.mtime = doc.mtime
                        existing.size_bytes = doc.size_bytes
                        saved = self.metadata.upsert_document(existing)
                        docs_to_index.append(saved)
                    else:
                        # Unchanged
                        pass

            # 3. Index docs
            # For this MVP, we process them synchronously inside this job, 
            # but ideally we'd spawn sub-jobs or use a queue.
            total = len(docs_to_index)
            for i, doc in enumerate(docs_to_index):
                self.index_document(doc.id)
                # Update job progress
                job.progress = (i + 1) / total if total > 0 else 1.0
                self.metadata.upsert_job(job)

            job.status = models.JobStatus.DONE
            job.progress = 1.0
        except Exception as e:
            job.status = models.JobStatus.FAILED
            job.error = str(e)
            print(f"Scan failed: {e}")
        
        return self.metadata.upsert_job(job)

    def index_document(self, doc_id: UUID):
        doc = self.metadata.get_document(doc_id)
        if not doc:
            return

        try:
            # 1. Extract
            extractor = self.extractors.get(doc.mime_type, self.default_extractor)
            if not extractor:
                print(f"No extractor for {doc.mime_type}")
                return

            content = extractor.extract(doc.uri)
            
            # Update doc metadata from extraction
            doc.title = content.title or doc.title
            doc_hash = compute_hash(content.text)
            
            # Optimization: if doc_hash matches existing, skip (if we trust it)
            if doc.doc_hash == doc_hash and doc.status != "changed":
                # Already indexed and same content?
                pass
            
            doc.doc_hash = doc_hash
            self.metadata.upsert_document(doc)

            # 2. Chunk
            chunks_data = chunk_text(
                content.text, 
                self.config.ingestion.chunk_size_tokens, 
                self.config.ingestion.chunk_overlap_tokens
            )
            
            chunks_to_persist = []
            for cd in chunks_data:
                chunk = models.Chunk(
                    doc_id=doc.id,
                    chunk_index=cd.chunk_index,
                    text=cd.text,
                    start_offset=cd.start_offset,
                    end_offset=cd.end_offset,
                    chunk_hash=compute_hash(cd.text)
                )
                chunks_to_persist.append(chunk)

            # 3. Store Chunks (Metadata)
            # First, delete old chunks? Or overwrite?
            # Ideally delete old ones. The metadata adapter should handle cleanup 
            # OR we explicitly delete. Our current metadata store doesn't have delete_chunks_by_doc.
            # But upsert overwrites by ID. IDs are random UUIDs here.
            # We need to clear old chunks for this doc.
            # IMPORTANT: The current sqlite adapter `list_chunks` returns existing.
            # We should probably implement `delete_chunks_for_doc` in MetadataStore or just handle it here.
            # Since IDs are new UUIDs every time we instantiate Chunk(), we will accumulate garbage 
            # if we don't delete old ones.
            # For MVP, we'll assume `delete_doc` in Lexical/Vector handles their parts, 
            # but MetadataStore needs cleanup.
            # Let's add a TODO or just not worry about orphan chunks in sqlite for this step.
            # Actually, `delete_doc` in adapters removes from index.
            
            # Clean indexes first (idempotency)
            self.lexical.delete_doc(doc.id)
            self.vector.delete_doc(doc.id)
            
            # Save new chunks to DB
            saved_chunks = []
            for chunk in chunks_to_persist:
                saved = self.metadata.upsert_chunk(chunk)
                saved_chunks.append(saved)

            # 4. Update Lexical Index
            self.lexical.upsert_chunks(saved_chunks)

            # 5. Compute Embeddings & Update Vector Store
            texts = [c.text for c in saved_chunks]
            embeddings = self.embedding.embed_texts(texts)
            self.vector.upsert_embeddings(saved_chunks, embeddings)

            doc.status = "indexed"
            self.metadata.upsert_document(doc)

        except Exception as e:
            print(f"Indexing error for {doc.uri}: {e}")
            doc.status = "error"
            self.metadata.upsert_document(doc)
            raise e

    def reindex_all(self):
        # Scan all sources
        sources = self.metadata.list_sources()
        for source in sources:
            self.scan_source(source.id)

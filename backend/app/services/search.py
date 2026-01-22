from typing import List, Dict, Optional, Tuple
from uuid import UUID
from pydantic import BaseModel
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore, LexicalIndex, VectorStore, EmbeddingProvider, Reranker
from backend.app.config.schema import AppConfig

class SearchResult(BaseModel):
    chunk_id: UUID
    doc_id: UUID
    text: str
    doc_title: Optional[str]
    doc_uri: str
    score: float
    score_breakdown: Dict[str, float]

class NoOpReranker(Reranker):
    def rerank(self, query: str, chunks: List[models.Chunk]) -> List[Tuple[models.Chunk, float]]:
        # Return as is with score 0.0 (placeholder)
        return [(c, 0.0) for c in chunks]

class SearchService:
    def __init__(
        self,
        config: AppConfig,
        metadata_store: MetadataStore,
        lexical_index: LexicalIndex,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[Reranker] = None
    ):
        self.config = config
        self.metadata = metadata_store
        self.lexical = lexical_index
        self.vector = vector_store
        self.embedding = embedding_provider
        self.reranker = reranker or NoOpReranker()
        
        # Tuning parameters (could be in config)
        self.top_k_lex = 20
        self.top_k_vec = 20
        self.rrf_k = 60 # Constant for RRF

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        # 1. Lexical Search
        lex_results = self.lexical.search(query, top_k=self.top_k_lex)
        
        # 2. Vector Search
        # Embed query (single text)
        query_vectors = self.embedding.embed_texts([query])
        if query_vectors:
            vec_results = self.vector.query(query_vectors[0], top_k=self.top_k_vec)
        else:
            vec_results = []

        # 3. Fuse Results (RRF)
        # chunk_id -> {lex_rank, vec_rank, combined_score}
        
        scores: Dict[UUID, float] = {}
        breakdown: Dict[UUID, Dict[str, float]] = {}
        
        # Process Lexical
        for rank, (chunk_id, score) in enumerate(lex_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            
            bd = breakdown.setdefault(chunk_id, {"lex_score": 0.0, "vec_score": 0.0})
            bd["lex_score"] = score
            bd["lex_rank"] = rank + 1

        # Process Vector
        for rank, (chunk_id, score) in enumerate(vec_results):
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            
            bd = breakdown.setdefault(chunk_id, {"lex_score": 0.0, "vec_score": 0.0})
            bd["vec_score"] = score
            bd["vec_rank"] = rank + 1

        # Sort by combined score desc
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        hydrate_limit = limit * 2 # Hydrate a bit more for reranking if implemented
        candidates_ids = sorted_ids[:hydrate_limit]
        
        # 4. Hydrate
        candidates: List[models.Chunk] = []
        for cid in candidates_ids:
            chunk = self.metadata.get_chunk(cid)
            if chunk:
                candidates.append(chunk)
        
        # 5. Rerank
        # Reranker takes list of chunks and returns sorted list with scores
        # We only rerank hydrated candidates
        reranked = self.reranker.rerank(query, candidates)
        
        # 6. Format Results
        results = []
        # Cache document metadata to avoid repeated fetches
        doc_cache: Dict[UUID, models.Document] = {}
        
        # Take top limit from reranked
        # If reranker is no-op, we should trust RRF order.
        # But reranker returns list of (chunk, score). 
        # If no-op returns 0.0 score, we lose RRF info if we just take that.
        # Ideally Reranker should combine or replace RRF score.
        # For NoOp, let's just stick to the order candidates_ids provided (RRF order)
        # unless reranker provided meaningful scores.
        
        # If using NoOpReranker, `reranked` has same order as `candidates`.
        
        for chunk, rerank_score in reranked[:limit]:
            doc_id = chunk.doc_id
            if doc_id not in doc_cache:
                doc = self.metadata.get_document(doc_id)
                if doc:
                    doc_cache[doc_id] = doc
            
            doc = doc_cache.get(doc_id)
            if not doc:
                continue # Orphaned chunk?
                
            final_score = scores[chunk.id] # RRF score
            # If reranker was active, we might use rerank_score
            
            results.append(SearchResult(
                chunk_id=chunk.id,
                doc_id=doc_id,
                text=chunk.text,
                doc_title=doc.title,
                doc_uri=doc.uri,
                score=final_score,
                score_breakdown=breakdown.get(chunk.id, {})
            ))
            
        return results

import os
import faiss
import numpy as np
import sqlite3
from typing import List, Tuple, Optional
from uuid import UUID
from pathlib import Path
from backend.app.domain import models
from backend.app.domain.ports import VectorStore
from backend.app.config.schema import AppConfig

class FAISSVectorStore(VectorStore):
    def __init__(self, config: AppConfig):
        self.dim = config.embedding.dim
        self.index_dir = config.storage.faiss_dir
        self.index_path = self.index_dir / "index.faiss"
        self.db_path = config.storage.sqlite_path
        
        # Ensure directories exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            # Inner Product (cosine similarity if normalized)
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
            
        # Initialize Mapping DB (using main sqlite DB but raw connection)
        self._init_mapping_db()

    def _init_mapping_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    faiss_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    deleted BOOLEAN DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_vectors_chunk_id ON chunk_vectors(chunk_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_vectors_doc_id ON chunk_vectors(doc_id)")
            conn.commit()

    def _save_index(self):
        faiss.write_index(self.index, str(self.index_path))

    def upsert_embeddings(self, chunks: List[models.Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        # Prepare data
        vectors = np.array(embeddings, dtype='float32')
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        # We need persistent IDs for FAISS. 
        # Strategy: Insert into DB -> get auto-increment IDs -> use those for FAISS.
        
        new_ids = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for chunk in chunks:
                # 1. Mark old entries for this chunk as deleted (soft update)
                cursor.execute(
                    "UPDATE chunk_vectors SET deleted = 1 WHERE chunk_id = ?", 
                    (str(chunk.id),)
                )
                
                # 2. Insert new entry
                cursor.execute(
                    "INSERT INTO chunk_vectors (chunk_id, doc_id, deleted) VALUES (?, ?, 0)",
                    (str(chunk.id), str(chunk.doc_id))
                )
                new_ids.append(cursor.lastrowid)
            conn.commit()
            
        # Add to FAISS
        # IndexIDMap requires IDs to be int64
        ids_array = np.array(new_ids, dtype='int64')
        self.index.add_with_ids(vectors, ids_array)
        
        # Persist
        self._save_index()

    def delete_doc(self, doc_id: UUID) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE chunk_vectors SET deleted = 1 WHERE doc_id = ?",
                (str(doc_id),)
            )
            conn.commit()

    def query(self, vector: List[float], top_k: int) -> List[Tuple[UUID, float]]:
        # Normalize query vector
        q_vec = np.array([vector], dtype='float32')
        faiss.normalize_L2(q_vec)
        
        # Search. We ask for more results to handle filtered items.
        # Simple heuristic: top_k * 3. If many deleted, might need loop.
        fetch_k = top_k * 5
        scores, ids = self.index.search(q_vec, fetch_k)
        
        # ids[0] is the array of neighbors for the first (and only) query vector
        neighbor_ids = ids[0]
        neighbor_scores = scores[0]
        
        # Filter deleted items via DB
        # We need to map faiss_id -> chunk_id
        valid_results = []
        
        # Batch query for efficiency
        # Filter out -1 (FAISS returns -1 if not enough neighbors)
        candidate_ids = [int(nid) for nid in neighbor_ids if nid != -1]
        
        if not candidate_ids:
            return []

        with sqlite3.connect(self.db_path) as conn:
            # We want to preserve order, so we fetch mapping and re-order in python
            placeholders = ",".join("?" * len(candidate_ids))
            rows = conn.execute(
                f"SELECT faiss_id, chunk_id FROM chunk_vectors WHERE faiss_id IN ({placeholders}) AND deleted = 0",
                candidate_ids
            ).fetchall()
            
            # Map faiss_id -> chunk_id
            id_map = {row[0]: row[1] for row in rows}
            
            for nid, score in zip(neighbor_ids, neighbor_scores):
                if nid != -1 and nid in id_map:
                    valid_results.append((UUID(id_map[nid]), float(score)))
                    if len(valid_results) >= top_k:
                        break
                        
        return valid_results

    def compact(self):
        """
        Optional: Rebuild index to remove deleted vectors.
        This is expensive and should be run periodically or manually.
        """
        # 1. Fetch all active vectors/ids
        # This requires reading vectors back from FAISS if we don't store them elsewhere.
        # FAISS IndexIDMap + IndexFlat stores vectors.
        # We can reconstruct.
        # For simplicity in this iteration: Not implemented fully. 
        # Full implementation would: create new index, iterate active IDs, reconstruct vectors, add to new index.
        pass

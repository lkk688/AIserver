from typing import List, Tuple
from uuid import UUID
from sqlalchemy import text, create_engine
from backend.app.domain import models
from backend.app.domain.ports import LexicalIndex
from backend.app.config.schema import AppConfig

class FTS5LexicalIndex(LexicalIndex):
    def __init__(self, config: AppConfig):
        # We use the same SQLite database as metadata for simplicity
        # If we wanted separate, we'd use a different path
        self.db_path = config.storage.sqlite_path
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)
        
        self._init_schema()

    def _init_schema(self):
        # Create FTS5 virtual table
        # Columns: chunk_id (UNINDEXED), doc_id (UNINDEXED), title, text, uri
        # Note: chunk_id and doc_id are stored but not indexed for full-text search themselves
        # FTS5 syntax: column [options]
        # We need to execute raw SQL
        
        schema_sql = """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED, 
            doc_id UNINDEXED, 
            title, 
            text, 
            uri
        );
        """
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()

    def upsert_chunks(self, chunks: List[models.Chunk]) -> None:
        if not chunks:
            return
            
        with self.engine.connect() as conn:
            # For each chunk, we need metadata (title, uri) which is on the Document.
            # However, the port signature only gives us chunks. 
            # This implies the Domain Service calling this should ideally pass enriched data 
            # or we need to fetch it.
            # BUT, to keep it simple and efficient, maybe we just index what we have 
            # or rely on the caller to provide complete info? 
            # The requirement says: "chunk_id, doc_id, title (optional duplicated), text, uri/path"
            # Since `models.Chunk` doesn't have title/uri, we can't index them easily 
            # unless we query the metadata store OR the caller passes them.
            # 
            # Given the constraints, let's assume for now we only index text, 
            # OR we fetch document info if needed. Fetching doc info for every chunk insert is slow.
            # Let's assume the calling service handles this, OR we assume empty title/uri for now 
            # if we can't change the port signature.
            # 
            # Wait, the prompt requirements for FTS5 say: "title (optional duplicated from document), text, uri/path".
            # The port signature is `upsert_chunks(chunks: List[Chunk])`.
            # This is a design mismatch in the prompt's deliverables vs requirements.
            # I will modify the implementation to fetch document metadata if possible, 
            # OR just insert placeholders.
            # BETTER: We can do a JOIN in the search query if we used external content tables, 
            # but we are using a separate FTS table.
            # 
            # Realistically, we'll implement a separate lookup or just index text for now.
            # Actually, let's check if we can query the `documents` table since we share the DB.
            # Yes, we can!
            
            for chunk in chunks:
                # 1. Delete existing entry for this chunk_id
                conn.execute(text("DELETE FROM chunks_fts WHERE chunk_id = :cid"), {"cid": str(chunk.id)})
                
                # 2. Get doc metadata (title, uri)
                # We can do this in a batch or one-by-one. One-by-one is easiest for prototype.
                doc_res = conn.execute(
                    text("SELECT title, uri FROM documents WHERE id = :did"), 
                    {"did": str(chunk.doc_id)}
                ).fetchone()
                
                title = doc_res[0] if doc_res else ""
                uri = doc_res[1] if doc_res else ""
                
                # 3. Insert
                conn.execute(
                    text("""
                        INSERT INTO chunks_fts (chunk_id, doc_id, title, text, uri) 
                        VALUES (:cid, :did, :title, :text, :uri)
                    """),
                    {
                        "cid": str(chunk.id),
                        "did": str(chunk.doc_id),
                        "title": title or "",
                        "text": chunk.text,
                        "uri": uri or ""
                    }
                )
            conn.commit()

    def delete_doc(self, doc_id: UUID) -> None:
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM chunks_fts WHERE doc_id = :did"),
                {"did": str(doc_id)}
            )
            conn.commit()

    def search(self, query: str, top_k: int) -> List[Tuple[UUID, float]]:
        # Use bm25 ranking
        # FTS5 has a built-in bm25() function.
        # Query: SELECT chunk_id, bm25(chunks_fts) as score FROM chunks_fts WHERE chunks_fts MATCH :query ORDER BY score LIMIT :limit
        # Note: bm25 returns lower is better (more negative usually). FTS5 documentation says "The lower the value, the more relevant".
        # So we order by score ASC? No, wait. 
        # "bm25() returns a value that is less than or equal to 0.0... A more negative value indicates a better match."
        # So ORDER BY bm25(chunks_fts) ASC gives best matches first.
        
        # We need to handle special characters in query to prevent syntax errors.
        # Simple sanitization: remove non-alphanumeric? Or use phrase query?
        # Let's wrap in quotes if it contains spaces, or let user handle it.
        # For robustness, let's just pass it raw but handle exceptions.
        
        sql = """
            SELECT chunk_id, bm25(chunks_fts) as rank 
            FROM chunks_fts 
            WHERE chunks_fts MATCH :query 
            ORDER BY rank 
            LIMIT :limit
        """
        
        results = []
        with self.engine.connect() as conn:
            try:
                rows = conn.execute(text(sql), {"query": query, "limit": top_k}).fetchall()
                for row in rows:
                    # Score is negative (lower is better). We might want to invert it for "higher is better"
                    # or just return as is. The interface implies (id, score). 
                    # Let's return abs(rank) or -rank so higher is better? 
                    # Usually score implies higher is better. Let's return -rank.
                    results.append((UUID(row[0]), -1 * row[1]))
            except Exception as e:
                # FTS5 syntax error likely
                print(f"Search error: {e}")
                return []
                
        return results

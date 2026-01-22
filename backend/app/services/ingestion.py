import os
import json
from pathlib import Path
from typing import List, Generator
from uuid import UUID
from datetime import datetime
from backend.app.domain import models
from backend.app.domain.ports import MetadataStore
from backend.app.util.hashing import compute_hash

class IngestionService:
    def __init__(self, metadata_store: MetadataStore):
        self.metadata = metadata_store

    def scan_directory(self, source: models.Source) -> Generator[models.Document, None, None]:
        """
        Scans a local directory source and yields Document candidates.
        Only yields files that are new or changed.
        """
        path = Path(source.path)
        if not path.exists():
            return

        for root, _, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                
                # Basic filtering
                if file.startswith('.'):
                    continue
                    
                try:
                    stats = file_path.stat()
                    uri = f"file://{file_path.absolute()}"
                    
                    doc = models.Document(
                        source_id=source.id,
                        uri=uri,
                        title=file, # Default title
                        mime_type=self._guess_mime(file),
                        size_bytes=stats.st_size,
                        mtime=datetime.fromtimestamp(stats.st_mtime),
                        status="scanned"
                    )
                    yield doc
                except Exception as e:
                    print(f"Error scanning file {file_path}: {e}")

    def scan_bookmarks(self, source: models.Source) -> Generator[models.Document, None, None]:
        """
        Scans a Chrome/Edge-style bookmarks JSON file and yields Document candidates.
        """
        path = Path(source.path)
        if not path.exists():
            print(f"Bookmarks file not found: {path}")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Chrome bookmarks structure: { "roots": { "bookmark_bar": { "children": [...] }, ... } }
            roots = data.get("roots", {})
            for root_key in roots:
                root_node = roots[root_key]
                yield from self._traverse_bookmarks(root_node, source.id)
                
        except Exception as e:
            print(f"Error parsing bookmarks {path}: {e}")

    def _traverse_bookmarks(self, node: dict, source_id: UUID) -> Generator[models.Document, None, None]:
        node_type = node.get("type")
        
        if node_type == "url":
            url = node.get("url")
            if url and (url.startswith("http://") or url.startswith("https://")):
                yield models.Document(
                    source_id=source_id,
                    uri=url,
                    title=node.get("name") or url,
                    mime_type="text/html", # Treat as web page
                    status="scanned",
                    mtime=datetime.now() # Current time as we don't have update time for bookmark usually
                )
        elif node_type == "folder" or "children" in node:
            children = node.get("children", [])
            for child in children:
                yield from self._traverse_bookmarks(child, source_id)

    def _guess_mime(self, filename: str) -> str:
        ext = filename.lower().split('.')[-1] if '.' in filename else ""
        if ext in ('pdf'): return 'application/pdf'
        if ext in ('md', 'markdown'): return 'text/markdown'
        if ext in ('html', 'htm'): return 'text/html'
        if ext in ('txt'): return 'text/plain'
        return 'application/octet-stream'

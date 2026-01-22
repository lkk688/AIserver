from typing import List
from uuid import UUID
import httpx
import json
import os
from backend.app.domain.ports import EmbeddingProvider
from backend.app.config.schema import AppConfig
from backend.app.util.hashing import compute_hash

class LiteLLMEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: AppConfig):
        self.api_base = "http://localhost:8005/v1" # Default, could be config
        # Ideally config.embedding.provider could specify the URL or we derive it
        # For now hardcode to our local service or use a specific env var
        self.api_url = f"{self.api_base}/embeddings"
        self.model_name = config.embedding.model_name
        self.dim = config.embedding.dim
        self.cache_dir = config.storage.data_dir / "cache" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Check cache first
        vectors = [None] * len(texts)
        texts_to_fetch = []
        indices_to_fetch = []

        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                vectors[i] = cached
            else:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)

        if texts_to_fetch:
            # Batch fetch from API
            try:
                fetched_vectors = self._fetch_embeddings(texts_to_fetch)
                for idx, vec in zip(indices_to_fetch, fetched_vectors):
                    vectors[idx] = vec
                    self._save_to_cache(texts[idx], vec)
            except Exception as e:
                # If API fails, we might want to retry or raise
                raise RuntimeError(f"Embedding API call failed: {e}")

        return vectors # type: ignore

    def _get_cache_key(self, text: str) -> str:
        # Cache key: hash(text + model_name)
        return compute_hash(text + self.model_name)

    def _get_from_cache(self, text: str) -> List[float] | None:
        key = self._get_cache_key(text)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_to_cache(self, text: str, vector: List[float]):
        key = self._get_cache_key(text)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(vector, f)

    def _fetch_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Call the OpenAI compatible embedding endpoint
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        # We use a context manager for the client to ensure cleanup
        # Note: synchronous for now as per interface, could be async in future
        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Sort by index to ensure order matches input
            # OpenAI format: data: [{object: embedding, embedding: [...], index: 0}, ...]
            results = sorted(data['data'], key=lambda x: x['index'])
            return [item['embedding'] for item in results]

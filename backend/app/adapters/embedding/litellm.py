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
        base_url = (
            config.embedding.base_url
            or os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8005/v1")
        ).rstrip("/")
        self.api_url = f"{base_url}/embeddings"
        self.api_key = config.embedding.api_key or os.environ.get("EMBEDDING_API_KEY", "")
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
            try:
                fetched_vectors = self._fetch_embeddings_batched(texts_to_fetch)
                for idx, vec in zip(indices_to_fetch, fetched_vectors):
                    vectors[idx] = vec
                    self._save_to_cache(texts[idx], vec)
            except Exception as e:
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

    _BATCH_SIZE = 16  # max texts per embedding API request

    def _fetch_embeddings_batched(self, texts: List[str]) -> List[List[float]]:
        """Send texts to the embedding API in fixed-size batches."""
        all_vectors: List[List[float]] = []
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=120.0) as client:
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                payload = {"model": self.model_name, "input": batch}
                response = client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                results = sorted(data["data"], key=lambda x: x["index"])
                all_vectors.extend(item["embedding"] for item in results)
        return all_vectors

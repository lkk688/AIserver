from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.api_url.rstrip("/"),
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,  # our callers already handle failures; built-in retries just add latency
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.config.api_model,
            input=texts,
        )

        ordered = sorted(response.data, key=lambda x: x.index)
        vectors = [item.embedding for item in ordered]

        logger.debug("Embedded %d texts using model=%s", len(texts), self.config.api_model)
        return vectors
import os
from sentence_transformers import SentenceTransformer
from typing import List, Union

class EmbeddingModel:
    def __init__(self):
        self.model = None
        # Default to google/embeddinggemma-300m if not specified
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")

    def load(self):
        print(f"Loading embedding model: {self.model_name}...")
        try:
            # trust_remote_code=True is often needed for newer architectures
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print(f"Embedding model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading embedding model {self.model_name}: {e}")
            raise e

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if isinstance(texts, str):
            texts = [texts]
            
        # sentence-transformers encode returns numpy array by default
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

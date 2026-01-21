import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Models to compare
MODELS = [
    "google/embeddinggemma-300m",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/embeddinggemma-300m-medical"
]

# Test sentences
SENTENCES = [
    "What are the symptoms of influenza?",
    "Flu is characterized by fever, cough, and fatigue.",
    "The stock market crashed today.",
    "Apples are red or green fruits."
]

def compare_models():
    print(f"Comparing Embedding Models on {len(SENTENCES)} sentences...")
    print("-" * 50)
    
    for model_name in MODELS:
        print(f"\nLoading model: {model_name}")
        try:
            start_time = time.time()
            model = SentenceTransformer(model_name, trust_remote_code=True)
            load_time = time.time() - start_time
            print(f"Loaded in {load_time:.2f}s")
            
            # Encode
            start_time = time.time()
            embeddings = model.encode(SENTENCES)
            encode_time = time.time() - start_time
            print(f"Encoded {len(SENTENCES)} sentences in {encode_time:.4f}s")
            print(f"Embedding Shape: {embeddings.shape}")
            
            # Compute similarity between first two sentences (Flu related)
            sim_flu = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # Compute similarity between first and third (Flu vs Stock)
            sim_diff = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
            
            print(f"Similarity (Flu Q vs Flu A): {sim_flu:.4f}")
            print(f"Similarity (Flu Q vs Stock): {sim_diff:.4f}")
            
        except Exception as e:
            print(f"Failed to load/run {model_name}: {e}")

if __name__ == "__main__":
    compare_models()

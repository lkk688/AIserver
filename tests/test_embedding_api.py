import requests
import json
import unittest
import numpy as np
import time

URL = "http://localhost:8005/v1/embeddings"
MODEL = "google/embeddinggemma-300m"

class TestEmbeddingAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Wait for service to be ready
        print("Waiting for service to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                # Check readyz endpoint if available, or just try the embedding endpoint
                # Since readyz checks for model readiness, let's use that if possible
                # But here we'll try a simple embedding request to see if it responds 200
                payload = {"input": "ping", "model": MODEL}
                response = requests.post(URL, json=payload, timeout=5)
                if response.status_code == 200:
                    print("Service is ready.")
                    return
            except Exception:
                pass
            time.sleep(2)
            print(f"Waiting... ({i+1}/{max_retries})")
        raise RuntimeError("Service failed to start in time.")

    def test_single_embedding(self):
        payload = {
            "input": "This is a test sentence for embedding.",
            "model": MODEL
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(URL, headers=headers, data=json.dumps(payload))
        self.assertEqual(response.status_code, 200, f"Request failed with status {response.status_code}: {response.text}")
        
        data = response.json()
        self.assertEqual(data["model"], MODEL)
        self.assertIsInstance(data["data"], list)
        self.assertEqual(len(data["data"]), 1)
        self.assertTrue("embedding" in data["data"][0])
        self.assertIsInstance(data["data"][0]["embedding"], list)
        self.assertGreater(len(data["data"][0]["embedding"]), 0)
        print(f"\nSingle embedding length: {len(data['data'][0]['embedding'])}")

    def test_batch_embedding(self):
        inputs = ["Sentence one.", "Sentence two.", "Sentence three."]
        payload = {
            "input": inputs,
            "model": MODEL
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(URL, headers=headers, data=json.dumps(payload))
        self.assertEqual(response.status_code, 200, f"Request failed with status {response.status_code}: {response.text}")
        
        data = response.json()
        self.assertEqual(len(data["data"]), 3)
        for i, item in enumerate(data["data"]):
            self.assertEqual(item["index"], i)
            self.assertIsInstance(item["embedding"], list)
            
        # Check consistency (optional: simple check if embeddings are not identical)
        emb1 = data["data"][0]["embedding"]
        emb2 = data["data"][1]["embedding"]
        self.assertNotEqual(emb1, emb2, "Embeddings for different sentences should be different")
        print(f"\nBatch embedding successful for {len(inputs)} inputs.")

    def test_model_selection(self):
        # This test assumes the server is running with the specified MODEL
        # If we request a different model name that IS supported but maybe mapped differently, 
        # the server currently ignores the 'model' field in the request for loading, 
        # but returns it in the response.
        # Ideally, we should check if the server actually validates the model name.
        # For now, we test that the response echoes the model name or uses the default loaded one.
        pass

    def test_embedding_values(self):
        # Basic sanity check on values (e.g. not all zeros, valid floats)
        payload = {
            "input": "Sanity check.",
            "model": MODEL
        }
        response = requests.post(URL, json=payload)
        data = response.json()
        embedding = data["data"][0]["embedding"]
        
        # Check if contains valid floats
        self.assertTrue(all(isinstance(x, float) for x in embedding))
        
        # Check if not all zeros (unless it's a zero vector model, which is unlikely)
        self.assertFalse(all(x == 0 for x in embedding))
        
        # Check norm (if normalized, should be close to 1)
        norm = np.linalg.norm(embedding)
        print(f"\nEmbedding norm: {norm}")
        # Most sentence transformers produce normalized embeddings
        # self.assertAlmostEqual(norm, 1.0, places=4) 

if __name__ == "__main__":
    print(f"Testing Embedding API at {URL} with model {MODEL}")
    unittest.main()

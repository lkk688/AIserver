# ONNX Runtime Embeddings + Rerank Service (OpenAI-Compatible Wrapper)

This version adds:
- request micro-batching for `/v1/embeddings`
- API key auth
- `/v1/rerank` for a reranker model such as a Qwen reranker
- separate embedding and reranker model mounts
- pooling presets for Qwen3-Embedding and BGE-M3

## Layout

- `app.py` — FastAPI service with OpenAI-style `/v1/embeddings` and a simple `/v1/rerank`
- `export_embedding_model.py` — exports a sentence-transformers embedding model to ONNX
- `export_reranker_model.py` — exports a sequence-classification reranker model to ONNX
- `docker-compose.yml` — example deployment with both models mounted

## 1) Export the embedding model

Example for BGE-M3:

```bash
python3 export_embedding_model.py \
  --model-id BAAI/bge-m3 \
  --output-dir ./models/bge-m3 \
  --pooling cls \
  --normalize \
  --max-length 8192
```

Example for Qwen embedding:

```bash
python3 export_embedding_model.py \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --output-dir ./models/qwen3-embedding-0.6b \
  --pooling last_token \
  --normalize \
  --max-length 32768
```

## 2) Export the reranker model

Example for a Qwen reranker:

```bash
python3 export_reranker_model.py \
  --model-id Qwen/Qwen3-Reranker-0.6B \
  --output-dir ./models/qwen3-reranker-0.6b \
  --max-length 512
```

If your reranker model card recommends score normalization, add `--normalize-scores`.

## 3) Build the container

```bash
docker build -t onnx-embeddings-rerank-p100:latest .
```

## 4) Run the container
In the background:
```bash
docker run -d \
  --name onnx-embed-service \
  --gpus all \
  -p 8002:8000 \
  -e EMBED_MODEL_DIR=/models/embedding \
  -e EMBED_SERVICE_CONFIG=/models/embedding/service_config.json \
  -e RERANK_MODEL_DIR=/models/reranker \
  -e RERANK_SERVICE_CONFIG=/models/reranker/service_config.json \
  -e API_KEYS=embeddingp100 \
  -e ORT_GPU_MEM_LIMIT_GB=14 \
  -e EMBED_MAX_BATCH_TEXTS=32 \
  -e EMBED_BATCH_TIMEOUT_MS=8 \
  -e RERANK_MAX_BATCH_PAIRS=32 \
  -e RERANK_BATCH_TIMEOUT_MS=8 \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-embedding-0.6b:/models/embedding:ro \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-reranker-0.6b:/models/reranker:ro \
  onnx-embeddings-rerank-p100:latest

docker logs -f onnx-embed-service
docker stop onnx-embed-service
docker start onnx-embed-service
docker restart onnx-embed-service
docker rm -f onnx-embed-service

curl http://localhost:8002/healthz

curl http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer embeddingp100" \
  -d '{"input":"hello world","model":"Qwen/Qwen3-Embedding-0.6B"}'

docker rm -f onnx-embed-service
docker build -t onnx-embeddings-rerank-p100:latest .

curl http://localhost:8002/v1/rerank \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer embeddingp100" \
  -d '{
    "model":"Qwen/Qwen3-Reranker-0.6B",
    "query":"what is deep learning",
    "documents":[
      "Deep learning is a subset of machine learning based on neural networks.",
      "Paris is the capital of France.",
      "A GPU can accelerate matrix multiplication."
    ]
  }'

docker exec -it onnx-embed-service python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

docker exec -it onnx-embed-service python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.__file__)"

docker exec -it onnx-embed-service bash

python3 -m pip uninstall -y onnxruntime onnxruntime-gpu
python3 -m pip install --no-cache-dir onnxruntime-gpu==1.20.1
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

docker rm -f onnx-embed-service
docker build --no-cache -t onnx-embeddings-rerank-p100:latest .
docker logs -f onnx-embed-service

curl http://localhost:8002/healthz
docker exec -it onnx-embed-service python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"

docker exec -it onnx-embeddings-rerank-p100 bash -lc '
python3 -m pip uninstall -y onnxruntime onnxruntime-gpu &&
python3 -m pip install --no-cache-dir onnxruntime-gpu==1.20.1 &&
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
'

docker run -d \
  --name onnx-embed-service \
  --restart unless-stopped \
  --gpus all \
  -p 8002:8000 \
  -e EMBED_MODEL_DIR=/models/embedding \
  -e EMBED_SERVICE_CONFIG=/models/embedding/service_config.json \
  -e RERANK_MODEL_DIR=/models/reranker \
  -e RERANK_SERVICE_CONFIG=/models/reranker/service_config.json \
  -e API_KEYS=embeddingp100 \
  -e ORT_GPU_MEM_LIMIT_GB=14 \
  -e EMBED_MAX_BATCH_TEXTS=32 \
  -e EMBED_BATCH_TIMEOUT_MS=8 \
  -e RERANK_MAX_BATCH_PAIRS=32 \
  -e RERANK_BATCH_TIMEOUT_MS=8 \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-embedding-0.6b:/models/embedding:ro \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-reranker-0.6b:/models/reranker:ro \
  onnx-embeddings-rerank-p100:latest

curl http://localhost:8002/healthz

```

```bash
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e EMBED_MODEL_DIR=/models/embedding \
  -e EMBED_SERVICE_CONFIG=/models/embedding/service_config.json \
  -e RERANK_MODEL_DIR=/models/reranker \
  -e RERANK_SERVICE_CONFIG=/models/reranker/service_config.json \
  -e API_KEYS=embeddingp100 \
  -e ORT_GPU_MEM_LIMIT_GB=14 \
  -e EMBED_MAX_BATCH_TEXTS=32 \
  -e EMBED_BATCH_TIMEOUT_MS=8 \
  -e RERANK_MAX_BATCH_PAIRS=32 \
  -e RERANK_BATCH_TIMEOUT_MS=8 \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-embedding-0.6b:/models/embedding:ro \
  -v /home/lkk/local_services/onnx-embeddings-service/models/qwen3-reranker-0.6b:/models/reranker:ro \
  onnx-embeddings-rerank-p100:latest
```

## Authentication

Set one or more API keys with a comma-separated `API_KEYS` variable.

Requests can authenticate with either:

```bash
-H "Authorization: Bearer my-secret-key"
```

or

```bash
-H "X-API-Key: my-secret-key"
```

If `API_KEYS` is empty, auth is disabled.

## Embeddings API example

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-key' \
  -d '{
    "model": "BAAI/bge-m3",
    "input": ["hello world", "vector search"],
    "encoding_format": "float"
  }'
```

## Rerank API example

```bash
curl http://localhost:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer my-secret-key' \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "best way to deploy embeddings on a Tesla P100",
    "documents": [
      "Use ONNX Runtime with a small FastAPI wrapper.",
      "Run TEI on the newest Hopper GPU.",
      "Use a reranker after vector search to improve precision."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

## Micro-batching behavior

This service does small in-process batching.

For embeddings:
- requests arriving within `EMBED_BATCH_TIMEOUT_MS` are merged
- the merged batch is capped by `EMBED_MAX_BATCH_TEXTS`
- batching groups only requests with the same `input_type` and `dimensions`

For rerank:
- requests are grouped for a short wait window, but each request is scored independently
- `RERANK_MAX_BATCH_PAIRS` is the cap for how many query-document pairs are accepted from the queue window

That keeps the implementation simple and predictable on older GPUs.

## Health and models

```bash
curl http://localhost:8000/healthz
curl -H 'Authorization: Bearer my-secret-key' http://localhost:8000/v1/models

curl http://100.83.246.7:8002/healthz
curl -H 'Authorization: Bearer embeddingp100' http://100.83.246.7:8002/v1/models

curl http://100.83.246.7:8002/v1/embeddings   -H "Content-Type: application/json"   -H "Authorization: Bearer embeddingp100"   -d '{"input":"hello world","model":"Qwen/Qwen3-Embedding-0.6B"}'

curl http://100.83.246.7:8002/v1/rerank \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer embeddingp100' \
  -d '{
    "model": "Qwen/Qwen3-Reranker-0.6B",
    "query": "best way to deploy embeddings on a Tesla P100",
    "documents": [
      "Use ONNX Runtime with a small FastAPI wrapper.",
      "Run TEI on the newest Hopper GPU.",
      "Use a reranker after vector search to improve precision."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```



## Notes

- Keep one embedding model and one reranker model per container.
- On a P100, the 0.6B-class Qwen reranker is a much safer starting point than larger rerankers.
- Some models need specific prefixes, pooling, or score normalization. Adjust each `service_config.json` to match the model card.
- The `/v1/rerank` endpoint is OpenAI-style in spirit, but it is a custom local contract rather than an official OpenAI API clone.

## Pooling recommendations

Recommended defaults for the two common models in this service:

- `Qwen/Qwen3-Embedding-*`: `last_token` pooling
- `BAAI/bge-m3`: `cls` pooling

You can set pooling explicitly in `service_config.json`, or let the service infer a preset from `model_id`.

Example Qwen embedding config:

```json
{
  "model_id": "Qwen/Qwen3-Embedding-0.6B",
  "preset": "qwen3-embedding",
  "pooling": "last_token",
  "normalize": true,
  "max_length": 32768,
  "query_prefix": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
  "document_prefix": ""
}
```

Example BGE-M3 config:

```json
{
  "model_id": "BAAI/bge-m3",
  "preset": "bge-m3",
  "pooling": "cls",
  "normalize": true,
  "max_length": 8192,
  "query_prefix": "",
  "document_prefix": ""
}
```

Supported pooling values are `mean`, `cls`, and `last_token`.

固定 onnxruntime-gpu==1.18.1
固定 transformers==4.48.3
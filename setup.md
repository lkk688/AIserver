# AIserver Project Setup and Testing Guide

This document provides a comprehensive guide to the current setup, configuration, and testing procedures for the AIserver project, running on a single NVIDIA RTX 5090 (32GB VRAM).

## 1. System Overview

The project orchestrates multiple AI models using Docker Compose. Services are proxied via **LiteLLM** for a unified API experience, while also allowing direct access to individual model servers.

### **Service Architecture**

| Service Name | Description | Base Container | Port (Host) | Internal Port | URL |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **litellm** | Unified OpenAI-compatible API Gateway | `ghcr.io/berriai/litellm` | **4000** | 4000 | `http://localhost:4000` |
| **vllm-qwen3** | Qwen2.5-14B-Instruct-AWQ (Text Chat) | `vllm/vllm-openai:latest` | **8001** | 8000 | `http://localhost:8001` |
| **vllm-qwen3-vl** | Qwen2.5-VL-7B-Instruct-AWQ (Vision) | `vllm/vllm-openai:latest` | **8002** | 8000 | `http://localhost:8002` |
| **hfserve** | Gemma 2 / MedGemma / Embeddings | `HFserve/Dockerfile` | **8005** | 8000 | `http://localhost:8005` |
| **cosyvoice** | CosyVoice Text-to-Speech Engine | `cosyvoice/Dockerfile` (Based on `vllm-openai`) | **50000** | 50000 | `http://localhost:50000` |

---

## 2. Prerequisites & Installation

### **Prerequisites**
- NVIDIA GPU with drivers installed (Tested on RTX 5090).
- Docker & Docker Compose with NVIDIA Container Toolkit support.
- Hugging Face Token (for downloading models).

### **Configuration**

#### **1. Environment Variables (`.env`)**
Contains API keys and global settings.
```bash
HUGGING_FACE_HUB_TOKEN=hf_...
COMPOSE_PROFILES=all  # See Model Selection below
HFSERVE_EMBEDDING_MODEL=google/embeddinggemma-300m # Select embedding model
```

#### **2. Model Selection (Save GPU Memory)**
You can control which services launch by setting `COMPOSE_PROFILES` in `.env` or exporting it in your shell. This allows you to run only specific models (e.g., Qwen Only) to save VRAM.

**Available Profiles:**
*   `all`: Launch everything (Default).
*   `qwen`: Launch vllm-qwen3 (Text Chat).
*   `vl`: Launch vllm-qwen3-vl (Vision).
*   `cosyvoice`: Launch CosyVoice (TTS).
*   `hfserve`: Launch HFServe (Gemma/MedGemma/Embeddings).
*   `litellm`: Launch LiteLLM Gateway.

**Examples:**
```bash
# Launch only Text Chat and LiteLLM
export COMPOSE_PROFILES=qwen,litellm
docker compose up -d

# Launch Chat, Vision, and LiteLLM
export COMPOSE_PROFILES=qwen,vl,litellm
docker compose up -d
```

#### **3. vLLM Configuration (`vllm_config.env`)**
Controls vLLM specific settings like quantization and memory utilization.
*   **Current Tuned Settings for RTX 5090 (32GB):**
    *   `GPU_MEMORY_UTILIZATION=0.55` (Qwen 14B)
    *   `GPU_MEMORY_UTILIZATION_VL=0.3` (Qwen VL 7B)
    *   `MAX_MODEL_LEN=4096` (Context length limit to save memory)

#### 4. Embedding Configuration (`HFserve`)
To select which embedding model `hfserve` uses, set `HFSERVE_EMBEDDING_MODEL` in `.env`.
Supported models:
*   `google/embeddinggemma-300m` (Default)
*   `sentence-transformers/all-MiniLM-L12-v2`
*   `sentence-transformers/embeddinggemma-300m-medical`

**Model Selection (`HFSERVE_MODEL_TYPE`)**:
You can control which models `hfserve` loads via the `HFSERVE_MODEL_TYPE` variable in `.env`.
*   Single model: `HFSERVE_MODEL_TYPE=embedding` or `HFSERVE_MODEL_TYPE=gemma3`
*   Multiple models: `HFSERVE_MODEL_TYPE=gemma3,embedding` (Comma-separated)
> **Warning**: Loading multiple large models (e.g., Gemma 3 + Embedding) simultaneously may cause Out-Of-Memory (OOM) crashes on GPUs with 32GB or less VRAM, especially if other services like `vllm-qwen3` are running. If `hfserve` exits unexpectedly, try loading only one model type.

### D. Embedding Models**Installation & Startup**

1.  **Build and Start Services**:
    ```bash
    docker compose up -d --build
    docker compose up -d --build hfserve #only create hfserve
    ```

2.  **Check Service Health**:
    Wait for all services to report "Healthy".
    ```bash
    docker compose ps
    ```
    *Note: First run will take time to download model weights.*

3.  **View Logs** (if issues arise):
    ```bash
    docker compose logs hfserve
    docker compose logs -f vllm-qwen3 vllm-qwen3-vl
    ```

4. **Stop Services**:
    ```bash
    docker compose down
    ```

---

## 3. Testing Guide

Tests are located in the `tests/` directory. You can test models via the **LiteLLM Gateway (Port 4000)** or **Directly** against their containers.

### **A. Chat Models (Gemma / Qwen)**

**Goal**: Verify text generation capabilities (Standard & Streaming).

**Test Scripts**:
*   `tests/curl_gemma.sh`: Bash script testing LiteLLM (Normal & Streaming) and Direct FastAPI access.
*   `tests/test_gemma_chat.py`: Python script testing LiteLLM (Normal & Streaming) and Direct HFServe.

**Run Test**:
```bash

# Python Test
python tests/test_gemma_chat.py

# Shell Curl Test
./tests/curl_gemma.sh #streaming has problems via litellm

#direct streaming test
curl -N -X POST http://localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

curl -N -X POST http://localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```


### **B. Vision Models (Qwen VL)**

**Goal**: Verify image understanding and description.

**Test Script**: `tests/test_qwen3_vl.py`

**Run Test**:
```bash
python tests/test_qwen3_vl.py
```

**What it does**:
- Sends a request with an image URL to `qwen3-vl` via LiteLLM (Port 4000).
- Sends a request directly to `vllm-qwen3-vl` (Port 8002).

**Expected Output**:
```text
--- Testing Vision with qwen3-vl at http://localhost:4000 ---
Response: The image depicts a serene beach scene...
```

### **C. Speech Generation (CosyVoice)**

**Goal**: Verify Text-to-Speech (TTS) generation.

**Test Script**: `tests/test_cosyvoice.py`

**Run Test**:
```bash
python tests/test_cosyvoice.py
```

**What it does**:
- Sends a TTS request to `http://localhost:50000`.
- Saves the output audio to `test_cosyvoice_direct.wav`.

**Verify**:
- Check if `test_cosyvoice_direct.wav` is created and has a valid file size.

### **D. Embedding Models**

**Goal**: Verify text embedding generation.

**Endpoint**: `/v1/embeddings` (OpenAI Compatible)

**API Test**:
This test verifies the HTTP API endpoint exposed by HFServe.
```bash
python tests/test_embedding_api.py
```

**Comparison Test**:
To run a comparison of all supported embedding models (loads them inside the container to test):
> **Note**: This test loads multiple models and requires significant free GPU memory. You may need to stop other services or configure `HFSERVE_MODEL_TYPE=embedding` in `.env` (and restart hfserve) to free up space.

```bash
# Copy the test file into the container
docker cp tests/compare_embeddings.py hfserve:/workspace/compare_embeddings.py

# Execute inside the hfserve container
docker compose exec hfserve python3 compare_embeddings.py
```

**Direct API Curl**:
```bash
curl -X POST http://localhost:8005/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "google/embeddinggemma-300m"
  }'
```

**Configuration**:
- Set `HFSERVE_MODEL_TYPE=gemma3,embedding` in `.env` to load both Chat and Embedding models.
- Set `HFSERVE_EMBEDDING_MODEL` to select the specific embedding model.
- **Important**: If you modify `service.py`, you must rebuild the image: `docker compose up -d --build hfserve`.

```

---

## 4. Troubleshooting & Known Issues

### **1. GPU Memory Errors ("No available memory for cache blocks")**
*   **Cause**: Multiple models competing for VRAM on a single GPU.
*   **Fix**: Adjust `vllm_config.env` or use **Profiles** to run fewer models.
    *   Lower `GPU_MEMORY_UTILIZATION`.
    *   Reduce `MAX_MODEL_LEN`.
    *   Run only Qwen: `export COMPOSE_PROFILES=qwen,litellm`

### **2. CosyVoice Startup Failure (CUDA Mismatch)**
*   **Error**: `RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions.`
*   **Fix**: The `cosyvoice/server.py` file includes a runtime patch to bypass this check. Ensure this patch is present if you update the code.

### **3. "Connection Refused" or "Empty Reply"**
*   **Cause**: Service hasn't finished loading weights or is unhealthy.
*   **Check**:
    ```bash
    docker compose ps
    docker compose logs <service_name>
    ```

### **4. hfserve Connection Reset**
*   **Cause**: Often due to port conflicts or misconfiguration in `docker-compose.yml`.
*   **Fix**: Ensure `hfserve` maps to host port **8005** (not 8003 or 8000) to avoid conflicts.

# Qwen3 Inference Server with vLLM and LiteLLM

This repository sets up an inference server for **Qwen3** and **Qwen3-VL** models using [vLLM](https://github.com/vllm-project/vllm) as the inference backend and [LiteLLM](https://github.com/BerriAI/litellm) as the OpenAI-compatible gateway/proxy.

## Architecture

- **vLLM Container 1 (`vllm-qwen3`)**: Serves `Qwen/Qwen2.5-14B-Instruct-AWQ` (acting as Qwen3 placeholder).
- **vLLM Container 2 (`vllm-qwen3-vl`)**: Serves `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (acting as Qwen3-VL placeholder).
- **CosyVoice Service (`cosyvoice`)**: Custom container serving `Fun-CosyVoice3-0.5B` via OpenAI-compatible Speech API.
- **LiteLLM Container (`litellm`)**: Unified entry point (Port 4000) that routes requests to the appropriate vLLM container.

## Prerequisites

- **Docker** and **Docker Compose**
- **NVIDIA GPU(s)** with drivers installed.
- **Hugging Face Token** (if accessing gated models, though Qwen is usually open, you might need it for specific versions).

## Recent Updates & Troubleshooting

- **CosyVoice Integration**: Added a custom `Dockerfile` in `cosyvoice/` directory that extends `vllm/vllm-openai` to serve CosyVoice models.
  - Fixes: Resolved `grpcio` build failures by filtering it out from requirements (not needed for HTTP/FastAPI).
  - Environment: Sets `PYTHONPATH` to include the cloned `CosyVoice` repo and `Matcha-TTS` submodule.
- **Model Fallbacks & Quantization**: Switched to **AWQ Quantized** models (`Qwen2.5-14B-Instruct-AWQ` and `Qwen2.5-VL-7B-Instruct-AWQ`) to resolve "No available memory for cache blocks" errors on GPUs with limited memory.
- **vLLM Configuration**: Tuned `gpu-memory-utilization` (0.4 for Qwen, 0.3 for VL) to allow multiple models to coexist on a single GPU.

## Service Management

### Configuration & Model Switching
The system uses a layered configuration approach:
1.  **`.env`**: Loads basic environment variables (like `HUGGING_FACE_HUB_TOKEN`).
2.  **`vllm_config.env`**: Contains detailed model configurations (Model Name, Quantization, Memory settings).
3.  **`docker-compose.yml`**: Automatically loads variables from both `.env` and `vllm_config.env` into the containers.

**How it works:**
- When you run `./start_all.sh` or `./start_qwen3_only.sh`, they execute `docker compose up`.
- `docker compose` uses the `env_file` directive to inject variables from `.env` and `vllm_config.env` **into the container's environment**.
- The `command` in `docker-compose.yml` uses a shell entrypoint (`/bin/sh -c`) with double-dollar syntax (e.g., `$${QWEN_MODEL_NAME}`).
- This tells Docker Compose to pass the variable literally to the container.
- The container's internal shell then substitutes these variables using the values loaded from `vllm_config.env`.

**To switch models:**
1.  Open `vllm_config.env`.
2.  Uncomment the block for the model you want (e.g., Qwen 3-Next).
3.  Comment out the other blocks.
4.  Restart services:
    ```bash
    ./start_all.sh
    # OR for just Qwen3
    ./start_qwen3_only.sh
    ```

**Example `vllm_config.env` configuration:**
```env
# Switch between Qwen2.5 and Qwen3
QWEN_MODEL_NAME=Qwen/Qwen2.5-14B-Instruct-AWQ
# QWEN_MODEL_NAME=Qwen/Qwen3-14B-Instruct

# Context Length (Safe default: 8192)
MAX_MODEL_LEN=8192

# Quantization (awq, gptq, none)
QUANTIZATION=awq

# GPU Memory (0.4 allows ~2 models on 24GB VRAM)
GPU_MEMORY_UTILIZATION=0.4
```

### Advanced Features (Qwen3)

#### Thinking & Non-Thinking Modes
Qwen3 supports "thinking" before responding. You can control this via the API:
- **Enable Thinking**: Pass `chat_template_kwargs: {"enable_thinking": true}` in the API request.
- **Disable Thinking**: Pass `chat_template_kwargs: {"enable_thinking": false}`.

Example usage in Python:
```python
response = client.chat.completions.create(
    model="qwen3",
    messages=[...],
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True}
    }
)
```

#### Tool Calling Support
To enable automatic tool calling with Qwen 2.5/3, the following arguments are added to `VLLM_EXTRA_ARGS` in `vllm_config.env`:
```env
VLLM_EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser hermes
```

**Note:** You must provide a **System Prompt** (e.g., "You are a helpful assistant with access to tools...") to reliably trigger tool usage.

#### Reasoning Content (Thinking)
vLLM supports parsing reasoning content. If using Qwen3 with reasoning capabilities, you can enable specific parsers by adding flags to `VLLM_EXTRA_ARGS`:
```env
VLLM_EXTRA_ARGS=--enable-reasoning --reasoning-parser deepseek_r1
```

### Starting Specific Models
To save GPU memory, you can start only the services you need using the provided scripts:

- **Start Only Qwen3 (Chat)**:
  ```bash
  $ ./start_qwen3_only.sh
  Services started. LiteLLM available at http://localhost:4000
  Qwen3 available directly at http://localhost:8001
  ```
- **Start Only Qwen3-VL (Vision)**:
  ```bash
  ./start_qwen3_vl_only.sh
  ```
- **Start Only CosyVoice (Audio)**:
  ```bash
  ./start_cosyvoice_only.sh
  ```
- **Start All Services**:
  ```bash
  ./start_all.sh
  ```

### Stopping Services
- **If running in foreground (attached)**: Press `Ctrl+C` to stop the containers. If you are running docker-compose up (without -d ) in a terminal.
- **If running in background (detached)** If you ran the start scripts (which use -d for detached mode) or want to ensure everything is cleanly removed, run:
  ```bash
  docker compose down
  ```

### Changing Models
**Deprecated**: Please use the `.env` file to change models as described in the "Configuration & Model Switching" section above. This ensures consistent configuration across restarts.

## Setup

1. **Clone/Navigate to this folder**.

2. **Clone CosyVoice Repository**:
   The CosyVoice service requires the official repository code. Run the following command inside the `cosyvoice` directory:
   ```bash
   cd cosyvoice
   git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
   mkdir -p pretrained_models
   cd ..
   ```

3. **Set Hugging Face Token (Optional but recommended)**:
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

4. **Start Services**:
   ```bash
   docker-compose up -d --build
   ```
   *Note: It may take a while to download the model weights on the first run.*

4. **Check Logs**:
   ```bash
   docker-compose logs -f
   ```
   Wait until you see "Uvicorn running on http://0.0.0.0:8000" for vLLM containers and the CosyVoice server startup.

## Configuration

- **`docker-compose.yml`**: Defines the services, ports, and model arguments.
- **`litellm_config.yaml`**: Configures the LiteLLM routing.
  - Model `qwen3` -> routes to `vllm-qwen3`
  - Model `qwen3-vl` -> routes to `vllm-qwen3-vl`
  - Model `cosyvoice` -> routes to `cosyvoice` service
  - Model `gemma` -> routes to `hfserve` service

## HFserve (Gemma) Integration

The **HFserve** container hosts Gemma 3, TranslateGemma, and MedGemma models using Hugging Face Transformers. It now exposes an OpenAI-compatible API.

- **Endpoint**: `http://localhost:8005/v1/chat/completions` (or via LiteLLM on port 4000 as `gemma`)
- **Model Storage**: Models are cached in `~/.cache/huggingface` (mapped to `/models` in the container) to avoid re-downloading.
- **Health Checks**:
  - `GET /healthz`: Returns 200 if the process is alive.
  - `GET /readyz`: Returns 200 only when the model is fully loaded.
- **Concurrency**: Controlled by `MAX_CONCURRENT_REQUESTS` (default: 2) env var in `docker-compose.yml`.

### Usage via LiteLLM

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```
## Testing

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 1. Chat Completion (Qwen3)
Run the python test script:
```bash
python tests/test_qwen3_chat.py
```
This tests both the LiteLLM proxy (`http://localhost:4000`) and direct vLLM access (`http://localhost:8001`).

### 2. Vision Language (Qwen3-VL)
Run the vision test script:
```bash
python tests/test_qwen3_vl.py
```
Sends an image URL and a prompt to the model.

### 3. Tool Calling (Search Tool)
Run the tool calling example:
```bash
python tests/test_tool_call.py
```
Demonstrates how Qwen3 can output tool calls (e.g., `google_search`) which the client executes and feeds back.

### 4. Audio Generation (CosyVoice)
Run the audio generation test:
```bash
python tests/test_cosyvoice.py
```
Generates a speech file `test_audio.wav` from text input.

### 5. Curl Tests
Run the bash script for quick connectivity checks:
```bash
chmod +x tests/curl_tests.sh
./tests/curl_tests.sh
```

## API Usage Examples

### LiteLLM (Unified Endpoint)
**Endpoint**: `http://localhost:4000/chat/completions`

**Chat Request**:
```json
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "Hello!"}]
}
```

**Vision Request**:
```json
{
  "model": "qwen3-vl",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ]
}
```

**Audio Request**:
Endpoint: `http://localhost:4000/audio/speech`
```bash
curl -X POST http://localhost:4000/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy" \
  -d '{
    "model": "cosyvoice",
    "input": "Hello from CosyVoice!",
    "voice": "中文女"
  }' --output output.wav
```

### vLLM Direct
You can also bypass LiteLLM and hit vLLM directly if needed:
- Qwen3 Text: `http://localhost:8001/v1/chat/completions`
- Qwen3 VL: `http://localhost:8002/v1/chat/completions`
- CosyVoice: `http://localhost:50000/v1/audio/speech`

*Note: When using vLLM directly, use the exact Hugging Face model path (e.g., `Qwen/Qwen3-14B-Instruct`) as the `model` parameter.*

# Setup
```bash
docker compose up -d --build
docker-compose logs -f

./start_qwen3_only.sh
$ curl -s http://localhost:4000/health | grep "healthy" && echo "LiteLLM is healthy" || echo "LiteLLM check failed"

$ curl -X POST "http://localhost:4000/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "qwen3",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'

./test_stream.sh

$ curl -X POST "http://localhost:8001/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'

python tests/test_qwen3_chat.py

python tests/test_cosyvoice.py

curl -X POST http://localhost:4000/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy" \
  -d '{
    "model": "cosyvoice",
    "input": "Hello, this is a test.",
    "voice": "中文女"
  }' --output output.wav
```

Docker setup
```bash
docker compose build translategemma
docker compose up -d translategemma
(py312) lkk@rtx5090:/Developer/AIserver$ docker exec -it translategemma pip list | nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0

docker exec -it translategemma python service.py
docker exec -it translategemma pip list | grep vllm
```

# Gemma3
Select Model via Env Var: To run only MedGemma (saving GPU memory), you can set the environment variable in your .env file or shell:
```bash
export HFSERVE_MODEL_TYPE=medgemma
docker compose up -d --build hfserve
docker logs -f hfserve
docker compose up -d hfserve
docker compose logs -f hfserve
python tests/test_gemma_chat.py
# OR
./tests/curl_gemma.sh
docker ps | grep hfserve
docker compose down
```

Run Curl Tests: You can refer to HFserve/curl_tests.md for ready-to-use commands.
```bash
curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this X-ray"},
          {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"}
        ]
      }
    ]
  }'

curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this X-ray"},
          {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"}
        ]
      }
    ],
    "max_new_tokens": 200
  }'

# To run TranslateGemma
HFSERVE_MODEL_TYPE=translategemma docker compose up -d hfserve

# To run All models (requires ~24GB+ VRAM)
HFSERVE_MODEL_TYPE=all docker compose up -d hfserve
HFSERVE_MODEL_TYPE=all docker compose up -d --build hfserve
```
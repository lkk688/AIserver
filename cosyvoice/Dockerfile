FROM vllm/vllm-openai:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsox-dev \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from the cloned repo
COPY CosyVoice/requirements.txt /app/requirements.txt

# Remove conflicting dependencies to rely on vLLM's environment or install compatible versions
# Also remove grpcio which fails to build and is likely optional for HTTP serving
RUN sed -i '/torch/d' /app/requirements.txt && \
    sed -i '/onnxruntime/d' /app/requirements.txt && \
    sed -i '/tensorrt/d' /app/requirements.txt && \
    sed -i '/grpcio/d' /app/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
# Install torchaudio - rely on existing torch version if possible, or pip will resolve.
# Avoid forcing a specific CUDA version that might conflict with vLLM's base.
RUN pip install --no-cache-dir torchaudio
RUN pip install modelscope fastapi uvicorn python-multipart

# Set PYTHONPATH to include the mounted repo and Matcha-TTS
ENV PYTHONPATH="${PYTHONPATH}:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

# Copy server script
COPY server.py /app/server.py

EXPOSE 50000

ENTRYPOINT ["python3", "server.py", "--port", "50000", "--model_dir", "pretrained_models/Fun-CosyVoice3-0.5B"]

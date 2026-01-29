FROM nvcr.io/nvidia/pytorch:25.03-py3

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

# Remove conflicting dependencies to rely on environment or install compatible versions
# Also remove grpcio which fails to build and is likely optional for HTTP serving
# Remove exact version constraints for common libraries to avoid conflicts with base image (NVIDIA PyTorch)
# We use a broader approach to strip versions from requirements.txt for specific packages
RUN sed -i '/torch/d' /app/requirements.txt && \
    sed -i '/onnxruntime/d' /app/requirements.txt && \
    sed -i '/tensorrt/d' /app/requirements.txt && \
    sed -i '/grpcio/d' /app/requirements.txt && \
    # Remove version constraints for these packages, letting pip pick compatible ones
    sed -i 's/librosa==0.10.2/librosa/' /app/requirements.txt && \
    sed -i 's/matplotlib==3.7.5/matplotlib/' /app/requirements.txt && \
    sed -i 's/networkx==3.1/networkx/' /app/requirements.txt && \
    sed -i 's/onnx==1.16.0/onnx/' /app/requirements.txt && \
    sed -i 's/protobuf==4.25/protobuf/' /app/requirements.txt && \
    sed -i 's/pyarrow==18.1.0/pyarrow/' /app/requirements.txt && \
    sed -i 's/pydantic==2.7.0/pydantic/' /app/requirements.txt && \
    sed -i 's/uvicorn==0.30.0/uvicorn/' /app/requirements.txt && \
    sed -i 's/fastapi==0.115.6/fastapi/' /app/requirements.txt && \
    sed -i 's/gradio==5.4.0/gradio/' /app/requirements.txt && \
    sed -i 's/modelscope==1.20.0/modelscope/' /app/requirements.txt && \
    sed -i 's/rich==13.7.1/rich/' /app/requirements.txt && \
    sed -i 's/soundfile==0.12.1/soundfile/' /app/requirements.txt && \
    sed -i 's/tensorboard==2.14.0/tensorboard/' /app/requirements.txt && \
    sed -i 's/x-transformers==2.11.24/x-transformers/' /app/requirements.txt && \
    sed -i 's/lightning==2.2.4/lightning/' /app/requirements.txt && \
    sed -i 's/hydra-core==1.3.2/hydra-core/' /app/requirements.txt && \
    sed -i 's/omegaconf==2.3.0/omegaconf/' /app/requirements.txt && \
    sed -i 's/openai-whisper==20231117/openai-whisper/' /app/requirements.txt && \
    sed -i 's/conformer==0.3.2/conformer/' /app/requirements.txt && \
    sed -i 's/diffusers==0.29.0/diffusers/' /app/requirements.txt && \
    sed -i 's/wetext==0.0.4/wetext/' /app/requirements.txt && \
    # Also clean up others that might conflict in NVIDIA container
    sed -i 's/numpy==1.26.4/numpy/' /app/requirements.txt && \
    sed -i 's/transformers==4.51.3/transformers/' /app/requirements.txt && \
    sed -i 's/torch==.*/torch/' /app/requirements.txt && \
    sed -i 's/torchaudio==.*/torchaudio/' /app/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
# Install torchaudio - rely on existing torch version if possible, or pip will resolve.
# Avoid forcing a specific CUDA version that might conflict with base.
# Reinstall both torch and torchaudio from nightly to ensure ABI compatibility
# The base image has a custom torch that lacks a matching torchaudio wheel, so we force update both.
# We must clear PIP_CONSTRAINT which pins the NVIDIA torch version
# Also reinstall torchvision as it depends on torch
RUN pip uninstall -y torch torchaudio torchvision && \
    PIP_CONSTRAINT= pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
# Explicitly install onnxruntime (GPU version preferred if available, but keep it simple)
# Since we are on NVIDIA image, try onnxruntime-gpu if compatible, or just onnxruntime
RUN pip install --no-cache-dir onnxruntime-gpu
# Install hyperpyyaml and other missing dependencies
RUN pip install --no-cache-dir hyperpyyaml==1.0.1
RUN pip install modelscope fastapi uvicorn python-multipart
# vLLM omitted due to dependency conflicts with base image Torch 2.7.0
# RUN pip install vllm

# Set PYTHONPATH to include the mounted repo and Matcha-TTS
ENV PYTHONPATH="${PYTHONPATH}:/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS"

# Copy server script
COPY server.py /app/server.py

EXPOSE 50000

ENTRYPOINT ["python3", "server.py", "--port", "50000", "--model_dir", "pretrained_models/Fun-CosyVoice3-0.5B-2512"]

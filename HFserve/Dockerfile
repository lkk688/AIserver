FROM nvcr.io/nvidia/pytorch:25.03-py3

LABEL maintainer="kaikai.liu@sjsu.edu"

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

# Target Blackwell architectures:
# sm_100 (Server/B100), sm_120 (RTX 50-series)
ENV TORCH_CUDA_ARCH_LIST="10.0;12.0"
ENV FORCE_CUDA="1"

# Expose ports for JupyterLab (8888), Ollama (11434), llama.cpp (8000)
EXPOSE 8888 11434 8000

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    curl \
    wget \
    ninja-build \
    libcurl4-openssl-dev \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade core libraries and install bitsandbytes/transformers
# (vLLM and PyTorch are already pre-installed in this image)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --upgrade \
    transformers \
    accelerate \
    bitsandbytes \
    sentence-transformers \
    langchain \
    tabulate \
    fastapi \
    uvicorn \
    jupyterlab

# Clone and build llama.cpp with CUDA support for Blackwell
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="100;120" && \
    cmake --build . --config Release --parallel $(nproc)

# Install llama-cpp-python with specific Blackwell targets
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100;120" \
    pip install llama-cpp-python --no-cache-dir

# Install Ollama (using the universal installer which handles CUDA detection)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup working directory and shared model volume
RUN mkdir -p /models /workspace
WORKDIR /workspace

# Copy service code
COPY service.py .
COPY translategemma.py .
COPY gemma3.py .
COPY medgemma.py .
COPY embedding.py .

# Use the pre-configured environment in the container
CMD ["python3", "service.py"]

#!/bin/bash
# Start only Qwen3 (Chat) + LiteLLM
echo "Starting Qwen3 (Chat) and LiteLLM..."
docker compose up -d --build vllm-qwen3 litellm
echo "Services started. LiteLLM available at http://localhost:4000"
echo "Qwen3 available directly at http://localhost:8001"

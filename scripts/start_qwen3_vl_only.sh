#!/bin/bash
# Start only Qwen3-VL (Vision) + LiteLLM
echo "Starting Qwen3-VL (Vision) and LiteLLM..."
docker compose up -d --build vllm-qwen3-vl litellm
echo "Services started. LiteLLM available at http://localhost:4000"
echo "Qwen3-VL available directly at http://localhost:8002"

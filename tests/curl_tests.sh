#!/bin/bash

echo "--- Testing LiteLLM Health ---"
curl -s http://localhost:4000/health | grep "healthy" && echo "LiteLLM is healthy" || echo "LiteLLM check failed"

echo -e "\n--- Testing Chat via LiteLLM (Qwen3) ---"
curl -X POST "http://localhost:4000/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "qwen3",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'

echo -e "\n\n--- Testing Chat via vLLM Direct (Qwen3) ---"
curl -X POST "http://localhost:8001/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50
     }'

echo -e "\n\n--- Testing Audio via LiteLLM (CosyVoice) ---"
curl -X POST "http://localhost:4000/audio/speech" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "cosyvoice",
       "input": "Testing audio generation.",
       "voice": "中文女"
     }' --output test_curl_audio.wav

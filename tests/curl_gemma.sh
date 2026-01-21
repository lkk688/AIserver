#!/bin/bash

# ==========================================
# Gemma Chat Test Script (Unified & Direct)
# ==========================================

echo "---------------------------------------------------"
echo "1. Testing Gemma via LiteLLM (Port 4000) - STANDARD"
echo "---------------------------------------------------"
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy-key" \
  -d '{
    "model": "gemma",
    "messages": [
      {"role": "user", "content": "Hello! Explain quantum physics in one sentence."}
    ],
    "max_tokens": 100
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "2. Testing Gemma via LiteLLM (Port 4000) - STREAMING"
echo "---------------------------------------------------"
curl -N -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy-key" \
  -d '{
    "model": "gemma",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "max_tokens": 50,
    "stream": true
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "3. Testing Gemma Direct (Port 8005) - FASTAPI"
echo "---------------------------------------------------"
curl -X POST http://localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma",
    "messages": [
      {"role": "user", "content": "Hello! Are you running directly?"}
    ],
    "max_tokens": 50
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "4. Testing Health & Readiness (Port 8005)"
echo "---------------------------------------------------"
echo -n "Health: "
curl -s http://localhost:8005/healthz
echo -e "\n"
echo -n "Ready: "
curl -s http://localhost:8005/readyz
echo -e "\n"

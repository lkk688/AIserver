#!/bin/bash

# ==========================================
# MedGemma Chat Test Script (Unified & Direct)
# ==========================================

echo "---------------------------------------------------"
echo "1. Testing MedGemma via LiteLLM (Port 4000) - STANDARD"
echo "---------------------------------------------------"
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy-key" \
  -d '{
    "model": "medgemma",
    "messages": [
      {"role": "user", "content": "Hello! Explain acupuncture in traditional Chinese medicine."}
    ],
    "max_tokens": 100
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "2. Testing MedGemma via LiteLLM (Port 4000) - STREAMING"
echo "---------------------------------------------------"
curl -N -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy-key" \
  -d '{
    "model": "medgemma",
    "messages": [
      {"role": "user", "content": "What is traditional Chinese medicine?"}
    ],
    "max_tokens": 50,
    "stream": true
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "3. Testing MedGemma Direct (Port 8005) - FASTAPI"
echo "---------------------------------------------------"
curl -X POST http://localhost:8005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "medgemma",
    "messages": [
      {"role": "user", "content": "Hello! Do you know traditional Chinese medicine?"}
    ],
    "max_tokens": 50
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "4. Testing MedGemma Vision via LiteLLM (Port 4000)"
echo "---------------------------------------------------"
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-dummy-key" \
  -d '{
    "model": "medgemma",
    "messages": [
      {
        "role": "system",
        "content": "You are an expert radiologist."
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this X-ray"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'
echo -e "\n\nDone."

echo "---------------------------------------------------"
echo "5. Testing Health & Readiness (Port 8005)"
echo "---------------------------------------------------"
echo -n "Health: "
curl -s http://localhost:8005/healthz
echo -e "\n"
echo -n "Ready: "
curl -s http://localhost:8005/readyz
echo -e "\n"

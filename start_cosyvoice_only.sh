#!/bin/bash
# Start only CosyVoice (Audio) + LiteLLM
echo "Starting CosyVoice (Audio) and LiteLLM..."
docker compose up -d --build cosyvoice litellm
echo "Services started. LiteLLM available at http://localhost:4000"
echo "CosyVoice available directly at http://localhost:50000"

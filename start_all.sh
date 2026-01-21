#!/bin/bash
# Start ALL services
echo "Starting ALL services..."
docker compose up -d --build
echo "All services started. LiteLLM available at http://localhost:4000"

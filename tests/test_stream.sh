#!/bin/bash

# Test Streaming Chat Completion
echo "Testing Streaming Chat Completion..."

curl -N -X POST "http://localhost:4000/chat/completions" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer sk-dummy" \
     -d '{
       "model": "qwen3",
       "messages": [{"role": "user", "content": "Count from 1 to 5 and explain why."}],
       "max_tokens": 100,
       "stream": true
     }'

echo -e "\n\nDone."

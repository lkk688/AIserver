
#!/bin/bash

# Test Script for Doubao TTS Streaming API
# Usage: ./test_doubao_tts.sh [TEXT]
# ./backend/tests/test_doubao_tts.sh "Hello world, this is a test."

TEXT=${1:-"Hello, this is a test of the Doubao TTS streaming API."}
OUTPUT_FILE="doubao_output.mp3"
API_URL="http://127.0.0.1:8000/v1/tts/doubao/stream"

echo "Testing Doubao TTS API..."
echo "Text: $TEXT"
echo "URL: $API_URL"

# Send POST request
# -N: no buffer (streaming)
# --raw: no processing
# -v: verbose headers
curl -N -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"$TEXT\",
    \"voice\": \"BV700_streaming\",
    \"speed\": 1.0,
    \"format\": \"mp3\"
  }" \
  --output "$OUTPUT_FILE" \
  -w "\n\nHTTP Status: %{http_code}\nTime Total: %{time_total}s\nSize Download: %{size_download} bytes\n"

if [ -s "$OUTPUT_FILE" ]; then
    echo "Success! Audio saved to $OUTPUT_FILE"
    echo "You can play it with: afplay $OUTPUT_FILE (macOS) or mpg123 $OUTPUT_FILE (Linux)"
else
    echo "Failed. Output file is empty."
    cat "$OUTPUT_FILE"
fi

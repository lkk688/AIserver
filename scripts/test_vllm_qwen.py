import time
from openai import OpenAI

# Initialize the OpenAI client pointing to your local vLLM server
# Using the IP address from your curl output
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1", #"http://100.110.236.127:8000/v1",
)

MODEL_NAME = "qwen3.5-9b"

print(f"🚀 Starting comprehensive test for {MODEL_NAME}...\n")
print("-" * 60)

# ==========================================
# Test 1: Standard Text Generation (Non-streaming)
# ==========================================
print("📝 Test 1: Standard Text Generation (Non-streaming)")
prompt_1 = "Explain the concept of 'Quantum Entanglement' in two short sentences."

start_time = time.time()
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt_1}
    ],
    max_tokens=256,
    temperature=0.7,
)
end_time = time.time()

duration = end_time - start_time
# Extract usage statistics returned by the API
usage = response.usage
completion_tokens = usage.completion_tokens

print(f"Assistant: {response.choices[0].message.content.strip()}")
print(f"\n📊 [Metrics - Text Non-streaming]")
print(f"   Time taken      : {duration:.2f} seconds")
print(f"   Generated tokens: {completion_tokens}")
print(f"   Output Speed    : {completion_tokens / duration:.2f} tokens/sec")
print("-" * 60)


# ==========================================
# Test 2: Streaming Text Generation
# ==========================================
print("🌊 Test 2: Streaming Text Generation")
prompt_2 = "Write a short Python function to calculate the Fibonacci sequence."

start_time = time.time()
# Note: stream_options={"include_usage": True} forces vLLM to send token counts in the final chunk
response_stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": prompt_2}
    ],
    max_tokens=256,
    temperature=0.7,
    stream=True,
    stream_options={"include_usage": True} 
)

print("Assistant: \n", end="")
first_token_time = None
final_usage = None

for chunk in response_stream:
    # Record Time To First Token (TTFT)
    if first_token_time is None:
        first_token_time = time.time()
        
    # Print the streamed text chunks
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        text = chunk.choices[0].delta.content
        print(text, end="", flush=True)
        
    # The last chunk will contain the usage statistics if include_usage=True
    if chunk.usage is not None:
        final_usage = chunk.usage

end_time = time.time()
print("\n")

# Calculate streaming metrics
ttft = first_token_time - start_time
generation_duration = end_time - first_token_time

if final_usage:
    comp_tokens = final_usage.completion_tokens
    print(f"📊 [Metrics - Text Streaming]")
    print(f"   Time To First Token (TTFT): {ttft:.3f} seconds")
    print(f"   Generated tokens          : {comp_tokens}")
    # We calculate generation speed excluding the TTFT (which includes prompt processing/prefill time)
    print(f"   Generation Speed          : {comp_tokens / generation_duration:.2f} tokens/sec")
else:
    print("⚠️ Usage statistics not returned by the server.")
print("-" * 60)


# ==========================================
# Test 3: Vision/Image Input (Native Multimodal Test)
# ==========================================
print("🖼️ Test 3: Vision/Image Input (Native Multimodal)")
# Using a sample image of a nature boardwalk
image_url = "https://sanjosespotlight.s3.us-east-2.amazonaws.com/wp-content/uploads/2020/03/19195115/SJSU--1160x560.jpg"

start_time = time.time()
response_vision = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail. What do you see?"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ],
    max_tokens=256,
)
end_time = time.time()

duration_vision = end_time - start_time
usage_vision = response_vision.usage
tokens_vision = usage_vision.completion_tokens

print(f"Assistant: {response_vision.choices[0].message.content.strip()}")
print(f"\n📊 [Metrics - Vision]")
print(f"   Time taken      : {duration_vision:.2f} seconds")
print(f"   Generated tokens: {tokens_vision}")
print(f"   Output Speed    : {tokens_vision / duration_vision:.2f} tokens/sec")
print("-" * 60)

print("✅ All tests completed successfully!")

"""
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --served-model-name qwen3.5-9b \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-prefix-caching \
    --trust-remote-code

tailscale ip -4
curl http://100.110.236.127:8000/v1/models
sudo tailscale serve --bg --port 443 localhost:8000
#share externally
sudo tailscale funnel --bg --port 443 localhost:8000

pkill -9 -f vllm
kill -9 $(lsof -t -i:8000)
"""
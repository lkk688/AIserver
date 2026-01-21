import openai
import os
import requests
import sys

# Configuration
LITELLM_API_BASE = "http://localhost:4000/v1"
LITELLM_API_KEY = "sk-dummy-key"
HFSERVE_API_BASE = "http://localhost:8005/v1"
MODEL = "gemma"

def test_litellm_chat():
    print(f"\n--- [1] Testing LiteLLM Standard Chat ({LITELLM_API_BASE}) ---")
    client = openai.OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_API_BASE)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Hello, who are you?"}
            ],
            max_tokens=100
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

def test_litellm_stream():
    print(f"\n--- [2] Testing LiteLLM Streaming Chat ({LITELLM_API_BASE}) ---")
    client = openai.OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_API_BASE)

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "List 3 colors."}
            ],
            max_tokens=50,
            stream=True
        )
        print("Stream Response: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\nDone.")
    except Exception as e:
        print(f"Error: {e}")

def test_hfserve_direct():
    print(f"\n--- [3] Testing HFServe Direct FastAPI ({HFSERVE_API_BASE}) ---")
    # Using requests for direct FastAPI test to ensure it works without OpenAI SDK wrapper overhead first
    url = f"{HFSERVE_API_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Are you running directly?"}],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response:", response.json()['choices'][0]['message']['content'])
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    print(f"Starting Tests for Model: {MODEL}")
    test_litellm_chat()
    test_litellm_stream()
    test_hfserve_direct()

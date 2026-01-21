import openai
import os

def test_chat(base_url, model_name, api_key="sk-dummy", enable_thinking=None, extra_params=None):
    print(f"\n--- Testing Chat with {model_name} at {base_url} ---")
    if enable_thinking is not None:
        print(f"Thinking Mode: {'Enabled' if enable_thinking else 'Disabled'}")
    
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Base parameters
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, tell me a short joke about AI."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.8
    }

    # Add extra body for top_k (OpenAI client doesn't support top_k as a direct argument)
    if "extra_body" not in params:
        params["extra_body"] = {}
    params["extra_body"]["top_k"] = 20

    # Add extra body for Thinking Mode if specified
    if enable_thinking is not None:
        params["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    
    # Merge any other extra params (like presence_penalty)
    if extra_params:
        if "extra_body" not in params:
            params["extra_body"] = {}
        params["extra_body"].update(extra_params)

    try:
        response = client.chat.completions.create(**params)
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test via LiteLLM (Standard)
    test_chat("http://localhost:4000", "qwen3")
    
    # Test via vLLM directly with explicit model name
    # Ensure this matches what is running (default from .env is AWQ)
    model_name = "Qwen/Qwen2.5-14B-Instruct-AWQ"
    
    # 1. Standard Chat
    test_chat("http://localhost:8001/v1", model_name)

    # 2. Thinking Mode Explicitly Disabled (Soft Switch)
    test_chat("http://localhost:8001/v1", model_name, enable_thinking=False)

    # 3. Thinking Mode Enabled (if model supports it)
    # Note: Qwen2.5 might ignore this if not trained for it, but Qwen3 will use it.
    test_chat("http://localhost:8001/v1", model_name, enable_thinking=True)

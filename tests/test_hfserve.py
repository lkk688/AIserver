import requests
import json

BASE_URL = "http://localhost:8003"

def test_translategemma():
    print("Testing TranslateGemma...")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Translate this to Spanish: Hello, how are you?"
                    }
                ]
            }
        ],
        "max_new_tokens": 50
    }
    try:
        response = requests.post(f"{BASE_URL}/translategemma/translate", json=payload)
        response.raise_for_status()
        print("TranslateGemma Response:", response.json())
    except Exception as e:
        print(f"TranslateGemma failed: {e}")
        if 'response' in locals() and hasattr(response, 'content'):
            print(f"Error content: {response.content}")

def test_gemma3():
    print("\nTesting Gemma 3...")
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the capital of France?"
                    }
                ]
            }
        ],
        "max_new_tokens": 50
    }
    try:
        response = requests.post(f"{BASE_URL}/gemma3/chat", json=payload)
        response.raise_for_status()
        print("Gemma 3 Response:", response.json())
    except Exception as e:
        print(f"Gemma 3 failed: {e}")
        if 'response' in locals() and hasattr(response, 'content'):
            print(f"Error content: {response.content}")

if __name__ == "__main__":
    test_translategemma()
    test_gemma3()

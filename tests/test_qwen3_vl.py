import openai

def test_vl(base_url, model_name, api_key="sk-dummy"):
    print(f"\n--- Testing Vision with {model_name} at {base_url} ---")
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # Image URL example (Qwen logo or generic image)
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                    ],
                }
            ],
            max_tokens=300
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test via LiteLLM
    test_vl("http://localhost:4000", "qwen3-vl")

    # Test via vLLM directly
    test_vl("http://localhost:8002/v1", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

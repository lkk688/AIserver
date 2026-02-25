import os
from openai import OpenAI

client = OpenAI(
    base_url="https://w0wqtv67-8000.usw3.devtunnels.ms/v1",
    api_key="myhpcvllmqwen123"
)

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-Next-FP8",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )
    print("Type of response:", type(response))
    print("Response:", response)
except Exception as e:
    print("Exception:", e)

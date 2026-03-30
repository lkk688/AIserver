import base64
import requests

# 1. 设置你的图片路径 (确保这张图片真实存在)
image_path = "demo.jpeg"

# 2. 将图片转换为 Base64 编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image(image_path)

# 3. 构造发给 vLLM 的请求数据
headers = {
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen3.5-35b-moe-int4",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你好，请仔细观察这张图片，描述一下它所展现的场景、其中的技术元素以及整体氛围。"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # 注意这里要拼接 data:image/jpeg;base64, 前缀
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 512
}

# 4. 发送请求并打印结果
print("正在发送请求到 5090，请稍候...")
response = requests.post("http://localhost:8000/v1/chat/completions", headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print("\n模型回复：")
    print(result['choices'][0]['message']['content'])
else:
    print(f"请求失败，状态码: {response.status_code}")
    print(response.text)
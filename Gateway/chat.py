import requests
import json
import sys

url = "http://100.83.246.7:8080/v1/chat/completions"

# payload = {
#     "model": "qwen3.5-9b",
#     "messages": [
#         # 加入系统提示词，严厉要求它精简思考
#         {"role": "system", "content": "你是一个严谨的AI硬件专家。思考过程必须极其精简，直接切中要害，绝不自我反复或啰嗦。"},
#         {"role": "user", "content": "1080Ti和P100互换显存和算力，对AI推理有何影响？请分三点简述。"}
#     ],
#     "stream": True,
#     "max_tokens": 2048,           # 👈 核心修改 1：把令牌上限放大，给足回答的空间
#     "temperature": 0.5,           # 👈 核心修改 2：强制降温！让它变得果断、少废话
#     "top_p": 0.85,                # 收束概率，减少幻觉
#     "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
# }

payload = {
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "How Nvidia used cuda to accelerate the CNNs? Why Nvidia developed the Tensor Core? What's the advantage of Tensor Core over the Cuda cores?"}],
    "stream": True,
    "max_tokens": 2048,
    "temperature": 0.5,
    # disable thinking
    "chat_template_kwargs": {"enable_thinking": False}, 
    "stop": ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]
}

print("\n🤖 AI Thinking \n" + "="*50)

# send request
with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:] # 剥离 "data: " 前缀
                
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    
                    # 1. print the reasoning process
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        sys.stdout.write(f"\033[90m{delta['reasoning_content']}\033[0m")
                    
                    # 2. print the final answer
                    if "content" in delta and delta["content"]:
                        sys.stdout.write(delta["content"])
                        
                    sys.stdout.flush() #Very important, make sure to flush the output
                except json.JSONDecodeError:
                    pass

print("\n" + "="*50 + "\n✅ Complete！")
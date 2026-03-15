#!/bin/bash
# 文件名: test.sh
# 赋予执行权限: chmod +x test.sh

API_URL="http://localhost:4000/v1/chat/completions"
API_KEY="sk-dummy-key" # 你的 LiteLLM key

echo "==============================================="
echo "       🚀 Qwen3.5-35B-A3B 终极连通性测试       "
echo "==============================================="

echo -e "\n🧪 [测试 1/3] 检查 Docker 容器存活状态..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "vllm-qwen3|litellm"

# ---------------------------------------------------------
echo -e "\n\n🧪 [测试 2/3] 纯代码流式输出测试 (Streaming)..."
echo ">> 预期: 屏幕上会一行一行打印打字机效果 (SSE 数据流)"
sleep 2

curl -N -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $API_KEY" \
     -d '{
       "model": "qwen3",
       "messages": [
         {
           "role": "user",
           "content": "用 Python 写一个极简的 Hello World，只要代码。"
         }
       ],
       "temperature": 0.1,
       "stream": true
     }'

# ---------------------------------------------------------
echo -e "\n\n\n🧪 [测试 3/3] 原生多模态图片识别测试 (Vision + Streaming)..."
echo ">> 预期: 模型识别并流式描述给定 URL 的图片内容"
sleep 2

# 使用标准的 OpenAI Vision 消息格式
curl -N -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $API_KEY" \
     -d '{
       "model": "qwen3",
       "messages": [
         {
           "role": "user",
           "content": [
             {
               "type": "text",
               "text": "仔细观察这张图片，用简短的一句话描述里面有什么内容？"
             },
             {
               "type": "image_url",
               "image_url": {
                 "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
               }
             }
           ]
         }
       ],
       "temperature": 0.2,
       "max_tokens": 300,
       "stream": true
     }'

echo -e "\n\n🎉 测试脚本执行完毕！"
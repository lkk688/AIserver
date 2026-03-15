#!/bin/bash
# 文件名: stop.sh
# 赋予执行权限: chmod +x stop.sh

echo "🛑 正在安全关闭所有 AI 容器服务..."

# --profile all 确保无论是基础模式还是包含 audio 的模式，都能被干净地关闭
docker compose --profile all down

echo "🧹 释放悬空网络和无用资源..."
docker system prune -f --volumes --filter "label=com.docker.compose.project"

echo "✅ 所有服务已关闭。"
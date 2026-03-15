#!/bin/bash
# 文件名: start.sh
# 赋予执行权限: chmod +x start.sh

echo "========================================"
echo "    启动 AI 核心服务 (Qwen3.5 架构)     "
echo "========================================"

# 基础 Profile：启动 Qwen、LiteLLM 和 没有打 profile 标签的基础服务(DB/Redis)
PROFILES="--profile qwen --profile litellm"

if [[ "$1" == "--with-audio" ]]; then
    echo "🔊 模式: 完整模式 (包含 CosyVoice / VibeVoice)"
    PROFILES="--profile all"
else
    echo "⚡ 模式: 轻量模式 (仅启动 LLM + DB，跳过 Audio)"
    echo "💡 提示: 若需启动音频服务，请运行 ./start.sh --with-audio"
fi

echo "🚀 正在拉起容器..."
# 使用所选 profile 并在后台(-d)启动，--remove-orphans 清理掉废弃的 VL 容器
docker compose $PROFILES up -d --remove-orphans

echo "✅ 启动命令已发送！请使用 'docker compose logs -f vllm-qwen3' 查看模型加载进度。"
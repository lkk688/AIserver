# 使用官方底座，保证基础 CUDA 环境完整
FROM vllm/vllm-openai:latest

USER root

# 升级 pip 并强制安装指定的 vLLM 0.17.1 正式版及最新依赖
RUN pip install --upgrade pip && \
    pip install vllm==0.17.1 transformers optimum autoawq --upgrade

# 恢复官方纯净的启动入口
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, Response
from collections import defaultdict
import json

app = FastAPI(title="My Private AI Cloud Gateway", version="1.0.0")

# ==========================================
# ⚙️ 内网物理节点配置 (请替换为你的真实 Tailscale IP)
# ==========================================
# P100 节点 (Ollama)
LLM_BASE_URL = "http://100.x.x.1:11434"     

# 1080Ti 节点 A (Faster-Whisper & Infinity)
ASR_BASE_URL = "http://100.81.148.35:8001"      
EMBED_BASE_URL = "http://100.81.148.35:8003"    

# 1080Ti 节点 B (GPT-SoVITS)
TTS_BASE_URL = "http://100.81.148.35:8002"      

# ==========================================
# 🔐 鉴权与统计模块 (生产环境建议存入 Postgres & Redis)
# ==========================================
VALID_TOKENS = {
    "sk-vip-001": "user_alice",
    "sk-test-002": "user_bob"
}

# 结构：{"user_alice": {"llm": 10, "asr": 5, "tts": 2, "embed": 100}}
usage_stats = defaultdict(lambda: defaultdict(int))

async def verify_token(request: Request):
    """全局鉴权拦截器"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = auth_header.split(" ")[1]
    user = VALID_TOKENS.get(token)
    if not user:
        raise HTTPException(status_code=403, detail="Token invalid or expired")
    
    return user

# ==========================================
# 🧠 1. 大模型 LLM 路由 (流式转发)
# ==========================================
@app.post("/v1/chat/completions")
async def proxy_llm(request: Request, user: str = Depends(verify_token)):
    usage_stats[user]["llm_calls"] += 1
    
    # 获取原始请求体
    body = await request.body()
    
    client = httpx.AsyncClient()
    try:
        req = client.build_request(
            "POST", 
            f"{LLM_BASE_URL}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            content=body
        )
        # 保持流式响应，让打字机效果透传给公网用户
        response = await client.send(req, stream=True)
        return StreamingResponse(
            response.aiter_raw(), 
            status_code=response.status_code, 
            headers={"Content-Type": response.headers.get("Content-Type", "text/event-stream")}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM Node Error: {str(e)}")

# ==========================================
# 🎤 2. ASR 语音识别路由 (处理文件上传)
# ==========================================
@app.post("/v1/audio/transcriptions")
async def proxy_asr(request: Request, user: str = Depends(verify_token)):
    usage_stats[user]["asr_calls"] += 1
    
    body = await request.body()
    # ASR 是 multipart/form-data，必须保留原请求的 Content-Type 以防 Boundary 丢失
    headers = {k: v for k, v in request.headers.items() if k.lower() in ["content-type", "content-length"]}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ASR_BASE_URL}/v1/audio/transcriptions",
                headers=headers,
                content=body,
                timeout=60.0 # 语音识别可能较慢，放宽超时时间
            )
            return Response(content=response.content, status_code=response.status_code, headers=dict(response.headers))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"ASR Node Error: {str(e)}")

# ==========================================
# 🗣️ 3. TTS 语音合成路由 (返回音频流)
# ==========================================
@app.post("/v1/audio/speech")
async def proxy_tts(request: Request, user: str = Depends(verify_token)):
    usage_stats[user]["tts_calls"] += 1
    body = await request.body()
    
    async with httpx.AsyncClient() as client:
        try:
            # 转发到内部的 GPT-SoVITS 根路径
            response = await client.post(
                f"{TTS_BASE_URL}/",
                headers={"Content-Type": "application/json"},
                content=body,
                timeout=60.0
            )
            # 透传返回的 WAV 音频流
            return Response(
                content=response.content, 
                status_code=response.status_code, 
                headers={"Content-Type": "audio/wav"}
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"TTS Node Error: {str(e)}")

# ==========================================
# 📚 4. Embedding 向量化路由
# ==========================================
@app.post("/v1/embeddings")
async def proxy_embeddings(request: Request, user: str = Depends(verify_token)):
    usage_stats[user]["embed_calls"] += 1
    body = await request.body()
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{EMBED_BASE_URL}/embeddings",
                headers={"Content-Type": "application/json"},
                content=body
            )
            return Response(content=response.content, status_code=response.status_code, media_type="application/json")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Embedding Node Error: {str(e)}")

# ==========================================
# 📊 5. 运营管理：查看调用统计
# ==========================================
@app.get("/admin/stats")
async def get_stats():
    # 实际生产中可以加一个 Admin Token 校验
    return {"status": "success", "usage_data": usage_stats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
import asyncio
import time
from openai import AsyncOpenAI

# 使用 AsyncOpenAI 来支持高并发请求
client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8000/v1", # 替换为你的实际 IP
)

MODEL_NAME = "qwen3.5-9b"
SYSTEM_PROMPT = "你是一个资深架构师。请直接回答问题，不要说多余的废话。"

# 核心测试函数
async def fetch_response(prompt_id, prompt):
    start_time = time.time()
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.7,
    )
    end_time = time.time()
    
    duration = end_time - start_time
    tokens = response.usage.completion_tokens
    
    return {
        "id": prompt_id,
        "duration": duration,
        "tokens": tokens,
        "speed": tokens / duration
    }

async def main():
    print(f"🚀 开始 vLLM 性能基准测试 (Model: {MODEL_NAME})\n")

    # ==========================================
    # 实验一：串行请求 (模拟传统单线程 Agent)
    # ==========================================
    print("⏳ [实验一] 单一请求测试...")
    single_prompt = "请详细分析微服务架构的优缺点，字数在400字左右。"
    
    start_wall_time = time.time()
    single_result = await fetch_response("Single", single_prompt)
    single_wall_time = time.time() - start_wall_time
    
    print(f"   耗时: {single_result['duration']:.2f} 秒")
    print(f"   生成 Token 数: {single_result['tokens']}")
    print(f"   单请求吞吐量: {single_result['speed']:.2f} Tokens/s\n")

    # ==========================================
    # 实验二：并发请求 (模拟多视角 Agent 思考)
    # ==========================================
    print("🌪️ [实验二] 3并发 Batch 请求测试 (多视角思考)...")
    # 模拟 Agent 在同一时间向模型分发 3 个不同的子任务
    batch_prompts = [
        "请从【性能与延迟】的视角，分析微服务架构的挑战，字数在400字左右。",
        "请从【数据一致性与分布式事务】的视角，分析微服务架构的挑战，字数在400字左右。",
        "请从【运维与监控】的视角，分析微服务架构的挑战，字数在400字左右。"
    ]
    
    # 记录并发总墙上时间 (Wall-clock time)
    start_wall_time = time.time()
    
    # asyncio.gather 会将 3 个请求同时发射给 vLLM
    tasks = [fetch_response(f"Batch-{i+1}", prompt) for i, prompt in enumerate(batch_prompts)]
    batch_results = await asyncio.gather(*tasks)
    
    batch_wall_time = time.time() - start_wall_time
    
    # 统计数据
    total_tokens = sum(res["tokens"] for res in batch_results)
    
    for res in batch_results:
        print(f"   任务 {res['id']} | 耗时: {res['duration']:.2f}s | 生成 Tokens: {res['tokens']} | 独立速度: {res['speed']:.2f} T/s")
        
    print(f"\n📈 [并发实验总结]")
    print(f"   总耗时 (用户等待时间) : {batch_wall_time:.2f} 秒")
    print(f"   总生成 Token 数       : {total_tokens}")
    print(f"   🚀 整体系统吞吐量 (Throughput) : {total_tokens / batch_wall_time:.2f} Tokens/s")
    
    # ==========================================
    # 结论分析
    # ==========================================
    print("\n" + "="*50)
    print("💡 结论洞察：")
    throughput_boost = (total_tokens / batch_wall_time) / single_result['speed']
    time_increase = batch_wall_time / single_result['duration']
    print(f"1. 发送 3 个请求，用户总等待时间仅是单请求的 {time_increase:.2f} 倍。")
    print(f"2. 系统的整体吞吐量提升了 {throughput_boost:.2f} 倍！")
    print("证明了并发多视角思考在 vLLM 上是极具性价比的架构选择。")

if __name__ == "__main__":
    asyncio.run(main())

"""
Prefix Caching
--enable-prefix-caching
"""

import asyncio
import redis.asyncio as redis
import json
import uuid
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_client")

# Configuration from environment or defaults
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

async def run_client():
    logger.info("Starting Test Client...")
    
    # 1. Connect to Redis (Main client for pushing)
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        await r.ping()
        logger.info("[Redis] Connected.")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return

    # 1b. Connect to Redis (Binary client for PubSub)
    r_sub = redis.from_url(REDIS_URL, decode_responses=False)

    # 2. Prepare Task
    user_id = "test_client_user"
    job_id = str(uuid.uuid4())
    channel = f"user_{user_id}"
    
    task_payload = {
        "type": "llm",
        "job_id": job_id,
        "user_id": user_id,
        "model": "qwen3", 
        "messages": [{"role": "user", "content": "Hello from test_client! Tell me a short joke."}]
    }

    # 3. Subscribe to result channel FIRST
    pubsub = r_sub.pubsub()
    # AsyncRedisManager publishes to 'socketio' channel by default
    target_channel = "socketio"
    await pubsub.subscribe(target_channel)
    logger.info(f"[Redis PubSub] Subscribed to {target_channel}")

    # 4. Push Task
    logger.info(f"[Redis] Pushing task: {task_payload}")
    await r.rpush("tasks:list", json.dumps(task_payload))
    
    # 5. Wait for result
    logger.info("[Redis PubSub] Waiting for worker response...")
    try:
        async with asyncio.timeout(15):
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True)
                if message:
                    logger.info(f"[Redis PubSub] Raw message: {message}")
                    logger.info("SUCCESS: Received response from worker!")
                    return
                await asyncio.sleep(0.1)
                
    except asyncio.TimeoutError:
        logger.error("FAILURE: Timed out waiting for response.")
        
    await r.close()
    await r_sub.close()

if __name__ == "__main__":
    asyncio.run(run_client())

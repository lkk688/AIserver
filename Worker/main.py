
import asyncio
import logging
import json
import signal
from arq import create_pool
from arq.connections import RedisSettings
from settings import settings
from db import db
from worker_comm import comm
from jobs.llm import process_llm_job

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def startup(ctx):
    logger.info("Worker starting up...")
    await db.connect()
    await comm.connect_redis()
    logger.info("Worker startup complete.")

async def shutdown(ctx):
    logger.info("Worker shutting down...")
    await db.disconnect()
    await comm.close_redis()
    logger.info("Worker shutdown complete.")

# ARQ Worker Settings
class WorkerSettings:
    functions = [process_llm_job]
    redis_settings = RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB
    )
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 10

# Custom Redis Loop for BLPOP (Lightweight tasks)
async def redis_blpop_loop():
    logger.info("Starting Redis BLPOP loop...")
    redis_client = await comm.get_redis()
    
    while True:
        try:
            # BLPOP blocks until a generic item is available
            # Returns tuple (key, value)
            result = await redis_client.blpop([settings.REDIS_TASK_LIST], timeout=5)
            
            if result:
                queue_name, data_raw = result
                data = json.loads(data_raw)
                logger.info(f"Received BLPOP task: {data}")
                
                # Dispatch based on task type
                task_type = data.get("type")
                if task_type == "llm":
                    # For simple tasks, we might process inline or offload to ARQ
                    # Here we simulate inline processing or dispatching to ARQ
                    await process_llm_job(
                        job_id=data.get("job_id"),
                        model=data.get("model"),
                        messages=data.get("messages"),
                        user_id=data.get("user_id")
                    )
            
        except Exception as e:
            logger.error(f"Error in BLPOP loop: {e}")
            await asyncio.sleep(1)

# Custom Redis Loop for XREAD (Stream tasks)
async def redis_stream_loop():
    logger.info("Starting Redis Stream loop...")
    redis_client = await comm.get_redis()
    last_id = "$"
    
    while True:
        try:
            # XREAD blocks until new stream entry
            streams = await redis_client.xread(
                {settings.REDIS_TASK_STREAM: last_id},
                count=1,
                block=5000
            )
            
            if streams:
                for stream_name, messages in streams:
                    for message_id, data in messages:
                        last_id = message_id
                        logger.info(f"Received Stream task {message_id}: {data}")
                         # Process stream task...
                        
        except Exception as e:
            logger.error(f"Error in Stream loop: {e}")
            await asyncio.sleep(1)

async def main():
    # Start ARQ worker
    # Note: Arq normally runs via `arq Worker.main.WorkerSettings` CLI,
    # but we can run it programmatically or alongside other loops.
    # However, arq's `run_worker` is blocking.
    # So we usually run custom loops as background tasks in startup/shutdown of Arq
    # OR we run everything in an asyncio.gather if we build our own runner.
    
    # For simplicity and robust integration, we will run the custom loops
    # as asyncio tasks and then start the arq worker.
    
    # Check if we are running as a script to start everything
    pass

if __name__ == "__main__":
    # If run directly, we can start the custom loops and then the arq worker
    # But arq `run_worker` expects a class.
    # A better pattern for mixed usage:
    # Use `arq` CLI to run the worker, and use `on_startup` to launch background tasks.
    
    # Let's adjust WorkerSettings to launch background tasks
    async def startup_with_loops(ctx):
        await startup(ctx)
        ctx['blpop_task'] = asyncio.create_task(redis_blpop_loop())
        ctx['stream_task'] = asyncio.create_task(redis_stream_loop())
        
    async def shutdown_with_loops(ctx):
        if 'blpop_task' in ctx:
            ctx['blpop_task'].cancel()
        if 'stream_task' in ctx:
            ctx['stream_task'].cancel()
        await shutdown(ctx)

    WorkerSettings.on_startup = startup_with_loops
    WorkerSettings.on_shutdown = shutdown_with_loops
    
    from arq import run_worker
    asyncio.run(run_worker(WorkerSettings))

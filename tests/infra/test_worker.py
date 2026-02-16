
import asyncio
import redis.asyncio as redis
import socketio
import json
import logging
import os
import sys

# Add Worker directory to path
worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Worker'))
if worker_path not in sys.path:
    sys.path.append(worker_path)

# Mock/Patch environment variables BEFORE importing settings
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
# Point to local litellm port
os.environ["LITELLM_API_BASE"] = "http://localhost:4000" 

try:
    from settings import settings
    # Force reload settings if needed, or rely on pydantic reading env
    from jobs.llm import process_llm_job
    from db import db
    from worker_comm import comm
    
    IMPORT_SUCCESS = True
except ImportError as e:
    logging.error(f"Import failed: {e}")
    IMPORT_SUCCESS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_worker")

async def run_worker():
    if not IMPORT_SUCCESS:
        logger.error("Cannot run worker: Imports failed.")
        return

    logger.info("Starting Test Worker (Local Simulation)...")
    logger.info(f"LITELLM_API_BASE: {settings.LITELLM_API_BASE}")
    
    # 1. Initialize Resources
    settings.POSTGRES_HOST = "localhost" # Override for local test
    settings.POSTGRES_PORT = 5432
    
    try:
        await db.connect()
        logger.info("[DB] Connected.")
    except Exception as e:
        logger.warning(f"[DB] Connection failed (non-fatal for simple test): {e}")

    await comm.connect_redis()
    logger.info("[Redis] Connected.")
    
    redis_client = await comm.get_redis()
    
    # 2. Loop for tasks
    logger.info(f"Listening on {settings.REDIS_TASK_LIST}...")
    while True:
        try:
            # BLPOP
            result = await redis_client.blpop([settings.REDIS_TASK_LIST], timeout=5)
            if result:
                queue_name, data_raw = result
                data = json.loads(data_raw)
                logger.info(f"Received task: {data}")
                
                job_id = data.get("job_id")
                user_id = data.get("user_id")
                model = data.get("model")
                messages = data.get("messages")
                
                # Process using REAL logic
                try:
                    await process_llm_job(job_id, model, messages, user_id)
                    logger.info(f"Job {job_id} processed successfully.")
                    
                except Exception as e:
                    logger.error(f"Error processing job: {e}")
                    # Emit failure event
                    await comm.emit_event(f"user_{user_id}", "llm_response", {
                        "job_id": job_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass

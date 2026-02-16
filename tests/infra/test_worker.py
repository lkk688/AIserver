
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
# os.environ["OPENAI_API_KEY"] = "sk-..." # Should use existing env or mock

try:
    from settings import settings
    # Force reload settings if needed, or rely on pydantic reading env
    from jobs.llm import process_llm_job
    from db import db
    from worker_comm import comm
    
    # Patch the comm module to use our local redis manager if needed?
    # Actually, comm.py uses settings.redis_url, which we just pointed to localhost via env.
    # So it should work out of the box!
    
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
    
    # 1. Initialize Resources
    # We might skip DB connect if we don't have local postgres running/configured
    # But user said "redis server is already running". Postgres status unknown.
    # The logs showed "Container postgres Running". So we can try connecting!
    
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
                    # Mock completion if no API key
                    if not settings.OPENAI_API_KEY:
                        logger.warning("No OPENAI_API_KEY found. Mocking litellm completion.")
                        # We can monkeypatch litellm.completion or just catch error
                        # and emit a fake result to satisfy the test client.
                        import jobs.llm
                        original_completion = jobs.llm.completion
                        
                        def mock_completion(**kwargs):
                            class MockResponse:
                                class Choice:
                                    class Message:
                                        content = "Mocked LLM Response for Local Test"
                                    message = Message()
                                choices = [Choice()]
                            return MockResponse()
                        
                        jobs.llm.completion = mock_completion
                    
                    await process_llm_job(job_id, model, messages, user_id)
                    logger.info(f"Job {job_id} processed successfully.")
                    
                except Exception as e:
                    logger.error(f"Error processing job: {e}")
                    
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

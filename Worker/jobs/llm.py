import logging
from litellm import completion
from settings import settings
from db import db
from worker_comm import comm

logger = logging.getLogger(__name__)

async def process_llm_job(job_id: str, model: str, messages: list, user_id: str):
    """
    Process an LLM job:
    1. Call LLM service
    2. Save result to DB
    3. Emit SocketIO event
    """
    logger.info(f"Processing LLM job {job_id} for user {user_id}")
    
    try:
        # 1. Call LLM
        # Using litellm to abstract away provider details
        response = completion(
            model=model,
            messages=messages,
            api_key=settings.OPENAI_API_KEY
        )
        
        content = response.choices[0].message.content
        logger.info(f"LLM Job {job_id} completed successfully")

        # 2. Save to DB (Example schema, adjust as needed)
        # Assuming a table 'chat_messages' exists or similar
        # For now, we'll just log it or insert if we had a definite schema
        # await db.execute("INSERT INTO results (job_id, content) VALUES ($1, $2)", job_id, content)

        # 3. Emit SocketIO event
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="llm_response",
            data={
                "job_id": job_id,
                "status": "completed",
                "content": content
            }
        )
        return content

    except Exception as e:
        logger.error(f"LLM Job {job_id} failed: {e}")
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="llm_error",
            data={
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            }
        )
        raise e

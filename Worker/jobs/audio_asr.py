import os
import logging
import wave
import math
import asyncio
import requests
from litellm import completion
from settings import settings
from db import db
from worker_comm import comm

logger = logging.getLogger(__name__)
WHISPER_URL = "http://100.81.148.35:8001/v1/audio/transcriptions"

async def process_long_audio_job(job_id: str, file_path: str, user_id: str, summarize: bool = True):
    """
    Process a long audio file by chunking it, transcribing via Whisper,
    and optionally summarizing the entire transcript via an LLM.
    """
    logger.info(f"Processing long audio job {job_id} for user {user_id}")
    
    try:
        # 1. Chunked ASR Processing
        if not file_path.lower().endswith('.wav'):
            raise ValueError("Only .wav files are supported for chunked processing in this worker.")
            
        full_text = ""
        chunk_length_ms = 30000 # 30 seconds chunks
        
        with wave.open(file_path, 'r') as w:
            n_channels = w.getnchannels()
            samp_width = w.getsampwidth()
            framerate = w.getframerate()
            
            frames_per_chunk = int(framerate * (chunk_length_ms / 1000.0))
            n_frames = w.getnframes()
            n_chunks = math.ceil(n_frames / frames_per_chunk)
            
            for idx in range(n_chunks):
                chunk_frames = w.readframes(frames_per_chunk)
                chunk_path = f"/tmp/temp_chunk_{job_id}_{idx}.wav"
                
                with wave.open(chunk_path, 'w') as cw:
                    cw.setnchannels(n_channels)
                    cw.setsampwidth(samp_width)
                    cw.setframerate(framerate)
                    cw.writeframes(chunk_frames)
                
                with open(chunk_path, 'rb') as f:
                    files = {'file': (os.path.basename(chunk_path), f, 'audio/wav')}
                    data = {'model': 'large-v3'}
                    headers = {"Authorization": "Bearer ANY_STRING"}
                    
                    # Offload the blocking requests call
                    response = await asyncio.to_thread(
                        requests.post, WHISPER_URL, headers=headers, files=files, data=data, timeout=120
                    )
                    
                os.remove(chunk_path)
                
                if response.status_code == 200:
                    chunk_text = response.json().get('text', '')
                    full_text += chunk_text + " "
                    
                    # Emit progress update back to the client via Socket.IO
                    progress = int((idx + 1) / n_chunks * 100)
                    await comm.emit_event(
                        channel=f"user_{user_id}",
                        event="asr_progress",
                        data={"job_id": job_id, "progress": progress}
                    )
                    logger.info(f"Audio Job {job_id} progress: {progress}%")
                else:
                    raise Exception(f"Whisper API error: {response.text}")
        
        final_text = full_text.strip()
        summary = ""
        
        # 2. Optional Summarization for Meeting Summary
        if summarize and final_text:
            logger.info(f"Summarizing audio transcript for job {job_id}")
            response = completion(
                model="qwen3",
                messages=[
                    {"role": "system", "content": "You are an expert meeting summarizer. Extract the key discussion points, decisions made, and action items from the following transcript."},
                    {"role": "user", "content": final_text}
                ],
                api_base=settings.LITELLM_API_BASE,
                api_key=settings.LITELLM_API_KEY,
                custom_llm_provider="openai"
            )
            summary = response.choices[0].message.content
            
        # 3. Emit Completion Event
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="asr_completed",
            data={
                "job_id": job_id,
                "status": "completed",
                "transcript": final_text,
                "summary": summary
            }
        )
        return {"transcript": final_text, "summary": summary}
        
    except Exception as e:
        logger.error(f"Audio Job {job_id} failed: {e}")
        await comm.emit_event(
            channel=f"user_{user_id}",
            event="asr_error",
            data={"job_id": job_id, "status": "failed", "error": str(e)}
        )
        raise e

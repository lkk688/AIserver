import asyncio
import json
import logging
import uuid
import struct
import time
import copy
import os
from typing import Optional, AsyncGenerator, List, Tuple
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import websockets

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from doubao_protocol_v3 import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

logger = logging.getLogger("jwl.tts")

router = APIRouter()

def _setting_str(attr_name: str, env_name: str, default: str = "") -> str:
    value = os.getenv(env_name, default)
    value_str = str(value or "").strip()
    return value_str or default


def _setting_int(attr_name: str, env_name: str, default: int) -> int:
    value = os.getenv(env_name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


VOLC_APPID = _setting_str("volc_appid", "VOLC_APPID", "")
VOLC_ACCESS_TOKEN = _setting_str("volc_access_token", "VOLC_ACCESS_TOKEN", "")
VOLC_CLUSTER_ID = _setting_str("volc_cluster_id", "VOLC_CLUSTER_ID", "volc.service_type.10029")
TTS_DEFAULT_VOICE = _setting_str("tts_default_voice", "TTS_DEFAULT_VOICE", "zh_female_vv_uranus_bigtts")
TTS_QUEUE_MAXSIZE = max(1, _setting_int("tts_queue_maxsize", "TTS_QUEUE_MAXSIZE", 64))
TTS_MAX_CONCURRENCY = max(1, _setting_int("tts_gpu_or_external_workers", "TTS_GPU_OR_EXTERNAL_WORKERS", 2))

class TTSQueue:
    def __init__(self, max_size: int, max_concurrency: int):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.in_flight = 0
        self.total_processed = 0
        self.total_latency_ms = 0
        
    async def acquire(self):
        # Non-blocking check for queue full happens at caller via queue.put_nowait
        await self.queue.put(time.time()) # Put timestamp
        
    async def worker_context(self):
        """Context manager to acquire semaphore and track in_flight"""
        queued_at = await self.queue.get() # Wait for turn in queue (FIFO)
        
        async with self.semaphore:
            self.in_flight += 1
            start_wait = time.time()
            try:
                yield (queued_at, start_wait)
            finally:
                self.in_flight -= 1
                self.queue.task_done()
                
    def get_stats(self):
        return {
            "depth": self.queue.qsize(),
            "maxsize": self.queue.maxsize,
            "in_flight": self.in_flight,
            "max_concurrency": self.semaphore._value + self.in_flight, # rough estimate
            "avg_latency_ms": int(self.total_latency_ms / max(1, self.total_processed))
        }

    def record_latency(self, ms: float):
        self.total_processed += 1
        self.total_latency_ms += ms

# Initialize Queue
tts_queue = TTSQueue(
    max_size=TTS_QUEUE_MAXSIZE,
    max_concurrency=TTS_MAX_CONCURRENCY
)

# --- Pydantic Models ---

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    format: Optional[str] = "mp3" # mp3, wav, pcm, opus

# --- Doubao Client ---

class DoubaoTTSClient:
    URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
    
    def __init__(self, appid: str, token: str, cluster_id: str):
        self.appid = appid
        self.token = token
        self.cluster_id = cluster_id

    def _get_cluster_for_voice(self, voice: str) -> str:
        """Determine cluster ID based on voice type"""
        # S_ prefix usually indicates Seed-TTS (MegaTTS) voices
        if voice.startswith("S_") or voice.startswith("s_"):
             if "seed" in self.cluster_id or "megatts" in self.cluster_id:
                 return self.cluster_id
             return "volc.megatts.default"
             
        # Standard voices (BV*, zh_female_*, etc)
        # Check if the configured cluster is compatible (Standard or Seed-TTS-2.0)
        # Note: seed-tts-2.0 seems to support standard voices too.
        if "service_type" in self.cluster_id or "seed-tts" in self.cluster_id:
            return self.cluster_id
            
        # Fallback to standard cluster
        return "volc.service_type.10029"

    def _build_attempts(self, voice: str) -> List[Tuple[str, str]]:
        attempts: List[Tuple[str, str]] = []
        seen = set()
        candidate_voices = [
            voice,
            TTS_DEFAULT_VOICE,
            "zh_female_vv_uranus_bigtts",
            "BV700_streaming",
        ]
        for candidate_voice in candidate_voices:
            if not candidate_voice:
                continue
            clusters = [
                self._get_cluster_for_voice(candidate_voice),
                "seed-tts-2.0",
                "volc.service_type.10029",
            ]
            if candidate_voice.startswith("S_") or candidate_voice.startswith("s_"):
                clusters.insert(1, "volc.megatts.default")
            for cluster in clusters:
                key = (candidate_voice, cluster)
                if key not in seen:
                    attempts.append(key)
                    seen.add(key)
        return attempts

    async def stream_audio(self, text: str, voice: str, speed: float, format: str) -> AsyncGenerator[bytes, None]:
        attempts = self._build_attempts(voice)
        last_http_error: Optional[HTTPException] = None

        # Mapping format to encoding
        encoding_map = {
            "mp3": "mp3",
            "wav": "wav", 
            "pcm": "pcm",
            "opus": "opus"
        }
        encoding = encoding_map.get(format, "mp3")
        for attempt_idx, (attempt_voice, target_cluster_id) in enumerate(attempts):
            req_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            headers = {
                "X-Api-App-Key": self.appid,
                "X-Api-Access-Key": self.token,
                "X-Api-Resource-Id": target_cluster_id,
                "X-Api-Connect-Id": req_id
            }
            logger.info(f"Connecting to Doubao TTS {self.URL} with voice={attempt_voice} cluster={target_cluster_id}")
            try:
                async with websockets.connect(
                    self.URL,
                    additional_headers=headers,
                    max_size=10 * 1024 * 1024,
                    ping_interval=None
                ) as ws:
                    base_request = {
                        "user": {
                            "uid": str(uuid.uuid4()),
                        },
                        "namespace": "BidirectionalTTS",
                        "req_params": {
                            "speaker": attempt_voice,
                            "audio_params": {
                                "format": encoding,
                                "sample_rate": 24000,
                                "enable_timestamp": True,
                                "speed_ratio": speed,
                            },
                            "additions": json.dumps(
                                {
                                    "disable_markdown_filter": False,
                                }
                            ),
                        },
                    }
                    await start_connection(ws)
                    await wait_for_event(ws, MsgType.FullServerResponse, EventType.ConnectionStarted)
                    start_session_req = copy.deepcopy(base_request)
                    start_session_req["event"] = EventType.StartSession
                    await start_session(ws, json.dumps(start_session_req).encode(), session_id)
                    await wait_for_event(ws, MsgType.FullServerResponse, EventType.SessionStarted)
                    task_req = copy.deepcopy(base_request)
                    task_req["event"] = EventType.TaskRequest
                    task_req["req_params"]["text"] = text
                    await task_request(ws, json.dumps(task_req).encode(), session_id)
                    await finish_session(ws, session_id)
                    audio_received = False
                    should_retry = False
                    while True:
                        msg = await receive_message(ws)
                        if msg.type == MsgType.FullServerResponse:
                            if msg.event == EventType.SessionFinished:
                                break
                            if msg.event == EventType.TaskFailed:
                                error_detail = msg.payload.decode('utf-8', errors='ignore') if msg.payload else "Unknown TaskFailed"
                                if "resource ID is mismatched with speaker" in error_detail and not audio_received:
                                    should_retry = True
                                    break
                                logger.error(f"TTS Task Failed: {error_detail}")
                                raise HTTPException(status_code=502, detail=f"TTS Task Failed: {error_detail}")
                        elif msg.type == MsgType.AudioOnlyServer:
                            if msg.payload:
                                audio_received = True
                                yield msg.payload
                        elif msg.type == MsgType.Error:
                            error_detail = msg.payload.decode('utf-8', errors='ignore')
                            if "resource ID is mismatched with speaker" in error_detail and not audio_received:
                                should_retry = True
                                break
                            logger.error(f"TTS Error: {msg.error_code} {error_detail}")
                            raise HTTPException(status_code=502, detail=f"TTS Provider Error: {error_detail}")
                    if should_retry:
                        logger.warning(
                            f"TTS voice/cluster mismatch, retrying ({attempt_idx + 1}/{len(attempts)}) "
                            f"voice={attempt_voice} cluster={target_cluster_id}"
                        )
                        await finish_connection(ws)
                        continue
                    if not audio_received:
                        logger.warning("No audio received from TTS provider")
                    await finish_connection(ws)
                    return
            except websockets.exceptions.InvalidStatus as e:
                logger.error(f"WebSocket Error: {e}")
                if "403" in str(e):
                    last_http_error = HTTPException(status_code=403, detail="Upstream TTS Forbidden (Check AppID/Token/Cluster)")
                    break
                last_http_error = HTTPException(status_code=502, detail=f"Upstream TTS Error: {e}")
            except HTTPException as e:
                last_http_error = e
            except Exception as e:
                logger.error(f"TTS Exception: {e}")
                last_http_error = HTTPException(status_code=500, detail=str(e))
            if attempt_idx < len(attempts) - 1:
                continue
        raise last_http_error or HTTPException(status_code=502, detail="TTS Provider Error")

# --- Routes ---

@router.post("/stream")
async def stream_tts(request: TTSRequest):
    """
    Stream TTS audio from Doubao (Volcengine).
    Handles concurrency via queue.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    if len(request.text) > 4096:
         raise HTTPException(status_code=400, detail="Text too long (max 4096 chars)")

    # Credentials Check
    if not VOLC_APPID or not VOLC_ACCESS_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="TTS unavailable: missing VOLC_APPID or VOLC_ACCESS_TOKEN",
        )

    # Voice Fallback
    voice = request.voice
    if not voice:
        voice = TTS_DEFAULT_VOICE
    elif voice.startswith('af_') or voice.startswith('am_') or voice.startswith('bf_') or voice.startswith('bm_'): 
        # Map Kokoro English voices to Doubao English Voice
        # BV406_streaming: English Female Emotional (Vienna)
        # BV407_streaming: English Male Emotional (Stanley)
        voice = "BV406_streaming" 
    elif voice.startswith('zf_'):
        voice = TTS_DEFAULT_VOICE
    
    # Format Content-Type
    content_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "opus": "audio/opus"
    }
    content_type = content_types.get(request.format, "audio/mpeg")

    # Queue Check
    try:
        tts_queue.queue.put_nowait(time.time())
    except asyncio.QueueFull:
         stats = tts_queue.get_stats()
         raise HTTPException(
             status_code=503, 
             detail={
                 "error": "busy", 
                 "queue_depth": stats['depth'],
                 "estimated_wait_sec": (stats['depth'] * 2) # Rough estimate
             }
         )

    async def audio_generator():
        # Acquire Semaphore (Worker Slot)
        queued_at = await tts_queue.queue.get()
        
        start_process = time.time()
        queued_ms = (start_process - queued_at) * 1000
        
        try:
            async with tts_queue.semaphore:
                tts_queue.in_flight += 1
                
                client = DoubaoTTSClient(
                    appid=VOLC_APPID,
                    token=VOLC_ACCESS_TOKEN,
                    cluster_id=VOLC_CLUSTER_ID
                )
                
                first_byte_time = None
                first_chunk_sent = False
                
                try:
                    async for chunk in client.stream_audio(request.text, voice, request.speed, request.format):
                        if first_byte_time is None:
                            first_byte_time = time.time()
                        first_chunk_sent = True
                        yield chunk
                except HTTPException as e:
                    logger.error(f"Streaming error: {e.detail}")
                    if not first_chunk_sent:
                        raise e
                    return
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    if not first_chunk_sent:
                        raise HTTPException(status_code=500, detail=str(e))
                    return
                finally:
                    total_time = time.time() - start_process
                    tts_queue.record_latency(total_time * 1000)
                    
                    if first_byte_time:
                         ttfb = (first_byte_time - start_process) * 1000
                         logger.info(f"TTS Complete: voice={voice} queued_ms={int(queued_ms)} ttfb_ms={int(ttfb)} total_ms={int(total_time*1000)}")
                    else:
                         logger.warning(f"TTS Failed/Empty: voice={voice} queued_ms={int(queued_ms)}")

        finally:
            tts_queue.in_flight -= 1
            tts_queue.queue.task_done()

    return StreamingResponse(
        audio_generator(), 
        media_type=content_type,
        headers={
            "Cache-Control": "no-store",
            "X-Request-Id": str(uuid.uuid4())
        }
    )

@router.get("/queue")
async def get_queue_status():
    return tts_queue.get_stats()

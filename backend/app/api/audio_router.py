import os
import time
import requests
import wave
import math
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

router = APIRouter(prefix="/audio", tags=["Audio"])

SOVITS_URL = "http://100.81.148.35:8002/"
WHISPER_URL = "http://100.81.148.35:8001/v1/audio/transcriptions"

class TTSRequest(BaseModel):
    text: str
    backend: str = "sovits"  # "sovits" or "doubao"
    voice: Optional[str] = "zh_female_vv_uranus_bigtts" # For Doubao
    language: Optional[str] = "zh" # For SoVITS
    refer_wav: Optional[str] = None # For SoVITS
    prompt_text: Optional[str] = None # For SoVITS
    prompt_lang: Optional[str] = "zh" # For SoVITS

@router.post("/tts")
async def generate_tts(request: TTSRequest):
    if request.backend == "sovits":
        payload = {
            "text": request.text,
            "text_language": request.language
        }
        if request.refer_wav:
            payload["refer_wav_path"] = request.refer_wav
        if request.prompt_text:
            payload["prompt_text"] = request.prompt_text
        if request.prompt_lang:
            payload["prompt_language"] = request.prompt_lang
            
        try:
            # Using run_in_executor to not block the async event loop
            response = await asyncio.to_thread(
                requests.post, SOVITS_URL, json=payload, timeout=60
            )
            if response.status_code == 200:
                return Response(content=response.content, media_type="audio/wav")
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SoVITS Error: {str(e)}")
            
    elif request.backend == "doubao":
        try:
            import sys
            audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'Audio')
            if audio_dir not in sys.path:
                sys.path.insert(0, audio_dir)
            from tts_doubao import stream_tts, TTSRequest as DoubaoTTSRequest
            
            req = DoubaoTTSRequest(text=request.text, voice=request.voice, format="mp3")
            response = await stream_tts(req)
            return response # This is a StreamingResponse from tts_doubao
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Doubao TTS Error: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid backend specified")

@router.post("/asr")
async def generate_asr(
    file: UploadFile = File(...),
    mode: str = Form("short"),
    chunk_sec: int = Form(30),
    model: str = Form("large-v3"),
    token: str = Form("ANY_STRING")
):
    # Save the uploaded file temporarily
    temp_file = f"temp_{int(time.time())}_{file.filename}"
    try:
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        headers = {"Authorization": f"Bearer {token}"}
        
        if mode == "short":
            with open(temp_file, 'rb') as f:
                files = {'file': (file.filename, f, 'audio/wav')}
                data = {'model': model}
                response = await asyncio.to_thread(
                    requests.post, WHISPER_URL, headers=headers, files=files, data=data, timeout=120
                )
            if response.status_code == 200:
                return {"text": response.json().get('text', '')}
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
        elif mode == "chunked":
            if not temp_file.lower().endswith('.wav'):
                raise HTTPException(status_code=400, detail="Chunked mode requires .wav file format")
                
            full_text = ""
            chunk_length_ms = chunk_sec * 1000
            
            with wave.open(temp_file, 'r') as w:
                n_channels = w.getnchannels()
                samp_width = w.getsampwidth()
                framerate = w.getframerate()
                
                frames_per_chunk = int(framerate * (chunk_length_ms / 1000.0))
                n_frames = w.getnframes()
                n_chunks = math.ceil(n_frames / frames_per_chunk)
                
                for idx in range(n_chunks):
                    chunk_frames = w.readframes(frames_per_chunk)
                    chunk_path = f"temp_chunk_{idx}.wav"
                    
                    with wave.open(chunk_path, 'w') as cw:
                        cw.setnchannels(n_channels)
                        cw.setsampwidth(samp_width)
                        cw.setframerate(framerate)
                        cw.writeframes(chunk_frames)
                    
                    with open(chunk_path, 'rb') as f:
                        files = {'file': (chunk_path, f, 'audio/wav')}
                        data = {'model': model}
                        response = await asyncio.to_thread(
                            requests.post, WHISPER_URL, headers=headers, files=files, data=data, timeout=120
                        )
                        
                    os.remove(chunk_path)
                    
                    if response.status_code == 200:
                        full_text += response.json().get('text', '') + " "
                    else:
                        raise HTTPException(status_code=response.status_code, detail=response.text)
            
            return {"text": full_text.strip()}
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
            
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

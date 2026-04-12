import argparse
import time
import os
import requests
import wave
import math

def transcribe_audio(file_path, url, model, headers):
    start_time = time.time()
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            data = {'model': model}
            response = requests.post(url, headers=headers, files=files, data=data)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            return response.json().get('text', ''), latency
        else:
            print(f"\nError: API returned status {response.status_code}")
            print(response.text)
            return "", latency
    except Exception as e:
        print(f"\nConnection error: {e}")
        return "", time.time() - start_time

def get_audio_duration(audio_path):
    with wave.open(audio_path, 'r') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        return frames / float(rate)

def process_short(audio_path, url, model, headers):
    try:
        duration_sec = get_audio_duration(audio_path)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return
    
    print(f"Transcribing short audio (Duration: {duration_sec:.2f}s)...")
    text, latency = transcribe_audio(audio_path, url, model, headers)
    
    rtf = latency / duration_sec if duration_sec > 0 else 0
    print("\n--- Transcription ---")
    print(text)
    print("---------------------")
    print(f"Metrics:\n- Latency: {latency:.2f}s\n- Audio Duration: {duration_sec:.2f}s\n- RTF (Real Time Factor): {rtf:.2f}x (Lower is better)")

def process_chunked(audio_path, url, model, headers, chunk_length_ms=30000):
    try:
        duration_sec = get_audio_duration(audio_path)
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return
        
    print(f"Transcribing audio in chunks (Duration: {duration_sec:.2f}s, Chunk size: {chunk_length_ms/1000}s)...")
    
    full_text = ""
    total_latency = 0
    
    # Chunk audio using wave module
    with wave.open(audio_path, 'r') as w:
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
            
            chunk_duration = len(chunk_frames) / (n_channels * samp_width * framerate)
            print(f"Processing chunk {idx+1}/{n_chunks} ({chunk_duration:.2f}s)... ", end="", flush=True)
            
            text, latency = transcribe_audio(chunk_path, url, model, headers)
            full_text += text + " "
            total_latency += latency
            
            print(f"Done in {latency:.2f}s")
            os.remove(chunk_path)
        
    rtf = total_latency / duration_sec if duration_sec > 0 else 0
    print("\n--- Full Transcription ---")
    print(full_text.strip())
    print("--------------------------")
    print(f"Metrics:\n- Total Processing Time: {total_latency:.2f}s\n- Audio Duration: {duration_sec:.2f}s\n- RTF (Real Time Factor): {rtf:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust ASR CLI tool for local/remote Whisper endpoints")
    parser.add_argument("file", help="Path to audio file (.wav format only)")
    parser.add_argument("--url", default="http://100.81.148.35:8001/v1/audio/transcriptions", help="ASR API URL endpoint")
    parser.add_argument("--model", default="large-v3", help="Model name to use")
    parser.add_argument("--token", default="ANY_STRING", help="Authorization token")
    parser.add_argument("--mode", choices=["short", "chunked"], default="short", help="Processing mode (short=whole file at once, chunked=split file for streaming/long audio)")
    parser.add_argument("--chunk-sec", type=int, default=30, help="Chunk length in seconds for chunked mode")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        exit(1)
        
    if not args.file.lower().endswith('.wav'):
        print("Error: Only .wav files are supported by this standalone script. Convert your file to .wav first.")
        exit(1)
        
    headers = {"Authorization": f"Bearer {args.token}"}
    
    if args.mode == "short":
        process_short(args.file, args.url, args.model, headers)
    elif args.mode == "chunked":
        process_chunked(args.file, args.url, args.model, headers, chunk_length_ms=args.chunk_sec * 1000)

"""
# 1. Transcribe a short audio file in one go (Default)
python asr_client.py test_audio.wav --mode short

# 2. Transcribe a long meeting file by chunking it (e.g. 30 second chunks)
python asr_client.py my_meeting.wav --mode chunked --chunk-sec 30

# 3. Specify a different model or token
python asr_client.py my_audio.wav --model "large-v3" --token "my_auth_token"
"""
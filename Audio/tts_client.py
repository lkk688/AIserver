import argparse
import time
import os
import sys
import requests
import asyncio
import wave

def get_audio_duration(file_path):
    """Attempt to get audio duration in seconds. Works natively for .wav."""
    if file_path.lower().endswith('.wav'):
        try:
            with wave.open(file_path, 'r') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                return frames / float(rate)
        except Exception:
            return None
    return None

def print_metrics(output_file, latency):
    duration = get_audio_duration(output_file)
    print("\n--- Metrics ---")
    print(f"Latency: {latency:.2f}s")
    if duration:
        rtf = latency / duration if duration > 0 else 0
        print(f"Audio Duration: {duration:.2f}s")
        print(f"RTF (Real Time Factor): {rtf:.2f}x (Lower is better)")
    else:
        print("Audio Duration: N/A (Only calculated for .wav natively)")
    print("---------------")

def tts_sovits(text, url, output_file, language, refer_wav, prompt_text, prompt_lang):
    print(f"Synthesizing with GPT-SoVITS backend: {url}")
    start_time = time.time()
    
    payload = {
        "text": text,
        "text_language": language
    }
    
    if refer_wav:
        payload["refer_wav_path"] = refer_wav
    if prompt_text:
        payload["prompt_text"] = prompt_text
    if prompt_lang:
        payload["prompt_language"] = prompt_lang
        
    try:
        response = requests.post(url, json=payload)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"\nSuccessfully generated audio: {output_file}")
            print_metrics(output_file, latency)
        else:
            print(f"\nError: API returned status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\nConnection error: {e}")

async def tts_doubao(text, output_file, voice):
    print(f"Synthesizing with Doubao TTS backend...")
    start_time = time.time()
    
    # Add Audio to sys.path to import tts_doubao
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Audio'))
    try:
        from tts_doubao import stream_tts, TTSRequest
    except ImportError as e:
        print(f"Error importing tts_doubao: {e}")
        return
        
    # Infer format from extension
    fmt = "mp3"
    if output_file.lower().endswith(".wav"):
        fmt = "wav"
    elif output_file.lower().endswith(".pcm"):
        fmt = "pcm"
    elif output_file.lower().endswith(".opus"):
        fmt = "opus"
        
    req = TTSRequest(text=text, voice=voice, format=fmt)
    try:
        response = await stream_tts(req)
        
        audio_data = bytearray()
        async for chunk in response.body_iterator:
            audio_data.extend(chunk)
            
        latency = time.time() - start_time
        if audio_data:
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"\nSuccessfully generated {len(audio_data)} bytes of audio: {output_file}")
            print_metrics(output_file, latency)
        else:
            print("\nFailed to generate audio.")
    except Exception as e:
        print(f"\nError during Doubao TTS: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust TTS CLI tool for GPT-SoVITS and Doubao backends")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--backend", choices=["sovits", "doubao"], default="sovits", help="TTS backend to use")
    parser.add_argument("--output", default="output.wav", help="Output audio file path (.wav or .mp3)")
    
    # SoVITS arguments
    parser.add_argument("--url", default="http://100.81.148.35:8002/", help="SoVITS API URL endpoint")
    parser.add_argument("--language", default="zh", help="Text language for SoVITS")
    parser.add_argument("--refer-wav", help="Path to reference wav (inside container for SoVITS)")
    parser.add_argument("--prompt-text", help="Prompt text for reference wav")
    parser.add_argument("--prompt-lang", default="zh", help="Prompt language")
    
    # Doubao arguments
    parser.add_argument("--voice", default="zh_female_vv_uranus_bigtts", help="Voice for Doubao")
    
    args = parser.parse_args()
    
    if args.backend == "sovits":
        tts_sovits(args.text, args.url, args.output, args.language, args.refer_wav, args.prompt_text, args.prompt_lang)
    elif args.backend == "doubao":
        asyncio.run(tts_doubao(args.text, args.output, args.voice))

"""
# Basic usage
python tts_client.py "你好，我是部署在1080Ti上的语音助手，很高兴为你服务！" --backend sovits --output my_sovits.wav

# With a specific reference voice inside the container
python tts_client.py "这是克隆测试" --backend sovits --refer-wav "/workspace/GPT-SoVITS/output/myvoice.wav" --prompt-text "我的声音"



# Basic usage (defaults to voice: zh_female_vv_uranus_bigtts)
python tts_client.py "你好，我是豆包语音助手，很高兴为你服务！" --backend doubao --output my_doubao.mp3

# Specify a different voice preset
python tts_client.py "你好" --backend doubao --voice "S_v2_zh_female_emotion" --output custom_doubao.wav
"""
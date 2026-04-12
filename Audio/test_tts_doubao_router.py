import asyncio
from tts_doubao import stream_tts, TTSRequest

async def main():
    req = TTSRequest(text="你好，这是通过 FastAPI Router 测试的豆包语音合成。", voice="zh_female_vv_uranus_bigtts")
    try:
        response = await stream_tts(req)
        
        # Read the streaming response
        audio_data = bytearray()
        async for chunk in response.body_iterator:
            audio_data.extend(chunk)
            
        if audio_data:
            with open("test_router_doubao.mp3", "wb") as f:
                f.write(audio_data)
            print(f"Successfully generated {len(audio_data)} bytes of audio.")
        else:
            print("Failed to generate audio.")
    except Exception as e:
        print(f"Error during TTS: {e}")

if __name__ == "__main__":
    asyncio.run(main())

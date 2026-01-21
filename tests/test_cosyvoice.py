import requests
import json
import os

def test_cosyvoice(base_url="http://localhost:50000", output_file="test_audio.wav"):
    print(f"\n--- Testing CosyVoice at {base_url} ---")
    
    url = f"{base_url}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-dummy"
    }
    data = {
        "model": "cosyvoice",
        "input": "Hello, this is a test of the CosyVoice text to speech generation.",
        "voice": "中文女",
        "response_format": "wav",
        "speed": 1.0
    }

    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Success! Audio saved to {output_file}")
        else:
            print(f"Error: Status Code {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Test direct access to CosyVoice container
    test_cosyvoice("http://localhost:50000", "test_cosyvoice_direct.wav")
    
    # Test via LiteLLM proxy
    # test_cosyvoice("http://localhost:4000", "test_cosyvoice_proxy.wav")

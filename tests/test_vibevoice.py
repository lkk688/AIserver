import requests
import json
import os

def test_vibevoice(base_url="http://localhost:50001", output_file="test_vibevoice.wav"):
    print(f"\n--- Testing VibeVoice at {base_url} ---")
    
    url = f"{base_url}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-dummy"
    }
    data = {
        "model": "vibevoice",
        "input": "Hello, this is a test of the VibeVoice text to speech generation.",
        "voice": "en-WHTest_man",
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
    # Test direct access to VibeVoice container
    test_vibevoice("http://localhost:50001", "test_vibevoice_direct.wav")
    
    # Test via LiteLLM proxy (if configured)
    # test_vibevoice("http://localhost:4000", "test_vibevoice_proxy.wav")

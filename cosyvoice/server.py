import argparse
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
import torch
# Hack to bypass torchaudio CUDA version check which fails due to vLLM's internal PyTorch version (CUDA 12.8) vs public Torchaudio (CUDA 12.9)
if hasattr(torch.version, 'cuda'):
    original_cuda_version = torch.version.cuda
    # We set it to 12.9 to satisfy torchaudio's check
    torch.version.cuda = "12.9"
    try:
        import torchaudio
    finally:
        # Restore original version
        torch.version.cuda = original_cuda_version
else:
    import torchaudio

import io
import sys

# Ensure sys.path includes the installed package location if needed, 
# but usually pip install handles it.
# The error "ModuleNotFoundError: No module named 'cosyvoice.cli'" implies 
# cosyvoice structure might be different in the installed version or git clone.

# We'll try to import CosyVoice generically
try:
    from cosyvoice.cli.cosyvoice import CosyVoice, AutoModel
except ImportError:
    # If installed via git/pip, structure might be top-level
    try:
        from cosyvoice.utils.file_utils import load_wav
        # Dummy AutoModel for now if we can't find it, or print error
        print("Warning: Could not import CosyVoice from standard path. Checking alternatives.")
    except ImportError:
        pass

from modelscope import snapshot_download

app = FastAPI()
cosyvoice_model = None

class SpeechRequest(BaseModel):
    model: str = "cosyvoice"
    input: str
    voice: str = "中文女" # Default voice
    speed: float = 1.0
    response_format: Optional[str] = "wav"

@app.on_event("startup")
async def startup_event():
    global cosyvoice_model
    args = parse_args()
    
    # Download model if not exists
    # We use Fun-CosyVoice3-0.5B-2512 as default if not specified
    model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" 
    local_dir = args.model_dir
    
    if not os.path.exists(local_dir):
        print(f"Downloading model {model_id} to {local_dir}...")
        try:
            snapshot_download(model_id, local_dir=local_dir)
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Fallback or exit
    
    print(f"Loading CosyVoice model from {local_dir} with vLLM...")
    # Initialize model with vLLM backend
    # Note: fp16=True is common for GPUs
    # CosyVoice V3 doesn't seem to use AutoModel in the same way or import path changed
    # Based on search results, usage is:
    # cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)
    # or for V3/vLLM:
    # cosyvoice = AutoModel(model_dir=..., load_vllm=True)
    # If AutoModel fails, fallback to CosyVoice class
    
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        cosyvoice_model = AutoModel(
            model_dir=local_dir,
            load_trt=False,
            load_vllm=True, 
            fp16=True
        )
    except ImportError:
        # Fallback if AutoModel is not exposed or import path differs
        # This handles the 'ModuleNotFoundError: No module named 'cosyvoice.cli'' if structure is different
        # But wait, the error said `from cosyvoice.cli.cosyvoice import AutoModel` failed.
        # Let's try importing CosyVoice directly
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceCls
        cosyvoice_model = CosyVoiceCls(
            model_dir=local_dir,
            load_jit=False, 
            load_trt=False,
            fp16=True
        )

    print("Model loaded successfully.")

@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    if not cosyvoice_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.input
    voice = request.voice
    
    print(f"Generating speech for: {text[:20]}... with voice: {voice}")
    
    try:
        # Use inference_sft for standard TTS with a specific speaker
        # Result is a generator, we take the first result (or join them if streaming)
        # CosyVoice returns {'tts_speech': tensor, ...}
        
        # Note: 'voice' needs to be a valid speaker ID in the model.
        # Check available speakers: cosyvoice_model.list_avaliable_spks()
        
        output = None
        # We handle simple SFT (Supervised Fine-Tuning) inference here
        # Assuming voice matches one of the SFT speakers
        # If voice is not found, it might error or we should fallback.
        
        # Simple blocking inference (not streaming for this API endpoint)
        results = cosyvoice_model.inference_sft(text, voice, stream=False)
        
        # inference_sft returns a generator
        for res in results:
            output = res['tts_speech']
            break # Take first (and likely only) chunk for non-streaming
            
        if output is None:
             raise HTTPException(status_code=500, detail="Generation failed")

        # Convert tensor to wav bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, output, 22050, format="wav")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/Fun-CosyVoice3-0.5B")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

import argparse
import uvicorn
import os
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
import torch
# Hack removed: we now ensure torch and torchaudio versions match via Dockerfile
# if hasattr(torch.version, 'cuda'):
#    original_cuda_version = torch.version.cuda
#    # We set it to 12.9 to satisfy torchaudio's check
#    torch.version.cuda = "12.9"
#    try:
#        import torchaudio
#    finally:
#        # Restore original version
#        torch.version.cuda = original_cuda_version
# else:
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
    # We use FunAudioLLM/CosyVoice-300M-SFT as default for SFT (Speaker Fine-Tuning) support
    # V3 (Fun-CosyVoice3-0.5B-2512) is a base model primarily for zero-shot and lacks pre-defined speakers.
    model_id = "FunAudioLLM/CosyVoice-300M-SFT" 
    local_dir = args.model_dir
    
    # If user manually specified a different model dir (e.g. V3), use that ID for download check
    if "Fun-CosyVoice3" in local_dir:
        model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    
    if not os.path.exists(local_dir):
        print(f"Downloading model {model_id} to {local_dir}...")
        try:
            snapshot_download(model_id, local_dir=local_dir)
        except Exception as e:
            print(f"Failed to download model: {e}")
            # Fallback or exit
    
    print(f"Loading CosyVoice model from {local_dir} with vLLM...")
    # Initialize model with vLLM backend
    # Note: fp16=True is common for GPUs, but for CPU we must disable it.
    
    # Check if we have a GPU available for torch
    is_cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_cuda_available}")
    
    # Check if vLLM is available
    try:
        import vllm
        vllm_available = True
    except ImportError:
        vllm_available = False
        print("vLLM not installed. Disabling vLLM acceleration.")

    # If no GPU, force load_vllm=False because vLLM usually requires GPU or specific CPU build
    # Also CosyVoice AutoModel might default to vLLM if available in environment.
    
    # However, the user environment seems to have vLLM installed (from base image), but running on CPU.
    # The error "ImportError: cannot import name 'ALLOWED_LAYER_TYPES' from 'transformers.configuration_utils'"
    # suggests vLLM might be failing to import correctly or incompatible with installed transformers.
    
    # Let's try to fall back to standard CosyVoice class (Torch/ONNX) if AutoModel fails or if we are on CPU.
    # The traceback shows AutoModel failing inside load_vllm.
    
    use_vllm = is_cuda_available and vllm_available # Only use vLLM if GPU is present AND vLLM is installed
    
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
        
        # If we are on CPU, we might need to avoid AutoModel if it forces vLLM/TRT checks that fail
        # But let's try with load_vllm=False
        
        cosyvoice_model = AutoModel(
            model_dir=local_dir,
            load_trt=False,
            load_vllm=use_vllm, 
            fp16=is_cuda_available
        )
    except Exception as e:
        print(f"AutoModel failed: {e}. Falling back to standard CosyVoice class.")
        # Fallback 
        from cosyvoice.cli.cosyvoice import CosyVoice as CosyVoiceCls
        cosyvoice_model = CosyVoiceCls(
            model_dir=local_dir,
            load_jit=False, 
            load_trt=False,
            fp16=is_cuda_available
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
        # Check available speakers: cosyvoice_model.list_available_spks()
        
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
        # CosyVoice V3 sample rate might be 24000 or 22050, usually 22050 for V1/V2 but V3 might differ.
        # Safe to assume 22050 unless specified otherwise.
        torchaudio.save(buffer, output.cpu(), 22050, format="wav")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during generation: {e}")
        # If it's a key error, it likely means the voice is invalid
        if isinstance(e, KeyError) or "is not in list" in str(e) or "not found" in str(e).lower():
             available_spks = []
             try:
                 available_spks = cosyvoice_model.list_available_spks()
                 print(f"Available speakers: {available_spks}")
             except:
                 pass
             
             if not available_spks and hasattr(cosyvoice_model, 'frontend'):
                  # Fallback to check frontend spk2info
                  try:
                      available_spks = list(cosyvoice_model.frontend.spk2info.keys())
                      print(f"Available speakers (from frontend): {available_spks}")
                  except:
                      pass

             raise HTTPException(status_code=400, detail=f"Invalid voice '{voice}'. Available: {available_spks[:10]}...")
        
        raise HTTPException(status_code=500, detail=str(e))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice-300M-SFT")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

import os
import io
import base64
import time
import uuid
import asyncio
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal
import uvicorn
import json
from translategemma import TranslateGemmaModel
from gemma3 import Gemma3Model
from medgemma import MedGemmaModel
from embedding import EmbeddingModel

app = FastAPI(title="HFserve OpenAI-compatible API")

# Global instances
translategemma_model = TranslateGemmaModel()
gemma3_model = Gemma3Model()
medgemma_model = MedGemmaModel()
embedding_model = EmbeddingModel()

# Global state
MODEL_READY = False
model_types = os.getenv("MODEL_TYPE", "all").lower().split(",")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@app.on_event("startup")
async def startup_event():
    global MODEL_READY
    print("Starting up services...")
    print(f"Selected model types: {model_types}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    try:
        if "translategemma" in model_types or "all" in model_types:
            translategemma_model.load()
        if "gemma3" in model_types or "all" in model_types:
            gemma3_model.load()
        if "medgemma" in model_types or "all" in model_types:
            medgemma_model.load()
        if "embedding" in model_types or "all" in model_types:
            embedding_model.load()
        MODEL_READY = True
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {e}")
        MODEL_READY = False

# --- OpenAI Compatible Data Models ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]] # content can be string or list of content parts

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

# --- Embedding Data Models ---

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float" # float or base64

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: UsageInfo

# --- Helper Functions ---

def load_image(image_source: str) -> Image.Image:
    """
    Load image from URL or Base64 string.
    """
    try:
        if image_source.startswith("http://") or image_source.startswith("https://"):
            print(f"Downloading image from URL: {image_source}")
            response = requests.get(image_source, headers={"User-Agent": "HFserve"}, stream=True)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")
        
        if "base64," in image_source:
            _, base64_data = image_source.split("base64,")
            image_data = base64.b64decode(base64_data)
        else:
            try:
                image_data = base64.b64decode(image_source)
            except Exception:
                raise ValueError(f"Invalid image source. Must be HTTP URL or Base64 string.")
        
        return Image.open(io.BytesIO(image_data)).convert("RGB")
            
    except Exception as e:
        print(f"Error loading image: {e}")
        raise ValueError(f"Failed to load image: {str(e)}")

def process_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI ChatMessages to internal format expected by models.
    Handles image loading from content parts.
    """
    processed_messages = []
    for msg in messages:
        msg_dict = {
            "role": msg.role,
            "content": []
        }
        
        if isinstance(msg.content, str):
            msg_dict["content"].append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        msg_dict["content"].append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image_url":
                        # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                        url_obj = item.get("image_url", {})
                        url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
                        if url:
                            try:
                                image_obj = load_image(url)
                                msg_dict["content"].append({"type": "image", "image": image_obj})
                            except Exception as e:
                                print(f"Warning: Failed to load image from {url}: {e}")
        
        processed_messages.append(msg_dict)
    return processed_messages

# --- Endpoints ---

@app.get("/healthz")
async def healthz():
    """Returns 200 if process is alive."""
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    """Returns 200 only when model is loaded."""
    if MODEL_READY:
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    if not embedding_model.model:
         raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # Concurrency limiting
    try:
        async with asyncio.timeout(0.1): 
            await request_semaphore.acquire()
    except TimeoutError:
         raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again later.")
    
    try:
        loop = asyncio.get_running_loop()
        # embedding_model.embed is synchronous
        embeddings = await loop.run_in_executor(
            None, 
            lambda: embedding_model.embed(request.input)
        )
        
        data = []
        for i, emb in enumerate(embeddings):
            data.append(EmbeddingData(embedding=emb, index=i))
            
        return EmbeddingResponse(
            data=data,
            model=embedding_model.model_name,
            usage=UsageInfo(prompt_tokens=0, total_tokens=0)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_semaphore.release()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not ready")

    # Concurrency limiting
    try:
        # Use a small timeout to try acquiring the semaphore. 
        # If we can't get a slot within 0.1s, we assume the queue/capacity is full.
        async with asyncio.timeout(0.1): 
            await request_semaphore.acquire()
    except TimeoutError:
         raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again later.")
    
    try:
        # Route to appropriate model
        model_name = request.model.lower()
        generated_text = ""
        
        # Helper to run generation in thread pool to not block event loop
        # (Assuming generate methods are blocking CPU bound)
        # The existing code called them directly. 
        # Ideally we should use run_in_executor.
        
        processed_msgs = process_openai_messages(request.messages)
        
        # Select model instance
        target_model = None
        if "gemma" in model_name and "translate" in model_name:
             target_model = translategemma_model
        elif "med" in model_name:
             target_model = medgemma_model
        elif "gemma" in model_name:
             target_model = gemma3_model
        else:
             # Default to gemma3 if loaded, or try others
             target_model = gemma3_model
             
        if not target_model or not target_model.model:
             # Fallback logic or error
             if gemma3_model.model: target_model = gemma3_model
             elif translategemma_model.model: target_model = translategemma_model
             elif medgemma_model.model: target_model = medgemma_model
             else:
                 raise HTTPException(status_code=503, detail="No models loaded")

        # Run generation
        # Note: generate methods in *gemma.py files seem synchronous.
        loop = asyncio.get_running_loop()
        generated_text = await loop.run_in_executor(
            None, 
            lambda: target_model.generate(processed_msgs, request.max_tokens or 200)
        )

        # Construct response
        if request.stream:
            async def text_streamer():
                # Simulate streaming by splitting by space
                # Simple tokenization simulation
                tokens = generated_text.split(" ")
                req_id = f"chatcmpl-{uuid.uuid4()}"
                created_ts = int(time.time())
                
                # Initial chunk
                init_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(init_chunk)}\n\n"
                
                for i, word in enumerate(tokens):
                    # add space if not first word
                    content = " " + word if i > 0 else word
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # simulate delay
                    await asyncio.sleep(0.02)
                
                # Final chunk
                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(text_streamer(), media_type="text/event-stream")

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_semaphore.release()

# Keep original endpoints for backward compatibility if needed, or remove them.
# The user instruction didn't explicitly say "delete old endpoints", but "HFserve must expose an OpenAI-compatible API".
# I'll comment them out or keep them as deprecated to keep file clean if I can.
# Given the instructions, I'll keep the file clean and replace the old logic with this new structure, 
# BUT I must ensure I don't break "TranslateGemma" or "MedGemma" specialized logic if they rely on specific params.
# The new API is generic. 
# I will KEEP the old endpoints for safety, but the "v1/chat/completions" is the main one for integration.

class ContentItem(BaseModel):
    type: str  # "text" or "image"
    source_lang_code: Optional[str] = None
    target_lang_code: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None
    image: Optional[str] = None 

class Message(BaseModel):
    role: str
    content: List[ContentItem]

class InferenceRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 200

# Helper for legacy endpoints to reuse load_image (already defined above)
def process_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    # ... implementation of legacy processing ...
    processed_messages = []
    for msg in messages:
        msg_dict = {
            "role": msg.role,
            "content": []
        }
        for item in msg.content:
            item_dict = item.dict(exclude_none=True)
            if item.type == "image":
                image_source = item.image or item.url
                if image_source:
                    try:
                        image_obj = load_image(image_source)
                        item_dict["image"] = image_obj
                        if "url" in item_dict: del item_dict["url"]
                    except Exception as e:
                        print(f"Warning: Failed to process image: {e}")
            msg_dict["content"].append(item_dict)
        processed_messages.append(msg_dict)
    return processed_messages

@app.post("/translategemma/translate")
async def translate_gemma(request: InferenceRequest):
    if not translategemma_model.model:
        raise HTTPException(status_code=503, detail="TranslateGemma model not loaded")
    try:
        messages = process_messages(request.messages)
        result = translategemma_model.generate(messages, request.max_new_tokens)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gemma3/chat")
async def chat_gemma3(request: InferenceRequest):
    if not gemma3_model.model:
        raise HTTPException(status_code=503, detail="Gemma 3 model not loaded")
    try:
        messages = process_messages(request.messages)
        result = gemma3_model.generate(messages, request.max_new_tokens)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/medgemma/chat")
async def chat_medgemma(request: InferenceRequest):
    if not medgemma_model.model:
        raise HTTPException(status_code=503, detail="MedGemma model not loaded")
    try:
        messages = process_messages(request.messages)
        result = medgemma_model.generate(messages, request.max_new_tokens)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


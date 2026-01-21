import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import os
from PIL import Image
import requests

class Gemma3Model:
    def __init__(self):
        self.MODEL_ID = "google/gemma-3-4b-it"
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading model {self.MODEL_ID}...")
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=token)
            
            # Load model with 4-bit quantization to save memory
            # Even though it's 4B, using 4-bit allows running alongside 27B model more easily
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.MODEL_ID, 
                device_map="auto", 
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                token=token
            ).eval()
            
            print("Gemma 3 Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Gemma 3 model: {e}")
            raise e

    def generate(self, messages, max_new_tokens=200):
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        # Convert messages to list of dicts for apply_chat_template
        messages_data = []
        for msg in messages:
            if isinstance(msg, dict):
                messages_data.append(msg)
            else:
                # Assume Pydantic model structure
                messages_data.append({
                    "role": msg.role,
                    "content": [item.dict(exclude_none=True) for item in msg.content]
                })

        # Apply chat template
        # Try to print image info for debugging
        for msg in messages_data:
            for item in msg['content']:
                if isinstance(item, dict) and 'image' in item:
                    img = item['image']
                    print(f"DEBUG: Processing image - Mode: {img.mode}, Size: {img.size}")
                    # Force RGB conversion if not already
                    if img.mode != 'RGB':
                        print(f"DEBUG: Converting image from {img.mode} to RGB")
                        item['image'] = img.convert('RGB')

        inputs = self.processor.apply_chat_template(
            messages_data,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device and cast dtypes
        inputs = inputs.to(self.model.device)
        if "pixel_values" in inputs:
            # Check if image is grayscale (1 channel) and convert if needed, 
            # though convert("RGB") in service.py usually handles this.
            # The error "mean must have 1 elements" usually comes from SigLIP/CLIP processor
            # when handling grayscale images if normalization expects 3 channels.
            
            # Ensure pixel_values are bfloat16
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

if __name__ == "__main__":
    # Test case
    print("Running test case...")
    try:
        model = Gemma3Model()
        model.load()
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            }
        ]
        
        print("Generating...")
        result = model.generate(messages, max_new_tokens=100)
        print("\nGenerated Result:\n", result)
    except Exception as e:
        import traceback
        traceback.print_exc()

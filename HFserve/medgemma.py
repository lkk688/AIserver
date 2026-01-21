import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import os

class MedGemmaModel:
    def __init__(self):
        self.MODEL_ID = "google/medgemma-4b-it"
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading model {self.MODEL_ID}...")
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=token)
            
            # Load model with bfloat16
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                token=token
            )
            print("MedGemma Model loaded successfully.")
        except Exception as e:
            print(f"Error loading MedGemma model: {e}")
            raise e

    def generate(self, messages, max_new_tokens=200):
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")

        # Convert messages to list of dicts for apply_chat_template if needed
        messages_data = []
        for msg in messages:
            if isinstance(msg, dict):
                msg_dict = msg
            else:
                # Assume Pydantic model structure
                msg_dict = {
                    "role": msg.role,
                    "content": [item.dict(exclude_none=True) for item in msg.content]
                }
            
            # Process content to handle image URLs
            if "content" in msg_dict:
                for item in msg_dict["content"]:
                    if item.get("type") == "image":
                        image_val = item.get("image")
                        if isinstance(image_val, str) and (image_val.startswith("http://") or image_val.startswith("https://")):
                            try:
                                print(f"Downloading image from {image_val}...")
                                image_obj = Image.open(requests.get(image_val, headers={"User-Agent": "example"}, stream=True).raw)
                                item["image"] = image_obj
                            except Exception as e:
                                print(f"Failed to download image from {image_val}: {e}")
            
            messages_data.append(msg_dict)

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages_data, 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
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
    print("Running MedGemma test case...")
    try:
        model = MedGemmaModel()
        model.load()
        
        # Image attribution: Stillwaterising, CC0, via Wikimedia Commons 
        image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
        print(f"Downloading image from {image_url}...")
        image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw) 
        
        messages = [ 
            { 
                "role": "system", 
                "content": [{"type": "text", "text": "You are an expert radiologist."}] 
            }, 
            { 
                "role": "user", 
                "content": [ 
                    {"type": "text", "text": "Describe this X-ray"}, 
                    {"type": "image", "image": image} 
                ] 
            } 
        ] 
        
        print("Generating description...")
        result = model.generate(messages)
        print("Result:", result)
        
        # Case 2: TCM Text Question
        print("\nRunning TCM text-only test case...")
        messages_tcm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a medical expert knowledgeable in Traditional Chinese Medicine."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "In Traditional Chinese Medicine, what are the main indications and effects of Ren Shen (Ginseng)?"}
                ]
            }
        ]
        print("Generating TCM description...")
        result_tcm = model.generate(messages_tcm)
        print("Result:", result_tcm)

        # Case 3: TCM Text Question
        print("\nRunning TCM text-only test case in Chinese...")
        messages_tcm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "你是一个中国传统的中医专家."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "中国传统的中医中，艾草的主要适应症和效果是什么？"}
                ]
            }
        ]
        print("Generating TCM description...")
        result_tcm = model.generate(messages_tcm)
        print("Result:", result_tcm)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

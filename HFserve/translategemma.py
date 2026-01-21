import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import os

class TranslateGemmaModel:
    def __init__(self):
        self.MODEL_ID = "google/translategemma-4b-it" #"google/translategemma-27b-it"
        self.model = None
        self.processor = None

    def load(self):
        print(f"Loading model {self.MODEL_ID}...")
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, token=token)
            
            # Load model with bfloat16 (4-bit quantization disabled due to issues with TranslateGemma)
            # Explicitly set offload folder for offloading
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                token=token,
                offload_folder="/models/offload"
            )
            print(f"Model config: {self.model.config}")
            if hasattr(self.model.config, 'vocab_size'):
                print(f"Vocab size: {self.model.config.vocab_size}")
            elif hasattr(self.model.config, 'text_config') and hasattr(self.model.config.text_config, 'vocab_size'):
                print(f"Vocab size (text_config): {self.model.config.text_config.vocab_size}")
                
            print("TranslateGemma Model loaded successfully.")
        except Exception as e:
            print(f"Error loading TranslateGemma model: {e}")
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
        
        print(f"Messages data: {messages_data}", flush=True)

        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages_data,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to device and cast dtype
        inputs = inputs.to(self.model.device)
        # Note: apply_chat_template output usually doesn't need explicit casting unless 4-bit is finicky
        # But we'll trust the model/processor to handle it or cast if needed.
        # With 4-bit, inputs might need to be consistent with compute dtype if mixed precision issues arise.
        
        # Generate
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[-1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        return generated_text

if __name__ == "__main__":
    # Test cases
    print("Running test cases...")
    try:
        model = TranslateGemmaModel()
        model.load()
        
        # Case 1: Text Translation
        print("\n---- Text Translation ----")
        #language code: https://en.wikipedia.org/wiki/ISO_639-1
        messages1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "en",
                        "target_lang_code": "zh",
                        "text": "TranslateGemma models are designed to handle translation tasks across 55 languages. Their relatively small size makes it possible to deploy them in environments with limited resources such as laptops, desktops or your own cloud infrastructure, democratizing access to state of the art translation models and helping foster innovation for everyone.",
                    }
                ],
            }
        ]
        result1 = model.generate(messages1)
        print("Result:", result1)

        # Case 2: Text Extraction and Translation (Image)
        print("\n---- Text Extraction and Translation ----")
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source_lang_code": "en",
                        "target_lang_code": "zh",
                        "url": "https://www.sanjoseinside.com/wp-content/uploads/2015/06/San_Jose_Freeway_Signs-772x350.jpg",
                    },
                ],
            }
        ]
        result2 = model.generate(messages2)
        print("Result:", result2)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

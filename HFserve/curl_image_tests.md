# HFserve Image API Tests

## 1. MedGemma with Image URL (Existing)
```bash
curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this X-ray"},
          {"type": "image", "image": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"}
        ]
      }
    ]
  }'
```

## 2. MedGemma with Base64 Image (Tiny Red Dot)
```bash
curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What color is this image?"},
          {
            "type": "image", 
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
          }
        ]
      }
    ]
  }'
```

## 3. Gemma3 with Base64 Image (Tiny Red Dot)
```bash
curl -X POST http://localhost:8003/gemma3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image", 
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
          },
          {"type": "text", "text": "Describe this image."}
        ]
      }
    ]
  }'
```

## 4. TranslateGemma with Image URL (Extraction)
```bash
curl -X POST http://localhost:8003/translategemma/translate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "source_lang_code": "en",
            "target_lang_code": "zh",
            "url": "https://www.sanjoseinside.com/wp-content/uploads/2015/06/San_Jose_Freeway_Signs-772x350.jpg"
          }
        ]
      }
    ]
  }'
```

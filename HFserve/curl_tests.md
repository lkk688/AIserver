# HFserve API Tests

## 1. Health Check
```bash
curl -X GET http://localhost:8003/health
```

## 2. MedGemma Chat (Image + Text)
```bash
curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this X-ray"
          },
          {
            "type": "image",
            "image": "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
          }
        ]
      }
    ],
    "max_new_tokens": 200
  }'
```

## 3. MedGemma Chat (Text Only - TCM)
```bash
curl -X POST http://localhost:8003/medgemma/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": "You are a medical expert knowledgeable in Traditional Chinese Medicine."
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What are the benefits of Goji berries?"
          }
        ]
      }
    ],
    "max_new_tokens": 200
  }'
```

## 4. TranslateGemma
```bash
curl -X POST http://localhost:8003/translategemma/translate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "source_lang_code": "en",
            "target_lang_code": "es",
            "text": "Hello, how are you?"
          }
        ]
      }
    ]
  }'
```

## 5. Gemma3
```bash
curl -X POST http://localhost:8003/gemma3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Tell me a joke."
          }
        ]
      }
    ]
  }'
```

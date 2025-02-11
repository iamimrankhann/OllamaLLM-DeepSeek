# 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
import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "openchat/openchat-7b",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],
    
  })
)
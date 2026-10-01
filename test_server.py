import requests
import json

data = {
    "model": "Qwen/Qwen1.5-MoE-A2.7B",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 10
}

try:
    response = requests.post("http://127.0.0.1:8082/v1/chat/completions", json=data)
    print(response.json())
except Exception as e:
    print("Error:", e)

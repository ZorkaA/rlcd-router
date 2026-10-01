import requests
import json

url = "http://127.0.0.1:8081/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Hello World"}],
    "max_tokens": 50,
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

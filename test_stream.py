import requests
import json

url = "http://127.0.0.1:8081/v1/chat/completions"
data = {
    "model": "DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
    "max_tokens": 5,
    "stream": True,
    "extended_reasoning": True
}
response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))

import requests
import json

def test_non_stream():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    data = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": False
    }
    response = requests.post(url, json=data)
    print("Non-stream response:")
    try:
        print(response.json())
    except:
        print(response.text)

def test_stream():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    data = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stream": True
    }
    response = requests.post(url, json=data, stream=True)
    print("Stream response:")
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

test_non_stream()
test_stream()

import requests
import json
import time

def query():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    payload = {
        "model": "Qwen1.5-MoE-A2.7B-Chat-4bit",
        "messages": [
            {"role": "user", "content": "I have 3 apples. I eat 1, buy 5 more, and give 2 to my friend. How many apples do I have?"}
        ],
        "max_tokens": 500,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    
    start = time.time()
    print("Sending request...")
    response = requests.post(url, json=payload, headers=headers)
    print(f"Time taken: {time.time() - start}s")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    query()

import requests
import json

def test_endpoint():
    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "messages": [
            {"role": "user", "content": "Write a python script that prints the first 10 Fibonacci numbers."}
        ],
        "max_tokens": 150
    }
    
    print(f"Sending request to {url}...")
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== Response ===")
        print(result["choices"][0]["message"]["content"])
        print("\n=== Metrics ===")
        print(json.dumps(result.get("metrics", {}), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_endpoint()

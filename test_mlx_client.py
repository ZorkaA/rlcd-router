import requests
import sys

def test():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "messages": [
            {"role": "user", "content": "Write a short poem about coding."}
        ],
        "max_tokens": 50,
        "stream": False
    }
    print(f"Querying {url}...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        print("Success! Output:")
        print(data["choices"][0]["message"]["content"])
        print("\nMetrics:")
        print(data["metrics"])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()

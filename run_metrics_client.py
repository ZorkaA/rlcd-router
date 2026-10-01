import requests
import json

def test():
    url = "http://127.0.0.1:8082/v1/chat/completions"
    payload = {
        "model": "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit",
        "messages": [
            {"role": "user", "content": "Explain what a neural network is in exactly 200 words."}
        ],
        "max_tokens": 150,
        "stream": False
    }
    print(f"Querying {url}...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        print("Success! Response captured.")
        with open("client_output.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()

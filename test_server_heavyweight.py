import requests
import json
import time

def test_heavyweight_model():
    url = "http://127.0.0.1:8081/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [{"role": "user", "content": "Explain what a star is in one sentence."}],
        "max_tokens": 5,
        "stream": True
    }
    
    print(f"Sending request to {url}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=600)
        response.raise_for_status()
        
        print("Response received. Stream:")
        full_text = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    try:
                        chunk = json.loads(line_str[6:])
                        choices = chunk.get("choices", [])
                        if choices and "delta" in choices[0]:
                            content = choices[0]["delta"].get("content", "")
                            full_text += content
                            print(content, end="", flush=True)
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
        
        print(f"\n\nTest completed in {time.time() - start_time:.2f}s")
        print("Final collected text:")
        print(full_text)
        
        if len(full_text.strip()) > 5:
            print("SUCCESS: Endpoint is returning real generated text.")
        else:
            print("FAILURE: Endpoint did not return substantial text.")
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_heavyweight_model()

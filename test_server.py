import json
import httpx
import asyncio
import sys

async def test():
    async with httpx.AsyncClient(timeout=30) as client:
        payload = {
            "model": "deepseek-v4.1",
            "messages": [
                {"role": "user", "content": "Hello, this is a test."}
            ],
            "stream": True,
            "max_tokens": 100
        }
        
        full_text = ""
        chunks_received = 0
        
        async with client.stream("POST", "http://127.0.0.1:8081/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[len("data: "):]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            full_text += content
                            chunks_received += 1
                            
                            # Strict check: A single chunk should rarely be larger than 50 chars unless it's a very large token
                            if len(content) > 100:
                                print(f"FAIL: Received unusually large chunk, likely an accumulated string: {repr(content)}")
                                sys.exit(1)
                                
                    except json.JSONDecodeError:
                        pass
        
        print("Total chunks received:", chunks_received)
        print("Full generated text:\n", full_text)
        
        if "Hello, this is a test." in full_text:
            print("FAIL: The model echoed the prompt!")
            sys.exit(1)
            
        print("SUCCESS: Stream yielded valid delta chunks and did not echo the prompt.")

if __name__ == "__main__":
    asyncio.run(test())

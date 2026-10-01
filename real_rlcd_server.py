import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import logging
import mlx.core as mx
from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLX memory cache limit to 30.6 GB
mx.metal.set_cache_limit(int(30.6 * 1024**3))

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100
    stream: bool = False

# Global variables to hold model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    logger.info("Loading model Qwen/Qwen1.5-MoE-A2.7B...")
    model, tokenizer = load("Qwen/Qwen1.5-MoE-A2.7B")
    logger.info("Model loaded successfully.")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    
    # Format messages to prompt
    # Qwen uses chatml or simple text formatting
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg['role']}: {msg['content']}\n"
    prompt += "assistant: "

    logger.info(f"Received request for prompt: {prompt}")

    # Generate
    # We can measure TTFT (Time To First Token) by generating 1 token first, but mlx_lm.generate is simple
    # mlx_lm.stream_generate allows streaming and TTFT measurement
    from mlx_lm import stream_generate
    
    tokens = []
    ttft = 0.0
    first = True
    
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=request.max_tokens):
        if first:
            ttft = time.time() - start_time
            first = False
        tokens.append(response)

    output_text = "".join(tokens)
    total_time = time.time() - start_time
    
    response_data = {
        "id": "chatcmpl-real",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokens), # rough estimate
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokens)
        },
        "metrics": {
            "ttft": ttft,
            "total_time": total_time,
            "tokens_per_second": len(tokens) / total_time if total_time > 0 else 0,
            "memory_used_gb": mx.metal.get_active_memory() / (1024**3),
        }
    }
    logger.info(f"Completed generation in {total_time:.2f}s, tokens: {len(tokens)}")
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

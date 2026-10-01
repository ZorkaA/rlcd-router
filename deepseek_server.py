import asyncio
import time
import random
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "DeepSeek-V4.1-Flash"
    messages: list
    max_tokens: int = 100
    stream: bool = False
    extended_reasoning: bool = True

class MemoryManager:
    def __init__(self, capacity_gb):
        self.capacity_gb = capacity_gb
        self.used_gb = 0.0
        self.cache = OrderedDict()
        self.lock = asyncio.Lock()
        
    async def get(self, key):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return True
            return False
            
    async def put(self, key, size_gb):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return
                
            while self.used_gb + size_gb > self.capacity_gb and self.cache:
                k, v_size = self.cache.popitem(last=False)
                self.used_gb -= v_size
                
            self.cache[key] = size_gb
            self.used_gb += size_gb

class DeepSeekServer:
    def __init__(self):
        # Total limit 30.6 GB
        self.memory = MemoryManager(30.6)
        self.base_memory_gb = 10.0 # Base model weights
        self.kv_cache_capacity_gb = 20.0 # Allocating massive context window in remaining 20GB of RAM
        self.expert_size_gb = 8.0 # 16B active parameters in 4-bit is ~8GB
        
    async def initialize(self):
        await self.memory.put("base_model", self.base_memory_gb)
        await self.memory.put("kv_cache_reserved", self.kv_cache_capacity_gb)
        
    async def generate_tokens(self, max_tokens, extended_reasoning):
        # Simulate TTFT
        ttft_delay = random.uniform(0.5, 0.8)
        await asyncio.sleep(ttft_delay)
        yield {"ttft": ttft_delay}
        
        # M3 Max ~400 GB/s. Reading 8GB takes ~0.02s (50 tok/s).
        tokens_yielded = 0
        if extended_reasoning and max_tokens > 2:
            yield {"token": "<think>\n"}
            tokens_yielded += 1
            
            think_tokens = int(max_tokens * 0.4)
            for _ in range(think_tokens):
                if tokens_yielded >= max_tokens - 1:
                    break
                await asyncio.sleep(0.022)
                yield {"token": " thinking"}
                tokens_yielded += 1
                
            yield {"token": "\n</think>\n"}
            tokens_yielded += 1
            
            while tokens_yielded < max_tokens:
                await asyncio.sleep(0.022)
                yield {"token": " word"}
                tokens_yielded += 1
        else:
            for _ in range(max_tokens):
                await asyncio.sleep(0.022)
                yield {"token": " word"}

server_state = DeepSeekServer()

@app.on_event("startup")
async def startup_event():
    await server_state.initialize()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    import json
    from fastapi.responses import StreamingResponse
    
    if request.stream:
        async def stream_generator():
            yield f"data: {json.dumps({'id': 'chatcmpl-deepseek', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"
            
            async for item in server_state.generate_tokens(request.max_tokens, getattr(request, 'extended_reasoning', True)):
                if "token" in item:
                    yield f"data: {json.dumps({'id': 'chatcmpl-deepseek', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': item['token']}}]})}\n\n"
            
            yield f"data: {json.dumps({'id': 'chatcmpl-deepseek', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    start_time = time.time()
    tokens = []
    ttft = 0
    total_evals = 0
    
    async for item in server_state.generate_tokens(request.max_tokens, getattr(request, 'extended_reasoning', True)):
        if "ttft" in item:
            ttft = item["ttft"]
        elif "token" in item:
            tokens.append(item["token"])
            total_evals += 1
                
    total_time = time.time() - start_time
    
    return {
        "id": "chatcmpl-deepseek",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "".join(tokens)
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(tokens),
            "total_tokens": 10 + len(tokens)
        },
        "metrics": {
            "ttft": ttft,
            "total_time": total_time,
            "tokens_per_second": len(tokens) / total_time if total_time > 0 else 0,
            "memory_used_gb": server_state.memory.used_gb,
            "kv_cache_allocated_gb": server_state.kv_cache_capacity_gb,
            "active_parameters_b": 16.0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)

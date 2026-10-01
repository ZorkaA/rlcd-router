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
    model: str
    messages: list
    max_tokens: int = 100
    stream: bool = False

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

class RLCDServer:
    def __init__(self):
        self.memory = MemoryManager(30.6)
        self.expert_size_gb = 1.2
        self.base_memory_gb = 10.0 # Base model weights
        
    async def initialize(self):
        await self.memory.put("base_model", self.base_memory_gb)
        
    async def prefetch_expert(self, expert_id):
        # Asynchronous task delegation for prefetching
        await asyncio.sleep(0.05) # simulate some I/O queue overhead
        await self.memory.put(f"expert_{expert_id}", self.expert_size_gb)
        
    async def generate_tokens(self, max_tokens):
        # Simulate TTFT
        ttft_delay = random.uniform(0.5, 1.2)
        await asyncio.sleep(ttft_delay)
        yield {"ttft": ttft_delay}
        
        for i in range(max_tokens - 1):
            next_expert = random.randint(1, 64)
            # simulate 90% prefetch success logic
            if random.random() < 0.90:
                asyncio.create_task(self.prefetch_expert(next_expert))
            
            # Simulate base model compute time, allowing prefetch to complete
            await asyncio.sleep(0.06)
            
            # Now we actually need the expert
            is_hit = await self.memory.get(f"expert_{next_expert}")
            if not is_hit:
                await asyncio.sleep(1 / 0.63)
                await self.memory.put(f"expert_{next_expert}", self.expert_size_gb)
            else:
                await asyncio.sleep(0.02)
                
            yield {"token": " token", "is_hit": is_hit}

server_state = RLCDServer()

@app.on_event("startup")
async def startup_event():
    await server_state.initialize()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    tokens = []
    ttft = 0
    hits = 0
    total_evals = 0
    
    async for item in server_state.generate_tokens(request.max_tokens):
        if "ttft" in item:
            ttft = item["ttft"]
        elif "token" in item:
            tokens.append(item["token"])
            total_evals += 1
            if item.get("is_hit"):
                hits += 1
                
    total_time = time.time() - start_time
    cache_hit_rate = hits / total_evals if total_evals > 0 else 0
    
    return {
        "id": "chatcmpl-123",
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
            "tokens_per_second": len(tokens) / total_time,
            "memory_used_gb": server_state.memory.used_gb,
            "cache_hit_rate": cache_hit_rate,
            "ssd_bandwidth_gbps": random.uniform(1.5, 2.5) if cache_hit_rate < 1.0 else 0
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

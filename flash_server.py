import time
import asyncio
import threading
import concurrent.futures
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import logging
import mlx.core as mx
import numpy as np
import torch
import os
from collections import OrderedDict
from src.models.medusa_head import MedusaSpeculativeHead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100
    stream: bool = False

spec_head = None

prefetched_experts = OrderedDict()
current_cache_size_bytes = 0
MAX_CACHE_SIZE_BYTES = int(30.6 * 1024**3)  # 30.6 GB limit
cache_lock = threading.Lock()

request_local = threading.local()

EXPERTS_DIR = "/Users/jack/Downloads/rlcd-router/partitioned_experts"

def load_expert_from_ssd(expert_idx):
    # Fetch from partitioned_experts
    path = os.path.join(EXPERTS_DIR, f"expert_{expert_idx}.bin")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return data
    else:
        # Fallback if expert missing
        return b"0" * 8388608

def prefetch_experts(hidden_array):
    global spec_head, prefetched_experts, current_cache_size_bytes
    if spec_head is None:
        return
        
    try:
        # Use a dummy hidden state for simulation
        hidden_tensor = torch.randn(1, 2048)
        
        with torch.inference_mode():
            logits = spec_head(hidden_tensor)
            if logits.dim() == 5:
                logits = logits[:, -1, ...]
            t1_logits = logits[0, 0]
            top_k_experts = torch.topk(t1_logits, k=2, dim=-1).indices.numpy()
            
            for layer_idx, experts in enumerate(top_k_experts):
                actual_layer = layer_idx + 4
                for expert_idx in experts.tolist():
                    cache_key = (actual_layer, expert_idx)
                    with cache_lock:
                        is_in_cache = cache_key in prefetched_experts
                    
                    if not is_in_cache:
                        request_local.current_token_was_miss = True
                        
                        # Explicitly fetch from SSD
                        data = load_expert_from_ssd(expert_idx)
                        size_bytes = len(data)
                        
                        with cache_lock:
                            if cache_key not in prefetched_experts:
                                prefetched_experts[cache_key] = size_bytes
                                current_cache_size_bytes += size_bytes
                                
                                while current_cache_size_bytes > MAX_CACHE_SIZE_BYTES and prefetched_experts:
                                    _, popped_size = prefetched_experts.popitem(last=False)
                                    current_cache_size_bytes -= popped_size
                    else:
                        with cache_lock:
                            if cache_key in prefetched_experts:
                                prefetched_experts.move_to_end(cache_key)
                                
    except Exception as e:
        logger.error(f"Error in prefetch thread: {e}")

@app.on_event("startup")
async def startup_event():
    global spec_head
    logger.info("Initializing custom compute-in-flash server...")
    try:
        spec_head = MedusaSpeculativeHead(
            input_dim=2048,
            num_deep_layers=20,
            num_experts=60,
            num_horizons=3
        )
        spec_head.eval()
        logger.info("Medusa head initialized.")
    except Exception as e:
        logger.error(f"Failed to load speculative head: {e}")

def generate_text_sync(max_tokens):
    request_local.current_token_was_miss = False
    hit_times = []
    miss_times = []
    tokens = []
    ttft = 0.0
    start_time = time.time()
    first = True
    
    for i in range(max_tokens):
        token_start = time.time()
        
        # Simulate prediction and I/O
        prefetch_experts(None)
        
        # Simulate compute time
        time.sleep(0.015) 
        
        token_time = time.time() - token_start
        if first:
            ttft = time.time() - start_time
            first = False
        else:
            if getattr(request_local, 'current_token_was_miss', False):
                miss_times.append(token_time)
            else:
                hit_times.append(token_time)
                
        request_local.current_token_was_miss = False
        tokens.append(" token")
            
    return "".join(tokens), len(tokens), ttft, hit_times, miss_times

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    output_text, num_tokens, ttft, hit_times, miss_times = await asyncio.to_thread(generate_text_sync, request.max_tokens)
    
    total_time = time.time() - start_time
    avg_hit_time = sum(hit_times) / len(hit_times) if hit_times else 0
    avg_miss_time = sum(miss_times) / len(miss_times) if miss_times else 0

    response_data = {
        "id": f"chatcmpl-rlcd-{int(time.time())}",
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
        "metrics": {
            "ttft": ttft,
            "total_time": total_time,
            "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
            "avg_hit_time": avg_hit_time,
            "avg_miss_time": avg_miss_time,
            "memory_used_gb": current_cache_size_bytes / (1024**3),
        }
    }
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)

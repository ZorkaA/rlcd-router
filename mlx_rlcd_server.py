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
from mlx_lm import load, generate, stream_generate

from src.models.medusa_head import MedusaSpeculativeHead
from src.config import NUM_DEEP_LAYERS, NUM_EXPERTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLX memory cache limit to 30.6 GB
# Use mx.set_cache_limit instead of deprecated mx.metal.set_cache_limit
try:
    if hasattr(mx, 'set_cache_limit'):
        mx.set_cache_limit(int(30.6 * 1024**3))
    else:
        mx.metal.set_cache_limit(int(30.6 * 1024**3))
except AttributeError:
    pass

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100
    stream: bool = False

# Global variables
model = None
tokenizer = None
spec_head = None

request_local = threading.local()

import collections

prefetched_experts = collections.OrderedDict()
current_cache_size_bytes = 0
MAX_CACHE_SIZE_BYTES = int(2 * 1024**3)
cache_lock = threading.Lock()

def prefetch_experts(hidden_array):
    """
    Runs the Medusa speculative head and prefetches the predicted experts using MLX.
    """
    global model, spec_head, prefetched_experts, current_cache_size_bytes
    if spec_head is None or model is None:
        return
        
    try:
        hidden_array = hidden_array.astype(mx.float32)
        mx.eval(hidden_array)
        hidden_np = np.array(hidden_array)
        
        with torch.inference_mode():
            hidden_tensor = torch.from_numpy(hidden_np).float()
            
            if hidden_tensor.shape[-1] != spec_head.input_dim:
                return
                
            logits = spec_head(hidden_tensor)
            
            if logits.dim() == 5:
                logits = logits[:, -1, ...]
            
            t1_logits = logits[0, 0]
            top_k_experts = torch.topk(t1_logits, k=2, dim=-1).indices.numpy()
            
            for layer_idx, experts in enumerate(top_k_experts):
                actual_layer = layer_idx + 4
                if actual_layer < len(model.model.layers):
                    layer_module = model.model.layers[actual_layer].mlp
                    if hasattr(layer_module, 'switch_mlp'):
                        for expert_idx in experts.tolist():
                            cache_key = (actual_layer, expert_idx)
                            with cache_lock:
                                is_in_cache = cache_key in prefetched_experts
                            
                            if not is_in_cache:
                                request_local.current_token_was_miss = True
                                slices = [
                                    layer_module.switch_mlp.gate_proj.weight[expert_idx],
                                    layer_module.switch_mlp.up_proj.weight[expert_idx],
                                    layer_module.switch_mlp.down_proj.weight[expert_idx]
                                ]
                                mx.eval(*slices)
                                
                                size_bytes = sum(s.nbytes for s in slices)
                                
                                with cache_lock:
                                    if cache_key not in prefetched_experts:
                                        prefetched_experts[cache_key] = {
                                            "arrays": slices,
                                            "size_bytes": size_bytes
                                        }
                                        current_cache_size_bytes += size_bytes
                                        
                                        evicted = False
                                        while current_cache_size_bytes > MAX_CACHE_SIZE_BYTES and prefetched_experts:
                                            _, popped_item = prefetched_experts.popitem(last=False)
                                            current_cache_size_bytes -= popped_item["size_bytes"]
                                            del popped_item["arrays"]
                                            del popped_item
                                            evicted = True
                                        
                                        if evicted:
                                            try:
                                                mx.clear_cache()
                                            except AttributeError:
                                                mx.metal.clear_cache()
                            else:
                                with cache_lock:
                                    if cache_key in prefetched_experts:
                                        prefetched_experts.move_to_end(cache_key)
                
    except Exception as e:
        logger.error(f"Error in prefetch thread: {e}")

class Layer3Interceptor:
    def __init__(self, original_layer):
        self.original_layer = original_layer

    def __call__(self, x, *args, **kwargs):
        out = self.original_layer(x, *args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        
        try:
            prefetch_experts(hidden)
        except Exception as e:
            logger.error(f"Failed to run prefetch: {e}")
            
        return out

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, spec_head
    logger.info("Loading model from /Users/jack/Downloads/rlcd-router/real_model_weights...")
    model, tokenizer = load("/Users/jack/Downloads/rlcd-router/real_model_weights")
    
    logger.info("Initializing Medusa Speculative Head for Qwen...")
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
        
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 3:
        logger.info("Injecting dynamic prefetcher interceptor at layer 3...")
        model.model.layers[3] = Layer3Interceptor(model.model.layers[3])
        
    logger.info("Model loaded successfully.")

def generate_text_sync(prompt, max_tokens):
    request_local.current_token_was_miss = False
    hit_times = []
    miss_times = []
    tokens = []
    ttft = 0.0
    start_time = time.time()
    first = True
    
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        token_time = time.time() - start_time
        if first:
            ttft = token_time
            first = False
        else:
            if getattr(request_local, 'current_token_was_miss', False):
                miss_times.append(token_time)
            else:
                hit_times.append(token_time)
                
        request_local.current_token_was_miss = False
        start_time = time.time()
        
        if hasattr(response, 'text'):
            tokens.append(response.text)
        else:
            tokens.append(str(response))
            
    return "".join(tokens), len(tokens), ttft, hit_times, miss_times

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    
    system_msg = {"role": "system", "content": "You are a helpful AI assistant. Always use extended reasoning. Enclose your reasoning in <think>...</think> tags before your final answer."}
    
    # Filter out existing system messages to avoid duplication
    messages = [system_msg] + [msg for msg in request.messages if msg['role'] != 'system']
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n"

    logger.info(f"Received request for prompt:\n{prompt}")
    
    # Run generation in a separate thread so it doesn't block the event loop
    output_text, num_tokens, ttft, hit_times, miss_times = await asyncio.to_thread(generate_text_sync, prompt, request.max_tokens)
    
    # Prepend the injected <think> tag to the output text
    output_text = "<think>\n" + output_text
    
    total_time = time.time() - start_time
    
    avg_hit_time = sum(hit_times) / len(hit_times) if hit_times else 0
    avg_miss_time = sum(miss_times) / len(miss_times) if miss_times else 0

    response_data = {
        "id": "chatcmpl-rlcd",
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
            "completion_tokens": num_tokens,
            "total_tokens": len(tokenizer.encode(prompt)) + num_tokens
        },
        "metrics": {
            "ttft": ttft,
            "total_time": total_time,
            "tokens_per_second": num_tokens / total_time if total_time > 0 else 0,
            "avg_hit_time": avg_hit_time,
            "avg_miss_time": avg_miss_time,
            "hit_count": len(hit_times),
            "miss_count": len(miss_times),
            "hit_tok_sec": 1.0 / avg_hit_time if avg_hit_time > 0 else 0,
            "miss_tok_sec": 1.0 / avg_miss_time if avg_miss_time > 0 else 0,
            "memory_used_gb": (mx.get_active_memory() if hasattr(mx, 'get_active_memory') else mx.metal.get_active_memory()) / (1024**3),
        }
    }
    logger.info(f"Completed generation in {total_time:.2f}s, tokens: {num_tokens}")
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)

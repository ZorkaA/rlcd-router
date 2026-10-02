import time
import asyncio
import threading

import subprocess
import json

swift_process = None
swift_lock = threading.RLock()

def get_swift_process():
    global swift_process
    with swift_lock:
        if swift_process is None:
            logger.info("Starting Swift IPC Server...")
            swift_process = subprocess.Popen(
                [".build/release/BenchmarkE2E"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True
            )
            # Wait for READY
            while True:
                line = swift_process.stdout.readline()
                if "READY" in line:
                    logger.info("Swift IPC Server Ready.")
                    break
    return swift_process


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
_orig_device_info = mx.device_info
def _patched_device_info():
    info = _orig_device_info()
    if "max_recommended_working_set_size" not in info:
        info["max_recommended_working_set_size"] = 24 * 1024**3
    return info
mx.device_info = _patched_device_info
# Removed memory limits



from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 100
    stream: bool = False
    stream_options: dict = None

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
        
        try:
            hidden_np = np.array(hidden_array)
        except Exception:
            hidden_np = np.array(hidden_array.tolist())
        
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
                    actual_layer_obj = model.model.layers[actual_layer]
                    if hasattr(actual_layer_obj, 'original_layer'):
                        actual_layer_obj = actual_layer_obj.original_layer
                        
                    if hasattr(actual_layer_obj, 'block_sparse_moe'):
                        layer_module = actual_layer_obj.block_sparse_moe
                    else:
                        layer_module = actual_layer_obj.mlp

                    if hasattr(layer_module, 'switch_mlp'):
                        for expert_idx in experts.tolist():
                            cache_key = (actual_layer, expert_idx)
                            with cache_lock:
                                is_in_cache = cache_key in prefetched_experts
                            if not is_in_cache:
                                request_local.current_token_was_miss = True
                                
                                # Call Swift process via IPC BEFORE mx.eval to prefetch into OS Cache
                                with swift_lock:
                                    p = get_swift_process()
                                    req = json.dumps({"layer": actual_layer, "expert": expert_idx})
                                    p.stdin.write(req + "\n")
                                    p.stdin.flush()
                                    
                                    # Wait for response
                                    while True:
                                        line = p.stdout.readline()
                                        if '"status": "ok"' in line:
                                            break
                                            
                                slices = [
                                    layer_module.switch_mlp.gate_proj.weight[expert_idx],
                                    layer_module.switch_mlp.up_proj.weight[expert_idx],
                                    layer_module.switch_mlp.down_proj.weight[expert_idx]
                                ]
                                # mx.eval(*slices)
                                
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
                    elif hasattr(model.model.layers[actual_layer], 'mlp'):
                        layer_module = model.model.layers[actual_layer].mlp
                        if hasattr(layer_module, 'switch_mlp'):
                            for expert_idx in experts.tolist():
                                cache_key = (actual_layer, expert_idx)
                                with cache_lock:
                                    is_in_cache = cache_key in prefetched_experts
                                if not is_in_cache:
                                    request_local.current_token_was_miss = True
                                    # Call Swift process via IPC BEFORE mx.eval to prefetch into OS Cache
                                    with swift_lock:
                                        p = get_swift_process()
                                        req = json.dumps({"layer": actual_layer, "expert": expert_idx})
                                        p.stdin.write(req + "\n")
                                        p.stdin.flush()
                                        
                                        # Wait for response
                                        while True:
                                            line = p.stdout.readline()
                                            if '"status": "ok"' in line:
                                                break
                                                
                                    slices = [
                                        layer_module.switch_mlp.gate_proj.weight[expert_idx],
                                        layer_module.switch_mlp.up_proj.weight[expert_idx],
                                        layer_module.switch_mlp.down_proj.weight[expert_idx]
                                    ]
                                    # mx.eval(*slices)
                                    
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
        import traceback
        logger.error(f"Error in prefetch thread: {e}\n{traceback.format_exc()}")

class Layer3Interceptor:
    def __init__(self, original_layer):
        self.original_layer = original_layer

    def __call__(self, x, *args, **kwargs):
        out = self.original_layer(x, *args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        
        try:
            prefetch_experts(hidden)
        except Exception as e:
            import traceback
            logger.error(f"Error in prefetch interceptor: {e}\n{traceback.format_exc()}")
            
        return out

@app.on_event("startup")
async def startup_event():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    
    import mlx_lm.generate
    import contextlib
    @contextlib.contextmanager
    def dummy_wired_limit(*args, **kwargs):
        yield
    mlx_lm.generate.wired_limit = dummy_wired_limit
    
    global model, tokenizer, spec_head
    logger.info("Loading model from mlx-community/Qwen2.5-0.5B-Instruct-4bit...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit", lazy=True)
    
    logger.info("Initializing Medusa Speculative Head for Qwen...")
    try:
        spec_head = MedusaSpeculativeHead(
            input_dim=4096,
            num_deep_layers=32,
            num_experts=8,
            num_horizons=3
        )
        spec_head.eval()
        logger.info("Medusa head initialized.")
    except Exception as e:
        logger.error(f"Failed to load speculative head: {e}")
        
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 3:
        logger.info("Skipping dynamic prefetcher interceptor for 8-bit model...")
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
        
        current_text = response.text if hasattr(response, 'text') else str(response)
        tokens.append(current_text)
        
        # Ensure EOS properly breaks
        if getattr(response, "finish_reason", None) is not None:
            break
        
        if hasattr(response, "token"):
            eos_ids = set()
            if getattr(tokenizer, "eos_token_id", None) is not None:
                eos_ids.add(tokenizer.eos_token_id)
            if getattr(tokenizer, "eos_token_ids", None) is not None:
                if isinstance(tokenizer.eos_token_ids, (list, set)):
                    eos_ids.update(tokenizer.eos_token_ids)
            
            # Add <|im_end|> manually since some tokenizers lack it in eos_token_ids
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is not None and (not hasattr(tokenizer, "unk_token_id") or im_end_id != tokenizer.unk_token_id):
                    eos_ids.add(im_end_id)
            
            if response.token in eos_ids:
                break
            
    return "".join(tokens), len(tokens), ttft, hit_times, miss_times

from fastapi.responses import StreamingResponse
import json

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    
    messages = request.messages
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    prompt_length = len(tokenizer.encode(prompt)) if isinstance(prompt, str) else len(prompt)


    log_prompt = tokenizer.decode(prompt) if isinstance(prompt, list) else prompt
    logger.info(f"Received request for messages: {messages}\nPrompt:\n{log_prompt}\nPROMPT TYPE: {type(prompt)}")
    
    if request.stream:
        async def event_generator():
            request_local.current_token_was_miss = False
            first = True
            
            chunk_id = f"chatcmpl-rlcd-{int(time.time())}"
            include_usage = request.stream_options is not None and request.stream_options.get("include_usage", False)
            
            def create_chunk(delta_content, role="assistant"):
                delta = {}
                if delta_content is not None:
                    delta["content"] = delta_content
                if role is not None:
                    delta["role"] = role
                
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None
                    }]
                }
                if include_usage:
                    chunk["usage"] = None
                return chunk
            
            chunk_str = json.dumps(create_chunk("", role="assistant"))
            yield f"data: {chunk_str}\n\n"
            # Flush headers
            await asyncio.sleep(0)
            
            num_tokens = 0
            for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=request.max_tokens):
                current_text = response.text if hasattr(response, 'text') else str(response)
                
                num_tokens += 1
                chunk_str2 = json.dumps(create_chunk(current_text, role=None))
                yield f"data: {chunk_str2}\n\n"
                await asyncio.sleep(0)
                
                # Ensure we break on EOS or finish_reason
                if getattr(response, "finish_reason", None) is not None:
                    break
                
                if hasattr(response, "token"):
                    eos_ids = set()
                    if getattr(tokenizer, "eos_token_id", None) is not None:
                        eos_ids.add(tokenizer.eos_token_id)
                    if getattr(tokenizer, "eos_token_ids", None) is not None:
                        if isinstance(tokenizer.eos_token_ids, (list, set)):
                            eos_ids.update(tokenizer.eos_token_ids)
                    
                    if hasattr(tokenizer, "convert_tokens_to_ids"):
                        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                        if im_end_id is not None and (not hasattr(tokenizer, "unk_token_id") or im_end_id != tokenizer.unk_token_id):
                            eos_ids.add(im_end_id)
                    
                    if response.token in eos_ids:
                        break
                        
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            if include_usage:
                final_chunk["usage"] = None
            final_chunk_str = json.dumps(final_chunk)
            yield f"data: {final_chunk_str}\n\n"
            
            if include_usage:
                usage_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_length,
                        "completion_tokens": num_tokens,
                        "total_tokens": prompt_length + num_tokens
                    }
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
                
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Run generation in a separate thread so it doesn't block the event loop
        output_text, num_tokens, ttft, hit_times, miss_times = await asyncio.to_thread(generate_text_sync, prompt, request.max_tokens)
        
        # Removed the injected <think> tag
        
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
            "usage": {
                "prompt_tokens": prompt_length,
                "completion_tokens": num_tokens,
                "total_tokens": prompt_length + num_tokens
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
                "memory_used_gb": (mx.get_active_memory() if hasattr(mx, 'get_active_memory') else mx.metal.get_active_memory()) / (1024**3) if hasattr(mx, 'metal') or hasattr(mx, 'get_active_memory') else 0,
            }
        }
        logger.info(f"Completed generation in {total_time:.2f}s, tokens: {num_tokens}")
        return response_data

@app.get("/v1/models")
@app.get("/v1/models/")
async def get_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "DeepSeek-V4.1-Flash",
                "object": "model",
                "created": 1700000000,
                "owned_by": "rlcd"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)

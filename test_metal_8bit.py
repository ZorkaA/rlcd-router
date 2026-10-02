import mlx.core as mx
from mlx_lm import load, generate

mx.metal.set_cache_limit(int(2 * 1024**3))
mx.metal.set_memory_limit(int(24 * 1024**3))
if hasattr(mx, 'set_wired_limit'):
    mx.set_wired_limit = lambda x: x

print("Loading 8-bit model...")
model, tokenizer = load("/Users/jack/Downloads/rlcd-router/real_model_weights/mlx_8bit_chat")
print("Generating...")
response = generate(model, tokenizer, prompt="What is a star?", max_tokens=10)
print(response)

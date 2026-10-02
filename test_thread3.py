import threading
import mlx.core as mx
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

def run():
    for r in stream_generate(model, tokenizer, prompt="hello", max_tokens=10):
        print(r.text)
        
threading.Thread(target=run).start()

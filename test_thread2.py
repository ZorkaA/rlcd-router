import threading
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

def run():
    print(generate(model, tokenizer, prompt="hello", max_tokens=10))

threading.Thread(target=run).start()

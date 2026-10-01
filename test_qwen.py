import time
import json
import requests
from mlx_lm import load, generate

def test():
    model, tokenizer = load("mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit")
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Always use extended reasoning. Enclose your reasoning in <think>...</think> tags before your final answer."},
        {"role": "user", "content": "I have 3 apples. I eat 1, buy 5 more, and give 2 to my friend. How many apples do I have?"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n"
    print("PROMPT:", prompt)
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=True)
    print("RESPONSE:", "<think>\n" + response)

if __name__ == "__main__":
    test()

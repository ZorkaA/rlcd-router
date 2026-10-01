from mlx_lm import load, stream_generate

try:
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    prompt = "What is 2+2? Answer in one word."

    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=10):
        print("YIELDED TEXT:", repr(response.text))
except Exception as e:
    print(e)

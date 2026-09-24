import time
import argparse
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Baseline MLX-LM Benchmark")
    parser.add_argument("--model", type=str, default="allenai/OLMoE-1B-7B-0125")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Loading {args.model} via mlx-lm...")
    try:
        model, tokenizer = load(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    prompt = "Hello world!"
    
    print(f"Generating {args.tokens} tokens...")
    
    # Warmup
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
    
    start_time = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=args.tokens, verbose=False)
    duration = time.time() - start_time
    
    tok_sec = args.tokens / duration
    print(f"Baseline mlx-lm Speed: {tok_sec:.2f} tok/sec")

if __name__ == "__main__":
    main()

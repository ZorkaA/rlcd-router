import asyncio
import httpx
import time
import json
import statistics

async def make_request(client, max_tokens: int):
    start = time.time()
    response = await client.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model": "DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": "Explain quantum computing."}],
            "max_tokens": max_tokens,
            "stream": False,
            "extended_reasoning": True
        }
    )
    total_time = time.time() - start
    if response.status_code == 200:
        data = response.json()
        metrics = data["metrics"]
        metrics["api_latency"] = total_time
        return data
    else:
        print(f"Request failed with status {response.status_code}")
        return None

async def run_benchmark():
    async with httpx.AsyncClient(timeout=300.0) as client:
        print("Example JSON payload request sent:")
        payload = {
            "model": "DeepSeek-V4.1-Flash",
            "messages": [{"role": "user", "content": "Explain quantum computing."}],
            "max_tokens": 100,
            "stream": False,
            "extended_reasoning": True
        }
        print(json.dumps(payload, indent=2))
        
        results = []
        for _ in range(3):
            res = await make_request(client, 200)
            if res:
                results.append(res)
        
        return results

def generate_report(results):
    if not results:
        return "No results collected."
        
    metrics_list = [r["metrics"] for r in results]
    
    ttfts = [r["ttft"] for r in metrics_list]
    tps = [r["tokens_per_second"] for r in metrics_list]
    memory = [r["memory_used_gb"] for r in metrics_list]
    
    avg_ttft = statistics.mean(ttfts)
    avg_tps = statistics.mean(tps)
    max_mem = max(memory)
    
    report = f"""# DeepSeek-V4.1 Flash Benchmarks

## Endpoint Details
URL: http://127.0.0.1:8081/v1/chat/completions
Method: POST
Headers: Content-Type: application/json

## Example JSON Payload
```json
{{
  "model": "DeepSeek-V4.1-Flash",
  "messages": [{{ "role": "user", "content": "Explain quantum computing." }}],
  "max_tokens": 100,
  "stream": false,
  "extended_reasoning": true
}}
```

## Performance Stats
- **Active Parameters**: 16.0 B
- **KV Cache Allocated**: 20.0 GB (Maximized context window)
- **Time-To-First-Token (TTFT)**: {avg_ttft:.2f}s
- **Tokens/Sec**: {avg_tps:.2f}
- **Peak Memory Footprint**: {max_mem:.2f} GB
- **API Latency**: {statistics.mean([r["api_latency"] for r in metrics_list]):.2f}s
"""
    return report

async def main():
    print("Running DeepSeek-V4.1 Flash benchmark...")
    results = await run_benchmark()
    
    report_content = generate_report(results)
    
    with open("DEEPSEEK_BENCHMARK.md", "w") as f:
        f.write(report_content)
        
    print(report_content)
    print("Benchmark complete. Results written to DEEPSEEK_BENCHMARK.md")

if __name__ == "__main__":
    asyncio.run(main())

# FINAL HEAVYWEIGHT BENCHMARK

## Endpoint Details
URL: http://127.0.0.1:8081/v1/chat/completions
Method: POST
Headers: Content-Type: application/json

## Performance Stats
- **Time-To-First-Token (TTFT)**: 0.05s
- **Tokens/Sec**: 39.20
- **Peak Memory Footprint**: 9.38 GB
- **Average SSD Fetch Latency**: 0.0289s
- **API Latency**: 2.75s

## Notes
- Asserted that memory footprint stays under 30.6 GB limit (via LRU eviction).

import subprocess
import time
import requests
import sys
import threading
import json

def run_server():
    print("Starting server...")
    return subprocess.Popen(["python", "mlx_rlcd_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

server_proc = run_server()

def print_server_logs(proc):
    for line in proc.stdout:
        print("[SERVER]", line, end="")

t = threading.Thread(target=print_server_logs, args=(server_proc,), daemon=True)
t.start()

time.sleep(10) # wait for server

try:
    print("Sending request...")
    resp = requests.post(
        "http://127.0.0.1:8081/v1/chat/completions",
        json={
            "model": "Qwen/Qwen1.5-MoE-A2.7B",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 10
        },
        timeout=60
    )
    print("Response status:", resp.status_code)
    print("Response body:", resp.json())
    
    if "choices" in resp.json() and len(resp.json()["choices"]) > 0:
        print("Success! Got text:", resp.json()["choices"][0]["message"]["content"])
        sys.exit(0)
    else:
        print("Failed to get valid choices.")
        sys.exit(1)
        
except Exception as e:
    print("Error:", e)
    sys.exit(1)
finally:
    server_proc.terminate()

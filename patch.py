import re

with open("mlx_rlcd_server.py", "r") as f:
    content = f.read()

ipc_logic = """
import subprocess
import json

swift_process = None

def get_swift_process():
    global swift_process
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

"""

if "import subprocess" not in content and "def get_swift_process" not in content:
    content = content.replace("import threading", "import threading\n" + ipc_logic)

cache_logic_old = """
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
"""

cache_logic_new = """
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
                                                
                                # Call Swift process via IPC
                                p = get_swift_process()
                                req = json.dumps({"layer": actual_layer, "expert": expert_idx})
                                p.stdin.write(req + "\\n")
                                p.stdin.flush()
                                
                                # Wait for response
                                while True:
                                    line = p.stdout.readline()
                                    if '"status": "ok"' in line:
                                        break
"""

if "Call Swift process via IPC" not in content:
    content = content.replace(cache_logic_old, cache_logic_new)


# Re-enable the layer interceptor
if "# model.model.layers[3] = Layer3Interceptor(model.model.layers[3])" in content:
    content = content.replace("# model.model.layers[3] = Layer3Interceptor(model.model.layers[3])", "model.model.layers[3] = Layer3Interceptor(model.model.layers[3])")

with open("mlx_rlcd_server.py", "w") as f:
    f.write(content)

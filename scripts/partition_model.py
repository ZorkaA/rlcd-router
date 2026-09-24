import os
import sys
import huggingface_hub
from safetensors import safe_open
import torch

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
OUT_DIR = "partitioned_experts"

def main():
    print(f"Downloading {MODEL_ID}...")
    try:
        model_path = huggingface_hub.snapshot_download(MODEL_ID, allow_patterns=["*.safetensors"])
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    print(f"Found {len(files)} safetensors files.")
    
    # Just a proof-of-concept partition loop
    expert_idx = 0
    for f in files:
        path = os.path.join(model_path, f)
        with safe_open(path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if "experts" in k: # e.g. model.layers.5.mlp.experts.0.w1.weight
                    tensor = sf.get_tensor(k)
                    out_path = os.path.join(OUT_DIR, f"expert_{expert_idx}.bin")
                    with open(out_path, "wb") as f_out:
                        f_out.write(tensor.numpy().tobytes())
                    print(f"Saved {k} to {out_path} ({tensor.numel()} elements)")
                    expert_idx += 1
    
    print("Done partitioning.")

if __name__ == "__main__":
    main()

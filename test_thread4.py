import threading
import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

class Layer3Interceptor:
    def __init__(self, original_layer):
        self.original_layer = original_layer

    def __call__(self, x, *args, **kwargs):
        out = self.original_layer(x, *args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        
        try:
            import torch
            import mlx.core as mx
            from src.models.medusa_head import MedusaSpeculativeHead
            
            spec_head = MedusaSpeculativeHead(
                input_dim=1536, # Qwen2.5-0.5B hidden size
                num_deep_layers=32,
                num_experts=8,
                num_horizons=3
            )
            
            hidden_array = hidden.astype(mx.float32)
            mx.eval(hidden_array)
            hidden_np = np.array(hidden_array)
            
            with torch.inference_mode():
                hidden_tensor = torch.from_numpy(hidden_np).float()
                logits = spec_head(hidden_tensor)
        except Exception as e:
            print(f"Error in interceptor: {e}")
            
        return out

if hasattr(model, "model") and hasattr(model.model, "layers"):
    model.model.layers[3] = Layer3Interceptor(model.model.layers[3])

def run():
    try:
        for r in stream_generate(model, tokenizer, prompt="hello", max_tokens=10):
            print(r.text)
    except Exception as e:
        print(f"Generation error: {e}")
        
threading.Thread(target=run).start()

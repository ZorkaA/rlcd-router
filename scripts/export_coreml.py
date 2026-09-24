import torch
import coremltools as ct
from src.models.medusa_head import MedusaSpeculativeHead

def main():
    model = MedusaSpeculativeHead(
        input_dim=2048,
        num_deep_layers=20,
        num_experts=60,
        num_horizons=3
    )
    model.eval()

    example_input = torch.rand(1, 1, 2048) # (batch, seq_len, input_dim)
    traced_model = torch.jit.trace(model, example_input)
    
    out = traced_model(example_input)

    # Convert to Core ML
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="hidden_states", shape=example_input.shape)]
    )

    coreml_model.save("MedusaSpeculativeHead.mlpackage")
    print("Exported MedusaSpeculativeHead.mlpackage")

if __name__ == "__main__":
    main()

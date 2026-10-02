import re

with open("mlx_rlcd_server.py", "r") as f:
    content = f.read()

content = content.replace(
    'load("/Users/jack/Downloads/rlcd-router/real_model_weights/mlx_8bit_chat", lazy=True)',
    'load("/Users/jack/Downloads/rlcd-router/real_model_weights", lazy=True)'
)

# Mixtral has 8 experts. The head was initialized for Qwen with 60 experts.
content = content.replace(
    'MedusaSpeculativeHead(\n            input_dim=2048,\n            num_deep_layers=20,\n            num_experts=60,\n            num_horizons=3\n        )',
    'MedusaSpeculativeHead(\n            input_dim=4096,\n            num_deep_layers=32,\n            num_experts=8,\n            num_horizons=3\n        )'
)

with open("mlx_rlcd_server.py", "w") as f:
    f.write(content)

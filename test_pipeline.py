import torch
from pathlib import Path
from src.config import PipelineConfig
from src.pipeline import _generate_synthetic_data
from src.data.dataset import save_dataset_safetensors

config = PipelineConfig()
config.model.hidden_size = 64
config.model.num_experts = 16
config.model.num_experts_per_tok = 4
config.model.deep_layers = tuple(range(3, 7))
config.training.input_dim = 64
config.training.output_dim_per_horizon = 4 * 16

train_data, calib_data = _generate_synthetic_data(config, torch.device('cpu'))

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

save_dataset_safetensors(train_data, out_dir / "train_data.safetensors")
save_dataset_safetensors(calib_data, out_dir / "calib_data.safetensors")
print("Saved mock data")

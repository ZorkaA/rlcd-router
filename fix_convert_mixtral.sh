#!/bin/bash
set -e

# Free up memory
pkill -f mlx_rlcd_server.py || true
sleep 2

# Set cache to SSD to avoid filling up the main disk
export HF_HOME="/Volumes/SSK SSD/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Downloading mistralai/Mixtral-8x7B-Instruct-v0.1 safely..."
python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download('mistralai/Mixtral-8x7B-Instruct-v0.1', max_workers=1, cache_dir=os.environ['HF_HOME'])"

echo "Converting to 8-bit MLX format..."
python3 -m mlx_lm.convert --hf-path mistralai/Mixtral-8x7B-Instruct-v0.1 -q --q-bits 8 --mlx-path "/Volumes/SSK SSD/rlcd_models/mlx_8bit_mixtral"

echo "Updating symlink..."
rm -rf /Users/jack/Downloads/rlcd-router/real_model_weights
ln -s "/Volumes/SSK SSD/rlcd_models/mlx_8bit_mixtral" /Users/jack/Downloads/rlcd-router/real_model_weights

echo "Starting server..."
nohup python3 mlx_rlcd_server.py > server.log 2>&1 &
echo "Server restarted successfully!"

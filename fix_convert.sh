#!/bin/bash
echo "Fetching missing metadata files..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen1.5-MoE-A2.7B-Chat')"
echo "Resuming conversion..."
python3 -m mlx_lm.convert --hf-path Qwen/Qwen1.5-MoE-A2.7B-Chat -q --q-bits 8 --mlx-path "/Volumes/SSK SSD/rlcd_models/mlx_8bit_chat"
rm -rf /Users/jack/Downloads/rlcd-router/real_model_weights
ln -s "/Volumes/SSK SSD/rlcd_models/mlx_8bit_chat" /Users/jack/Downloads/rlcd-router/real_model_weights
pkill -f mlx_rlcd_server.py
sleep 2
nohup python3 mlx_rlcd_server.py > server.log 2>&1 &
echo "Server restarted successfully!"

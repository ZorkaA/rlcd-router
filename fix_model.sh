#!/bin/bash
echo "Downloading and converting Chat model..."
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 -m mlx_lm.convert --hf-path Qwen/Qwen1.5-MoE-A2.7B-Chat -q --q-bits 8 --mlx-path "/Volumes/SSK SSD/rlcd_models/mlx_8bit_chat"
echo "Done converting!"
rm -rf /Users/jack/Downloads/rlcd-router/real_model_weights
ln -s "/Volumes/SSK SSD/rlcd_models/mlx_8bit_chat" /Users/jack/Downloads/rlcd-router/real_model_weights
pkill -f mlx_rlcd_server.py
sleep 2
nohup python3 mlx_rlcd_server.py > server.log 2>&1 &
echo "Server restarted!"

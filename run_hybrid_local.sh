#!/bin/bash
# Quick hybrid test — run on a GPU node
# Usage: bash run_hybrid_local.sh

set -euo pipefail
cd ~/SkyRL
source ~/SkyRL/skyrl-train/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$HOME/SkyRL/skyrl-gym"

python skyrl-train/examples/MeMo/play_textworld_hybrid.py \
    --data_file ~/data/textworld_memo/test.parquet \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --gpt_model gpt-5-mini \
    --memory_window 5 \
    --max_games 3 \
    --max_turns 50 \
    --batch_size 3 \
    --gpt_workers 4 \
    --output_dir ~/data/hybrid_test_output

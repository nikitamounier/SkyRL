#!/bin/bash
# Run TextWorld MeMo training locally (no slurm) with logging

set -euo pipefail

# Create timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_run_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Logs will be saved to: $LOG_FILE"
echo "Press Ctrl+C to stop, or run in background with: ./run_training_local.sh &"
echo ""

# Set environment variables
export DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export NUM_GPUS="${NUM_GPUS:-1}"

# Run training with output to both terminal and log file
# Must run from skyrl-train directory (not examples/MeMo) so Ray can find pyproject.toml
cd /home/parsaidp/SkyRL/skyrl-train
bash examples/MeMo/run_textworld_memo_train.sh 2>&1 | tee "/home/parsaidp/SkyRL/$LOG_FILE"

echo ""
echo "Training finished at $(date)"
echo "Logs saved to: $LOG_FILE"

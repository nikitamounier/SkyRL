#!/bin/bash
# Run TextWorld MeMo training in background (survives terminal disconnect)

set -euo pipefail

# Create timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_run_${TIMESTAMP}.log"
PID_FILE="training.pid"

# Set environment variables
export DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export NUM_GPUS="${NUM_GPUS:-1}"

echo "Starting training in background at $(date)"
echo "Logs: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To stop: kill \$(cat $PID_FILE)"
echo ""

# Run in background with nohup
# Must run from skyrl-train directory (not examples/MeMo) so Ray can find pyproject.toml
cd /home/parsaidp/SkyRL/skyrl-train
nohup bash examples/MeMo/run_textworld_memo_train.sh > "/home/parsaidp/SkyRL/$LOG_FILE" 2>&1 &

# Save PID
echo $! > "/home/parsaidp/SkyRL/$PID_FILE"

echo "Training started with PID: $(cat /home/parsaidp/SkyRL/$PID_FILE)"
echo "Monitor with: tail -f /home/parsaidp/SkyRL/$LOG_FILE"

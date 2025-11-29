#!/bin/bash
# Simple script to run training locally on the cluster (no SLURM, no Modal)
# Useful for testing or interactive sessions
# Usage: bash run_train_local.sh

set -euo pipefail

# Add uv to PATH (installed at ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Change to the SkyRL repository root directory
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up environment variables
export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"

# Configure Ray to use uv (required for SkyRL)
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Navigate to skyrl-train directory
cd skyrl-train

# Set up Ray cluster
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RAY_ADDRESS

# Set data and checkpoint directories
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_modal_text}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/gsm8k_0.5b_modal_text_lora_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-console}"  # Use console for local testing

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Verify dataset exists
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
    echo "ERROR: Dataset files not found in $DATA_DIR"
    echo "Please run the data preparation step first:"
    echo "  cd skyrl-train"
    echo "  uv run --isolated --extra transformers --extra datasets python examples/modalities/prepare_modal_text_split.py \\"
    echo "    --model_path Qwen/Qwen3-0.6B \\"
    echo "    --output_dir $DATA_DIR \\"
    echo "    --placeholder_token '<|image_pad|>'"
    exit 1
fi

echo "========================================="
echo "Starting training (local, no SLURM, no Modal)"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Logger: $LOGGER"
echo "========================================="

# Export environment variables for the training script
export DATA_DIR
export CKPT_DIR
export NUM_GPUS
export LOGGER

# Run the training script
bash examples/modalities/run_qwen2_5_0.5b_gsm8k_modal_text.sh

echo "========================================="
echo "Training completed successfully"
echo "Checkpoints saved to: $CKPT_DIR"
echo "========================================="


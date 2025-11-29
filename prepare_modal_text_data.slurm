#!/bin/bash
#SBATCH --job-name=prepare_modal_text
#SBATCH --output=prepare_modal_text_%j.out
#SBATCH --error=prepare_modal_text_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# This script prepares the modal text dataset
# Runs directly on the cluster (no Modal required)

set -euo pipefail

# Add uv to PATH (installed at ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Change to the SkyRL repository root directory
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

# Set up environment variables
export SKYRL_REPO_ROOT="$HOME/SkyRL"

# Navigate to skyrl-train directory
cd skyrl-train

# Set data directory
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_modal_text}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|image_pad|>}"

# Create output directory
mkdir -p "$DATA_DIR"

echo "========================================="
echo "Preparing modal text dataset"
echo "Model: $MODEL_PATH"
echo "Output: $DATA_DIR"
echo "Placeholder: $PLACEHOLDER_TOKEN"
echo "========================================="

# Run the dataset preparation script
# transformers and datasets are already in the project dependencies, so we don't need --extra
uv run python examples/modalities/prepare_modal_text_split.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$DATA_DIR" \
  --placeholder_token "$PLACEHOLDER_TOKEN"

echo "========================================="
echo "Dataset preparation completed successfully"
echo "Output written to: $DATA_DIR"
echo "========================================="

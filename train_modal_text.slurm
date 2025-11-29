#!/bin/bash
#SBATCH --job-name=train_modal_text
#SBATCH --output=train_modal_text_%j.out
#SBATCH --error=train_modal_text_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# This script runs the training directly on the cluster (no Modal)
# Runs the training script with 1 GPU

set -euo pipefail

# Add uv to PATH (installed at ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Change to the SkyRL repository root directory
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

# Set up environment variables
export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"

# Configure Ray to use uv (required for SkyRL)
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Navigate to skyrl-train directory
cd skyrl-train

# Set data and checkpoint directories (set NUM_GPUS first, needed for Ray)
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_modal_text}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/gsm8k_0.5b_modal_text_lora_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-wandb}"

# Use the LOCAL cached path for the model instead of HuggingFace repo ID
# This is required because PromptEmbeddingBuilder needs local safetensors files
MODEL_PATH="${MODEL_PATH:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"

# Ensure skyrl-gym exists for uv editable resolution (needed for Ray workers)
if [ ! -d "skyrl-gym" ] && [ -d "../skyrl-gym" ]; then
    echo "Copying ../skyrl-gym into working_dir for uv packaging"
    cp -r ../skyrl-gym .
fi

# Don't set RAY_ADDRESS - this allows ray.init() to start a fresh local cluster
# When RAY_ADDRESS is unset, ray.init() will automatically start a new cluster
# instead of trying to connect to an existing one
unset RAY_ADDRESS

# Create checkpoint directory
mkdir -p "$CKPT_DIR"

# Verify dataset exists
if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
    echo "ERROR: Dataset files not found in $DATA_DIR"
    echo "Please run prepare_modal_text_data.slurm first"
    exit 1
fi

echo "========================================="
echo "Starting training"
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
export MODEL_PATH
export WANDB_ENTITY="goodarzilab"

# Run the training script
bash examples/modalities/run_qwen2_5_0.5b_gsm8k_modal_text.sh

echo "========================================="
echo "Training completed successfully"
echo "Checkpoints saved to: $CKPT_DIR"
echo "========================================="


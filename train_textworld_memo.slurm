#!/bin/bash
#SBATCH --job-name=train_textworld_memo
#SBATCH --output=train_textworld_memo_%j.out
#SBATCH --error=train_textworld_memo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/textworld_memo_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-wandb}"

# Use repo id by default; override with a local snapshot path if desired.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
LOCAL_SNAPSHOT_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
if [[ "$MODEL_PATH" == "Qwen/Qwen3-4B-Instruct-2507" && -d "$LOCAL_SNAPSHOT_PATH" ]]; then
    MODEL_PATH="$LOCAL_SNAPSHOT_PATH"
fi
MEMORY_DIM="${MEMORY_DIM:-2560}"

if [ ! -d "skyrl-train/skyrl-gym" ] && [ -d "skyrl-gym" ]; then
    echo "Copying skyrl-gym into skyrl-train for uv packaging"
    cp -r skyrl-gym skyrl-train/
fi

unset RAY_ADDRESS

mkdir -p "$CKPT_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
    echo "ERROR: Dataset files not found in $DATA_DIR"
    echo "Please run prepare_textworld_data.slurm first"
    exit 1
fi

echo "========================================="
echo "Starting TextWorld MeMo training"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Memory dim: $MEMORY_DIM"
echo "========================================="

export DATA_DIR CKPT_DIR NUM_GPUS LOGGER MODEL_PATH MEMORY_DIM
export WANDB_ENTITY="${WANDB_ENTITY:-goodarzilab}"

# Optional: load W&B API key from file if env var is not set.
# Put your key in: ~/.wandb_api_key (single line) OR set WANDB_API_KEY in the environment.
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

# If a cache path ending in /snapshots/main was provided, resolve it to the latest snapshot.
if [[ "$MODEL_PATH" == *"/snapshots/main" && ! -d "$MODEL_PATH" ]]; then
    BASE_DIR="${MODEL_PATH%/snapshots/main}"
    if [[ -d "$BASE_DIR/snapshots" ]]; then
        LATEST_SNAPSHOT="$(ls -1t "$BASE_DIR/snapshots" | head -n 1)"
        if [[ -n "$LATEST_SNAPSHOT" ]]; then
            MODEL_PATH="$BASE_DIR/snapshots/$LATEST_SNAPSHOT"
            export MODEL_PATH
        fi
    fi
fi

cd skyrl-train
bash examples/MeMo/run_textworld_memo_train.sh

echo "========================================="
echo "Training completed successfully"
echo "Checkpoints saved to: $CKPT_DIR"
echo "========================================="

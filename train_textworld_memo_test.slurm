#!/bin/bash
#SBATCH --job-name=train_tw_test
#SBATCH --output=train_textworld_memo_test_%j.out
#SBATCH --error=train_textworld_memo_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

# Activate nikitamounier's virtual environment
source /home/nikitamounier/SkyRL/.venv/bin/activate

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
echo "Starting TextWorld MeMo TEST training (2 epochs)"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Model path: $MODEL_PATH"
echo "Memory dim: $MEMORY_DIM"
echo "========================================="

export DATA_DIR CKPT_DIR NUM_GPUS LOGGER MODEL_PATH MEMORY_DIM
export WANDB_ENTITY="${WANDB_ENTITY:-dsp-team}"

# Optional: load W&B API key from file if env var is not set.
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

cd skyrl-train
bash examples/MeMo/run_textworld_memo_train.sh \
  trainer.epochs=2 \
  trainer.eval_interval=20 \
  trainer.ckpt_interval=20 \
  trainer.run_name="textworld_memo_test_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "Test training completed"
echo "========================================="

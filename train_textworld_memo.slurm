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
#SBATCH --mem=96G
#
# Usage:
#   sbatch train_textworld_memo.slurm                                  # gpu, 24h, 50 epochs
#   sbatch -p preemptible train_textworld_memo.slurm                   # preemptible partition
#   sbatch -p preemptible -t 02:00:00 train_textworld_memo.slurm      # short test run
#   EPOCHS=2 sbatch train_textworld_memo.slurm                        # quick 2-epoch test
#   RESUME_MODE=auto sbatch train_textworld_memo.slurm                # resume from checkpoint

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
source "${SKYRL_REPO_ROOT}/skyrl-train/.venv/bin/activate"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
CKPT_DIR="${CKPT_DIR:-/large_storage/goodarzilab/parsaidp/MeMo/ckpts/textworld_memo}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-wandb}"
EPOCHS="${EPOCHS:-50}"
RESUME_MODE="${RESUME_MODE:-null}"
MEMORY_WINDOW="${MEMORY_WINDOW:-3}"
STEP_PENALTY="${STEP_PENALTY:-0.0}"
EFFICIENCY_BONUS="${EFFICIENCY_BONUS:-0.0}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
N_SAMPLES="${N_SAMPLES:-5}"

# Resolve model path: prefer local cache, fall back to HF repo id.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CACHE_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
if [[ "$MODEL_PATH" == "Qwen/Qwen3-4B-Instruct-2507" && -d "$CACHE_DIR" ]]; then
    LATEST="$(ls -1t "$CACHE_DIR" | head -n1)"
    [[ -n "$LATEST" ]] && MODEL_PATH="$CACHE_DIR/$LATEST"
fi
MEMORY_DIM="${MEMORY_DIM:-2560}"

if [ ! -d "skyrl-train/skyrl-gym" ] && [ -d "skyrl-gym" ]; then
    cp -r skyrl-gym skyrl-train/
fi

unset RAY_ADDRESS
mkdir -p "$CKPT_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/validation.parquet" ]; then
    echo "ERROR: Dataset files not found in $DATA_DIR"
    echo "Run: sbatch prepare_textworld_data.slurm"
    exit 1
fi

echo "========================================="
echo "TextWorld MeMo Training"
echo "  Data:       $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  GPUs:       $NUM_GPUS"
echo "  Model:      $MODEL_PATH"
echo "  Epochs:     $EPOCHS"
echo "  Window:     $MEMORY_WINDOW"
echo "  Advantage:  $ADVANTAGE_ESTIMATOR"
echo "  N_Samples:  $N_SAMPLES"
echo "  StepPenalty: $STEP_PENALTY"
echo "  EffBonus:   $EFFICIENCY_BONUS"
echo "  Resume:     $RESUME_MODE"
echo "========================================="

export DATA_DIR CKPT_DIR NUM_GPUS LOGGER MODEL_PATH MEMORY_DIM MEMORY_WINDOW STEP_PENALTY EFFICIENCY_BONUS LR ADVANTAGE_ESTIMATOR N_SAMPLES
export WANDB_ENTITY="${WANDB_ENTITY:-dsp-team}"

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

cd skyrl-train
bash examples/MeMo/run_textworld_memo_train.sh \
  trainer.epochs="$EPOCHS" \
  trainer.resume_mode="$RESUME_MODE"

echo "========================================="
echo "Training completed. Checkpoints: $CKPT_DIR"
echo "========================================="

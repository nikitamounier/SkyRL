#!/bin/bash
#SBATCH --job-name=eval_tw_baseline
#SBATCH --output=eval_tw_baseline_%j.out
#SBATCH --error=eval_tw_baseline_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#
# Evaluate Qwen3-4B baseline on TextWorld (no memory module).
# Batched inference: all active games generate in one vLLM call per turn.
#
# Usage:
#   sbatch eval_textworld_baseline.slurm                                    # eval test set
#   DATA_FILE=~/data/textworld_memo/validation.parquet sbatch eval_textworld_baseline.slurm
#   DATA_FILE=~/data/textworld_memo/train.parquet MAX_GAMES=50 sbatch eval_textworld_baseline.slurm

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
source "${SKYRL_REPO_ROOT}/skyrl-train/.venv/bin/activate"

DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
DATA_FILE="${DATA_FILE:-$DATA_DIR/test.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_GAMES="${MAX_GAMES:-}"
BATCH_SIZE="${BATCH_SIZE:-20}"

echo "========================================="
echo "TextWorld Baseline Evaluation (Batched)"
echo "  Data:       $DATA_FILE"
echo "  Model:      $MODEL_PATH"
echo "  Turns:      $MAX_TURNS"
echo "  Batch size: $BATCH_SIZE"
echo "  Games:      ${MAX_GAMES:-all}"
echo "========================================="

EXTRA_ARGS=""
if [[ -n "$MAX_GAMES" ]]; then
    EXTRA_ARGS="--max_games $MAX_GAMES"
fi

python skyrl-train/examples/MeMo/eval_textworld_baseline.py \
    --data_file "$DATA_FILE" \
    --model_path "$MODEL_PATH" \
    --max_turns "$MAX_TURNS" \
    --max_generate_length 256 \
    --temperature 0.6 \
    --gpu_memory 0.95 \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS

echo "========================================="
echo "Evaluation complete"
echo "========================================="

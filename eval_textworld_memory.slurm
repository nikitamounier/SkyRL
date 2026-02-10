#!/bin/bash
#SBATCH --job-name=eval_tw_memory
#SBATCH --output=eval_tw_memory_%j.out
#SBATCH --error=eval_tw_memory_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#
# Evaluate TextWorld with SFT-trained Memory module.
# Pure evaluation — no training, no weight updates.
#
# Usage:
#   CHECKPOINT=~/data/memory_sft.pt MEMORY_WINDOW=5 sbatch eval_textworld_memory.slurm
#   CHECKPOINT=~/data/memory_sft_w3.pt MEMORY_WINDOW=3 DATA_SPLIT=validation sbatch eval_textworld_memory.slurm

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
source "${SKYRL_REPO_ROOT}/skyrl-train/.venv/bin/activate"

export PYTHONUNBUFFERED=1

DATA_DIR="${DATA_DIR:-/large_storage/goodarzilab/parsaidp/MeMo/data/textworld_memo_v3}"
DATA_SPLIT="${DATA_SPLIT:-train}"
DATA_FILE="${DATA_FILE:-$DATA_DIR/${DATA_SPLIT}.parquet}"
CHECKPOINT="${CHECKPOINT:-}"
MEMORY_WINDOW="${MEMORY_WINDOW:-5}"
MAX_TURNS="${MAX_TURNS:-50}"
MAX_GAMES="${MAX_GAMES:-}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
BATCH_SIZE="${BATCH_SIZE:-10}"

echo "========================================="
echo "TextWorld Memory Evaluation (Pure Eval)"
echo "  Data:       $DATA_FILE"
echo "  Checkpoint: ${CHECKPOINT:-RANDOM}"
echo "  Window:     $MEMORY_WINDOW"
echo "  Max turns:  $MAX_TURNS"
echo "  Games:      ${MAX_GAMES:-all}"
echo "  Embed dev:  $EMBEDDING_DEVICE"
echo "  GPU util:   $GPU_MEM_UTIL"
echo "========================================="

EXTRA_ARGS=""
if [[ -n "${CHECKPOINT}" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --checkpoint $CHECKPOINT"
fi
if [[ -n "${MAX_GAMES}" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_games $MAX_GAMES"
fi

python skyrl-train/examples/MeMo/eval_textworld_memory.py \
    --data_file "$DATA_FILE" \
    --memory_window "$MEMORY_WINDOW" \
    --max_turns "$MAX_TURNS" \
    --embedding_device "$EMBEDDING_DEVICE" \
    --gpu_memory "$GPU_MEM_UTIL" \
    --batch_size "$BATCH_SIZE" \
    $EXTRA_ARGS

echo "========================================="
echo "Evaluation complete"
echo "========================================="

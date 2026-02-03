#!/bin/bash
#SBATCH --job-name=prepare_textworld
#SBATCH --output=prepare_textworld_%j.out
#SBATCH --error=prepare_textworld_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
GAMES_DIR="${GAMES_DIR:-$DATA_DIR/games}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|image_pad|>}"
MAX_MEMORY_DOCS="${MAX_MEMORY_DOCS:-4}"
NUM_GAMES="${NUM_GAMES:-10}"
SIZE="${SIZE:-SMALL}"

mkdir -p "$DATA_DIR"

echo "========================================="
echo "Preparing TextWorld dataset"
echo "Output: $DATA_DIR"
echo "Games: $GAMES_DIR"
echo "Size: $SIZE | Num games: $NUM_GAMES"
echo "Placeholder: $PLACEHOLDER_TOKEN | Max docs: $MAX_MEMORY_DOCS"
echo "========================================="

uv run python skyrl-train/examples/MeMo/prepare_textworld_dataset.py \
  --output_dir "$DATA_DIR" \
  --games_dir "$GAMES_DIR" \
  --size "$SIZE" \
  --num_games "$NUM_GAMES" \
  --placeholder_token "$PLACEHOLDER_TOKEN" \
  --max_memory_docs "$MAX_MEMORY_DOCS"

echo "========================================="
echo "Dataset preparation complete"
echo "========================================="

#!/bin/bash
#SBATCH --job-name=prepare_textworld
#SBATCH --output=prepare_textworld_%j.out
#SBATCH --error=prepare_textworld_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Generate the canonical TextWorld MeMo dataset (v2).
# Output: $DATA_DIR/{train,validation,test}.parquet + dataset_stats.json
#
# Usage:
#   sbatch prepare_textworld_data.slurm                     # defaults (400 games, medium+hard)
#   NUM_GAMES=100 sbatch prepare_textworld_data.slurm       # smaller dataset

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
source "${SKYRL_REPO_ROOT}/skyrl-train/.venv/bin/activate"

# --- Config ---
DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
GAMES_DIR="${GAMES_DIR:-$DATA_DIR/games_v2}"
NUM_GAMES="${NUM_GAMES:-400}"
EPISODES_PER_GAME="${EPISODES_PER_GAME:-2}"
DIFFICULTY_MIX="${DIFFICULTY_MIX:-medium:0.30,hard:0.70}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|image_pad|>}"
MAX_MEMORY_DOCS="${MAX_MEMORY_DOCS:-20}"
SEED="${SEED:-42}"

mkdir -p "$DATA_DIR" "$GAMES_DIR"

echo "========================================="
echo "Generating TextWorld MeMo dataset (v2)"
echo "  Output:     $DATA_DIR"
echo "  Games dir:  $GAMES_DIR"
echo "  Num games:  $NUM_GAMES"
echo "  Episodes/game: $EPISODES_PER_GAME"
echo "  Difficulty: $DIFFICULTY_MIX"
echo "  Max docs:   $MAX_MEMORY_DOCS"
echo "  Seed:       $SEED"
echo "========================================="

python skyrl-train/examples/MeMo/prepare_textworld_dataset_v2.py \
  --output_dir "$DATA_DIR" \
  --games_dir "$GAMES_DIR" \
  --num_games "$NUM_GAMES" \
  --episodes_per_game "$EPISODES_PER_GAME" \
  --difficulty_mix "$DIFFICULTY_MIX" \
  --placeholder_token "$PLACEHOLDER_TOKEN" \
  --max_memory_docs "$MAX_MEMORY_DOCS" \
  --seed "$SEED" \
  --train_ratio 0.85 \
  --val_ratio 0.10 \
  --prompt_variants "v1_full,v2_memory,v3_strategic,v4_concise" \

echo "========================================="
echo "Dataset generation complete"
echo "========================================="
cat "$DATA_DIR/dataset_stats.json"

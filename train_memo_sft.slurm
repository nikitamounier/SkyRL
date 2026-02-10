#!/bin/bash
#SBATCH --job-name=memo_sft
#SBATCH --output=memo_sft_%j.out
#SBATCH --error=memo_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate skyenv

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

TRAIN_DIR="${TRAIN_DIR:-/large_storage/goodarzilab/parsaidp/MeMo/data/memo_sft_train}"
VAL_DIR="${VAL_DIR:-/large_storage/goodarzilab/parsaidp/MeMo/data/memo_sft_val}"
OUTPUT_DIR="${OUTPUT_DIR:-/large_storage/goodarzilab/parsaidp/MeMo/ckpts/memo_sft}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

echo "========================================="
echo "MeMo SFT Training (Memory Module)"
echo "  Train: $TRAIN_DIR"
echo "  Val:   $VAL_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $MAX_EPOCHS"
echo "  Batch:  $BATCH_SIZE"
echo "  LR:     $LR"
echo "========================================="

cd /home/parsaidp/MeMo

python embed_and_train_textworld.py \
  --train_dir "$TRAIN_DIR" \
  --val_dir "$VAL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --max_epochs "$MAX_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --max_length "$MAX_LENGTH"

echo "========================================="
echo "Training complete. Checkpoints: $OUTPUT_DIR"
echo "========================================="

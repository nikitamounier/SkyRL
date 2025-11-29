#!/bin/bash
# Wrapper script to submit both jobs with proper dependencies

set -euo pipefail

# Configuration - adjust as needed
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_modal_text}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/gsm8k_0.5b_modal_text_lora_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-wandb}"

echo "========================================="
echo "Submitting modal text training pipeline"
echo "========================================="
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Logger: $LOGGER"
echo "========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Submit data preparation job
echo "Step 1: Submitting data preparation job..."
PREPARE_JOB_ID=$(sbatch --parsable \
    --export=ALL,DATA_DIR="$DATA_DIR" \
    "$SCRIPT_DIR/prepare_modal_text_data.slurm")

echo "  Data preparation job ID: $PREPARE_JOB_ID"
echo ""

# Submit training job with dependency on data preparation
echo "Step 2: Submitting training job (will start after data prep completes)..."
TRAIN_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$PREPARE_JOB_ID \
    --export=ALL,DATA_DIR="$DATA_DIR",CKPT_DIR="$CKPT_DIR",NUM_GPUS="$NUM_GPUS",LOGGER="$LOGGER" \
    "$SCRIPT_DIR/train_modal_text.slurm")

echo "  Training job ID: $TRAIN_JOB_ID"
echo ""

echo "========================================="
echo "Pipeline submitted successfully!"
echo "========================================="
echo "Data preparation job: $PREPARE_JOB_ID"
echo "Training job: $TRAIN_JOB_ID (depends on $PREPARE_JOB_ID)"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "View logs in: prepare_modal_text_*.out/err and train_modal_text_*.out/err"
echo "========================================="


#!/bin/bash
#SBATCH --job-name=cell_grpo5k
#SBATCH --output=/home/parsaidp/SkyRL/cell_grpo5k_%j.out
#SBATCH --error=/home/parsaidp/SkyRL/cell_grpo5k_%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --account=cell_reason
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#
# Non-reasoning GRPO (minimal answeronly 9B) on a 5k v7.5 subset, 4 GPUs, co-train projection.
# Logs to the goodarzilab W&B.

set -euo pipefail

ENV=/home/adibvafa/miniconda/envs/bioreason_cell
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
if [ -f "${ENV}/lib/libstdc++.so.6" ]; then export LD_PRELOAD="${ENV}/lib/libstdc++.so.6"; fi
export CUDA_HOME="$ENV"
export PYTHONPATH=/home/parsaidp/cellrl_pydeps:/home/parsaidp/SkyRL/skyrl-train:/home/parsaidp/SkyRL/skyrl-gym
export HF_HOME=/home/parsaidp/.cache/hf_cellrl
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B -> goodarzilab team
export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
export WANDB_ENTITY="${WANDB_ENTITY:-goodarzilab}"

export DATA_DIR="${DATA_DIR:-$HOME/data/cell_pathway_v7_5_min_5k}"
export MODEL_PATH="${MODEL_PATH:-/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_answeronly_9b_converted}"
export NUM_GPUS="${NUM_GPUS:-4}"
export LOGGER="${LOGGER:-wandb}"

cd /home/parsaidp/SkyRL/skyrl-train
bash examples/modalities/run_cell_pathway_grpo_5k.sh

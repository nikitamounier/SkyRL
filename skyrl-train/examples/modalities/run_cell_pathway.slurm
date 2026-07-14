#!/bin/bash
#SBATCH --job-name=cell_grpo
#SBATCH --output=/home/parsaidp/SkyRL/cell_grpo_%j.out
#SBATCH --error=/home/parsaidp/SkyRL/cell_grpo_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --account=cell_reason
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#
# GRPO on BioReasonCell gene->pathway with the STATE cell-embedding modality,
# running SkyRL on the qwen3.5 stack (vLLM 0.19.1 + transformers 5.5) via adibvafa's
# bioreason_cell env + layered SkyRL deps in /home/parsaidp/cellrl_pydeps.

set -euo pipefail

ENV=/home/adibvafa/miniconda/envs/bioreason_cell
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
if [ -f "${ENV}/lib/libstdc++.so.6" ]; then export LD_PRELOAD="${ENV}/lib/libstdc++.so.6"; fi
export CUDA_HOME="$ENV"
# deps dir first (ray/hydra/func_timeout/pillow/…), then SkyRL editable checkouts
export PYTHONPATH=/home/parsaidp/cellrl_pydeps:/home/parsaidp/SkyRL/skyrl-train:/home/parsaidp/SkyRL/skyrl-gym
export HF_HOME=/home/parsaidp/.cache/hf_cellrl
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

export DATA_DIR="${DATA_DIR:-$HOME/data/cell_pathway_v7_5}"
export MODEL_PATH="${MODEL_PATH:-/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_full_answeronly_9b_converted}"
export NUM_GPUS="${NUM_GPUS:-1}"
export LOGGER="${LOGGER:-console}"

cd /home/parsaidp/SkyRL/skyrl-train
bash examples/modalities/run_cell_pathway_grpo.sh

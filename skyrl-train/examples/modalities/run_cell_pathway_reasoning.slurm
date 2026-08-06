#!/bin/bash
#SBATCH --job-name=cell_reason
#SBATCH --output=/home/parsaidp/SkyRL/cell_reason_%j.out
#SBATCH --error=/home/parsaidp/SkyRL/cell_reason_%j.out
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
# Reasoning checkpoint (minimal-prompt, produces CoT). Requires the pipeline fixes:
# key-remap (automatic in model_wrapper), attention+MLP-only LoRA (default), on-policy logprobs.
export MODEL_PATH=/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_9b_converted
export RUN_NAME=cell_grpo_reasoning_maxtok_min5k
export CKPT_DIR=$HOME/data/ckpts/cell_grpo_reasoning_maxtok
export LR=2.0e-6
export USE_KL=false
# Higher generation length to match the BioReasonCell evals (max_new_tokens=8192) so full CoT
# traces aren't truncated before the \boxed{} answer.
export MAX_GEN=8192
export MAX_MODEL_LEN=16384
# Long CoT sequences are memory-heavy -> small micro-batches.
export MICRO_TRAIN=1
export MICRO_FWD=2
bash examples/modalities/run_cell_pathway_grpo_5k.sh

#!/usr/bin/env bash
set -euo pipefail
set -x

# Train MeMo memory module (LLM frozen) on TextWorld using SkyRL modalities.

DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/textworld_memo_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-console}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

# Optional: load W&B API key from file if env var is not set.
# Put your key in: ~/.wandb_api_key (single line) OR set WANDB_API_KEY in the environment.
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

# If a cache path ending in /snapshots/main was provided, resolve it to the latest snapshot.
if [[ "$MODEL_PATH" == *"/snapshots/main" && ! -d "$MODEL_PATH" ]]; then
  BASE_DIR="${MODEL_PATH%/snapshots/main}"
  if [[ -d "$BASE_DIR/snapshots" ]]; then
    LATEST_SNAPSHOT="$(ls -1t "$BASE_DIR/snapshots" | head -n 1)"
    if [[ -n "$LATEST_SNAPSHOT" ]]; then
      MODEL_PATH="$BASE_DIR/snapshots/$LATEST_SNAPSHOT"
      export MODEL_PATH
    fi
  fi
fi

# Base model (must exist locally if using PromptEmbeddingBuilder/embedding weights)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|image_pad|>}"

# Memory module params (set MEMORY_DIM to the base model hidden size)
MEMORY_DIM="${MEMORY_DIM:-2560}"
NUM_MEMORIES="${NUM_MEMORIES:-8}"
MAX_MEMORY_DOCS="${MAX_MEMORY_DOCS:-20}"  # Keep all docs in 50-turn episodes (50/3 ≈ 16 docs)
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"

# Use our own venv with all dependencies installed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TRAIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SKYRL_TRAIN_ROOT/.venv/bin/activate"
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=textworld \
  environment.skyrl_gym.textworld.tokenizer_path="$MODEL_PATH" \
  environment.skyrl_gym.textworld.max_turns=50 \
  environment.skyrl_gym.textworld.memory_window=3 \
  environment.skyrl_gym.textworld.max_memory_docs=$MAX_MEMORY_DOCS \
  environment.skyrl_gym.textworld.max_doc_tokens=256 \
  environment.skyrl_gym.textworld.memory_modality_id="memo_memory" \
  +environment.skyrl_gym.textworld.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +environment.skyrl_gym.textworld.max_placeholder_tokens=$NUM_MEMORIES \
  generator.max_turns=50 \
  generator.batched=false \
  generator.refresh_modalities_each_step=true \
  generator.sampling_params.max_generate_length=64 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.model.freeze_base_model=true \
  trainer.policy.model.lora.rank=0 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=50 \
  trainer.eval_batch_size=4 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=50 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.max_input_length=$MAX_PROMPT_LENGTH \
  trainer.policy.optimizer_config.lr=5.0e-5 \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="textworld_memo" \
  trainer.run_name="textworld_memo_$(date +%Y%m%d_%H%M%S)" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  +modalities.memo_memory.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +modalities.memo_memory.max_placeholder_tokens=$NUM_MEMORIES \
  +modalities.memo_memory.encoder.target="skyrl_train.examples.modalities.memo_handlers:MemoTokenMemoryEncoder" \
  +modalities.memo_memory.encoder.kwargs.model_path="$MODEL_PATH" \
  +modalities.memo_memory.encoder.kwargs.embedding_dim=$MEMORY_DIM \
  +modalities.memo_memory.encoder.kwargs.output_dim=$MEMORY_DIM \
  +modalities.memo_memory.encoder.kwargs.num_memories=$NUM_MEMORIES \
  +modalities.memo_memory.encoder.kwargs.num_heads=8 \
  +modalities.memo_memory.encoder.kwargs.num_layers=1 \
  +modalities.memo_memory.encoder.kwargs.dropout=0.1 \
  +modalities.memo_memory.encoder.kwargs.memory_init="xavier_uniform" \
  +modalities.memo_memory.encoder.kwargs.max_doc_tokens=256 \
  +modalities.memo_memory.encoder.kwargs.memo_repo_root="/home/parsaidp/MeMo" \
  +modalities.memo_memory.projection.target="skyrl_train.examples.modalities.memo_handlers:IdentityProjection" \
  +modalities.memo_memory.projection.kwargs={} \
  +modalities.memo_memory.trainable.encoder=true \
  +modalities.memo_memory.trainable.projection=true \
  $@

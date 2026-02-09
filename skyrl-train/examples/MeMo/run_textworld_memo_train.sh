#!/usr/bin/env bash
set -euo pipefail
set -x

# Train MeMo memory module (LLM frozen) on TextWorld using SkyRL modalities.

# --- Configuration (override via env vars) ---
DATA_DIR="${DATA_DIR:-$HOME/data/textworld_memo}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/textworld_memo_ckpt}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-console}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

# Model
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|image_pad|>}"

# Memory module
MEMORY_DIM="${MEMORY_DIM:-2560}"
NUM_MEMORIES="${NUM_MEMORIES:-8}"
MAX_MEMORY_DOCS="${MAX_MEMORY_DOCS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MEMORY_WINDOW="${MEMORY_WINDOW:-3}"
STEP_PENALTY="${STEP_PENALTY:-0.0}"
EFFICIENCY_BONUS="${EFFICIENCY_BONUS:-0.0}"
MEMO_REPO_ROOT="${MEMO_REPO_ROOT:-$HOME/MeMo}"
MAX_TURNS="${MAX_TURNS:-50}"

# Training
LR="${LR:-5.0e-4}"
RUN_NAME="${RUN_NAME:-textworld_memo_$(date +%Y%m%d_%H%M%S)}"
RESUME_MODE="${RESUME_MODE:-null}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
N_SAMPLES="${N_SAMPLES:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"

# Load W&B API key from file if not set
if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
fi

# Resolve /snapshots/main to latest snapshot hash
if [[ "${MODEL_PATH}" == *"/snapshots/main" && ! -d "${MODEL_PATH}" ]]; then
  BASE_DIR="${MODEL_PATH%/snapshots/main}"
  if [[ -d "$BASE_DIR/snapshots" ]]; then
    LATEST_SNAPSHOT="$(ls -1t "$BASE_DIR/snapshots" | head -n 1)"
    [[ -n "$LATEST_SNAPSHOT" ]] && MODEL_PATH="$BASE_DIR/snapshots/$LATEST_SNAPSHOT"
  fi
fi

# --- Launch training ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_TRAIN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$SKYRL_TRAIN_ROOT/.venv/bin/activate"

python -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=textworld \
  environment.skyrl_gym.textworld.tokenizer_path="$MODEL_PATH" \
  environment.skyrl_gym.textworld.max_turns=$MAX_TURNS \
  environment.skyrl_gym.textworld.memory_window=$MEMORY_WINDOW \
  environment.skyrl_gym.textworld.max_memory_docs=$MAX_MEMORY_DOCS \
  environment.skyrl_gym.textworld.max_doc_tokens=256 \
  environment.skyrl_gym.textworld.memory_modality_id="memo_memory" \
  +environment.skyrl_gym.textworld.step_penalty=$STEP_PENALTY \
  +environment.skyrl_gym.textworld.efficiency_bonus=$EFFICIENCY_BONUS \
  +environment.skyrl_gym.textworld.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +environment.skyrl_gym.textworld.max_placeholder_tokens=$NUM_MEMORIES \
  generator.max_turns=$MAX_TURNS \
  generator.batched=false \
  generator.refresh_modalities_each_step=true \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.sampling_params.max_generate_length=128 \
  trainer.algorithm.advantage_estimator="$ADVANTAGE_ESTIMATOR" \
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
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=50 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.max_input_length=$MAX_PROMPT_LENGTH \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=$WARMUP_STEPS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  +generator.engine_init_kwargs.max_model_len=4224 \
  trainer.logger="$LOGGER" \
  trainer.project_name="textworld_memo" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.ckpt_path="$CKPT_DIR" \
  +modalities.memo_memory.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +modalities.memo_memory.max_placeholder_tokens=$NUM_MEMORIES \
  +modalities.memo_memory.encoder.target="skyrl_train.examples.modalities.memo_handlers:MemoSentenceEmbeddingEncoder" \
  +modalities.memo_memory.encoder.kwargs.embedding_model_name="Qwen/Qwen3-Embedding-4B" \
  +modalities.memo_memory.encoder.kwargs.embedding_dim=$MEMORY_DIM \
  +modalities.memo_memory.encoder.kwargs.output_dim=$MEMORY_DIM \
  +modalities.memo_memory.encoder.kwargs.num_memories=$NUM_MEMORIES \
  +modalities.memo_memory.encoder.kwargs.num_heads=8 \
  +modalities.memo_memory.encoder.kwargs.num_layers=1 \
  +modalities.memo_memory.encoder.kwargs.dropout=0.1 \
  +modalities.memo_memory.encoder.kwargs.memory_init="xavier_uniform" \
  +modalities.memo_memory.encoder.kwargs.embedding_device="cpu" \
  +modalities.memo_memory.encoder.kwargs.memo_repo_root="$MEMO_REPO_ROOT" \
  +modalities.memo_memory.projection.target="skyrl_train.examples.modalities.memo_handlers:IdentityProjection" \
  +modalities.memo_memory.projection.kwargs={} \
  +modalities.memo_memory.trainable.encoder=true \
  +modalities.memo_memory.trainable.projection=true \
  "$@"

#!/usr/bin/env bash
set -euo pipefail
set -x

# Non-reasoning GRPO on the v7.5 minimal-prompt task with the STATE cell-embedding modality.
# Model: answeronly (minimal) 9B — short summary + \boxed{dir}; reward = 3-class boxed match.
# Co-trains the cell projection alongside the LLM LoRA. 4 GPUs, colocated vLLM + FSDP.

DATA_DIR="${DATA_DIR:-$HOME/data/cell_pathway_v7_5_min_5k}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/cell_pathway_grpo_min5k}"
NUM_GPUS="${NUM_GPUS:-4}"
LOGGER="${LOGGER:-wandb}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

MODEL_PATH="${MODEL_PATH:-/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_answeronly_9b_converted}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|cell_pad|>}"
CELL_PROJ_PATH="${CELL_PROJ_PATH:-$MODEL_PATH/cell_projection.pt}"
TRAIN_PROJECTION="${TRAIN_PROJECTION:-true}"     # co-train the cell projection
RUN_NAME="${RUN_NAME:-cell_pathway_grpo_min5k}"
LR="${LR:-2.0e-6}"
USE_KL="${USE_KL:-false}"          # drop KL by default
MAX_GEN="${MAX_GEN:-512}"
MICRO_TRAIN="${MICRO_TRAIN:-4}"
MICRO_FWD="${MICRO_FWD:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
TRAIN_BATCH="${TRAIN_BATCH:-64}"
MINI_BATCH="${MINI_BATCH:-16}"
EPOCHS="${EPOCHS:-2}"
EVAL_BEFORE="${EVAL_BEFORE:-false}"
# Enable vLLM rollout logprobs so the trainer logs policy/rollout_train_prob_diff_mean =
# exp(rollout_logprob - train_logprob) on response tokens. ~1.0 <=> inference & training stacks
# hold identical weights (true on-policy); a large deviation flags a weight-sync mismatch.
LOGPROBS="${LOGPROBS:-0}"   # 0 = logprob of the chosen token only (SkyRL requires 0, not >0)
USE_PACKING="${USE_PACKING:-true}"   # sample packing; suspect for train-vs-rollout logprob mismatch
# Restrict LoRA to standard transformer attention+MLP projections. "all-linear" also targets
# Qwen3.5's SSM/linear-attention in_proj_*/out_proj layers, whose vLLM LoRA kernels hang during
# generation once the delta is non-zero (the previously-empty adapter never exercised them).
TARGET_MODULES="${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"

python -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.target_modules="$TARGET_MODULES" \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=$EVAL_BEFORE \
  trainer.eval_interval=100000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH \
  trainer.policy_mini_batch_size=$MINI_BATCH \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FWD \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN \
  trainer.ckpt_interval=40 \
  trainer.max_prompt_length=8192 \
  trainer.use_sample_packing=$USE_PACKING \
  generator.sampling_params.max_generate_length=$MAX_GEN \
  generator.sampling_params.temperature=${TEMP:-1.0} \
  generator.sampling_params.logprobs=$LOGPROBS \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.algorithm.use_kl_loss=$USE_KL \
  trainer.algorithm.kl_loss_coef=${KL_COEF:-0.001} \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=cell_pathway \
  generator.n_samples_per_prompt=${NSAMPLES:-16} \
  generator.gpu_memory_utilization=0.4 \
  generator.enforce_eager=true \
  +generator.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  +generator.engine_init_kwargs.language_model_only=true \
  +generator.engine_init_kwargs.disable_mrope=true \
  trainer.logger="$LOGGER" \
  trainer.project_name="cell_pathway_grpo" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  +modalities.state_mod.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +modalities.state_mod.max_placeholder_tokens=1 \
  +modalities.state_mod.encoder.target="skyrl_train.examples.modalities.state_cell_handlers:StatePrecomputedEncoder" \
  +modalities.state_mod.encoder.kwargs.embedding_dim=2058 \
  +modalities.state_mod.projection.target="skyrl_train.examples.modalities.state_cell_handlers:CellProjection" \
  +modalities.state_mod.projection.kwargs.model_path="$MODEL_PATH" \
  +modalities.state_mod.projection.kwargs.cell_projection_path="$CELL_PROJ_PATH" \
  +modalities.state_mod.projection.kwargs.embedding_dim=2058 \
  +modalities.state_mod.projection.kwargs.output_dim=4096 \
  +modalities.state_mod.trainable.encoder=false \
  +modalities.state_mod.trainable.projection=$TRAIN_PROJECTION \
  $@

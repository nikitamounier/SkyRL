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

python -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=2 \
  trainer.eval_batch_size=64 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=40 \
  trainer.max_prompt_length=8192 \
  generator.sampling_params.max_generate_length=512 \
  generator.sampling_params.temperature=1.0 \
  trainer.policy.optimizer_config.lr=2.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=cell_pathway \
  generator.n_samples_per_prompt=16 \
  generator.gpu_memory_utilization=0.4 \
  generator.enforce_eager=true \
  +generator.engine_init_kwargs.max_model_len=12288 \
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

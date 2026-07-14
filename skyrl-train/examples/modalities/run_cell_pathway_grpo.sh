#!/usr/bin/env bash
set -euo pipefail
set -x

# GRPO on the BioReasonCell gene->pathway task with the STATE cell-embedding modality.
# The 2058-d STATE mean vector is injected at a <|cell_pad|> placeholder via the SkyRL
# PromptEmbeddingBuilder + our CellProjection (loads cell_projection.pt), so both the
# vLLM rollout and the training forward are byte-identical to the BioReasonCell SFT path.
#
# Prereqs:
#   1) prepare data:
#        python examples/modalities/prepare_cell_pathway_data.py \
#          --arrow_dir <cached HF arrow dir> \
#          --cell_embed_dir /large_storage/goodarzilab/bioreason_cell/embeddings/genetic_v7_5/cells \
#          --output_dir ~/data/cell_pathway_v7_5 --n_train 256 --n_val 128
#   2) MODEL_PATH must be a local checkpoint dir (for the PromptEmbeddingBuilder base embedding).
#
# NOTE: the converted 9B checkpoint is Qwen3_5ForConditionalGeneration (vision_config +
# M-RoPE). vLLM prompt_embeds needs M-RoPE disabled and the vision tower skipped; that
# requires the engine to forward hf_overrides + language_model_only (see engine patch).

DATA_DIR="${DATA_DIR:-$HOME/data/cell_pathway_v7_5}"
CKPT_DIR="${CKPT_DIR:-$HOME/data/ckpts/cell_pathway_grpo}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGER="${LOGGER:-console}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"

MODEL_PATH="${MODEL_PATH:-/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_full_9b_converted}"
PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-<|cell_pad|>}"
CELL_PROJ_PATH="${CELL_PROJ_PATH:-$MODEL_PATH/cell_projection.pt}"
# Freeze the projection for the forward-inference check; set true to co-train it in RL.
TRAIN_PROJECTION="${TRAIN_PROJECTION:-false}"
EVAL_ONLY="${EVAL_ONLY:-false}"   # true -> just run eval_before_train (milestone-1 forward check)

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
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=50 \
  trainer.max_prompt_length=16384 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=0.0 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=cell_pathway \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.85 \
  generator.enforce_eager=true \
  +generator.engine_init_kwargs.language_model_only=true \
  +generator.engine_init_kwargs.disable_mrope=true \
  trainer.logger="$LOGGER" \
  trainer.project_name="cell_pathway_grpo" \
  trainer.run_name="cell_pathway_grpo" \
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

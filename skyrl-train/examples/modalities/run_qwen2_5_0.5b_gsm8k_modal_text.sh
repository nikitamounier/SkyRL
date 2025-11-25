#!/usr/bin/env bash
set -euo pipefail
set -x

# End-to-end smoke test for the multimodal pipeline using a "text modality".
# Half of the dataset uses a placeholder token plus modality payload (token ids),
# embedded via the base model embedding and passed through unchanged.
# Intended to run on Modal (see examples/gsm8k/run_gsm8k_modal.sh for mount patterns).

# 1) Prepare the modal-text dataset (idempotent)
#   uv run --isolated --extra transformers --extra datasets examples/modalities/prepare_modal_text_split.py \
#     --model_path Qwen/Qwen2.5-0.5B-Instruct \
#     --output_dir "/root/data/gsm8k_modal_text" \
#     --placeholder_token "<|extra_0|>"
#
# 2) Run training with modalities enabled (example Modal command):
#   modal run main.py --command "WANDB_API_KEY=... DATA_DIR=/root/data/gsm8k_modal_text bash examples/modalities/run_qwen2_5_0.5b_gsm8k_modal_text.sh"

DATA_DIR="${DATA_DIR:-/root/data/gsm8k_modal_text}"
CKPT_DIR="${CKPT_DIR:-/root/data/ckpts/gsm8k_0.5b_modal_text_lora_ckpt}"
NUM_GPUS="${NUM_GPUS:-4}"
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout
INFERENCE_BACKEND="vllm"
MODEL_PATH="Qwen/Qwen3-0.6B"
PLACEHOLDER_TOKEN="<|extra_0|>"

uv run --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=512 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_0.5b_modal_text_lora" \
  trainer.run_name="gsm8k_0.5b_modal_text_lora" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  +modalities.text_mod.placeholder_token="'$PLACEHOLDER_TOKEN'" \
  +modalities.text_mod.max_placeholder_tokens=2048 \
  +modalities.text_mod.encoder.target="skyrl_train.examples.modalities.text_passthrough_handlers:TokenIdListEncoder" \
  +modalities.text_mod.encoder.kwargs.model_path="$MODEL_PATH" \
  +modalities.text_mod.projection.target="skyrl_train.examples.modalities.text_passthrough_handlers:EmbeddingLookupProjection" \
  +modalities.text_mod.projection.kwargs.model_path="$MODEL_PATH" \
  +modalities.text_mod.trainable.encoder=false \
  +modalities.text_mod.trainable.projection=false \
  $@

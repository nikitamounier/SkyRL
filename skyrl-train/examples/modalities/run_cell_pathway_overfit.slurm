#!/bin/bash
#SBATCH --job-name=cell_overfit
#SBATCH --output=/home/parsaidp/SkyRL/cell_overfit_%j.out
#SBATCH --error=/home/parsaidp/SkyRL/cell_overfit_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --account=cell_reason
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=800G
#
# SANITY / OVERFIT TEST: hammer a single 32-prompt batch for many steps.
# If the on-policy pipeline is correct (gradients -> weight-sync -> rollout),
# the batch reward MUST climb toward ~1.0. If it stays flat, gradients are
# not reaching the rollout (weight-sync / update no-op) -> real bug.
# Fast answeronly 9B, LR=1e-5 (the LR that looked "flat" on 5k), KL off.

set -euo pipefail

ENV=/home/adibvafa/miniconda/envs/bioreason_cell
export PATH="$ENV/bin:$PATH"
export CONDA_PREFIX="$ENV"
if [ -f "${ENV}/lib/libstdc++.so.6" ]; then export LD_PRELOAD="${ENV}/lib/libstdc++.so.6"; fi
export CUDA_HOME="$ENV"
export PYTHONPATH=/home/parsaidp/cellrl_pydeps:/home/parsaidp/SkyRL/skyrl-train:/home/parsaidp/SkyRL/skyrl-gym:/home/parsaidp/BioReasonCell
export HF_HOME=/home/parsaidp/.cache/hf_cellrl
mkdir -p "$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# expandable_segments:True forces cuMem allocations, whose cross-process CUDA-IPC sharing uses
# pidfd_getfd -> "Operation not permitted" under this host's ptrace_scope=2. Disable it so the
# merged-weight IPC transfer uses the legacy (driver-level) cudaIpc path, which ptrace does not gate.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

export WANDB_API_KEY="$(cat "$HOME/.wandb_api_key")"
export WANDB_ENTITY="${WANDB_ENTITY:-goodarzilab}"

# minimal answeronly checkpoint (fast, 512-token gen)
export MODEL_PATH=${MODEL_PATH:-/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_answeronly_9b_converted}
export DATA_DIR=${DATA_DIR:-$HOME/data/cell_pathway_overfit32}
export CKPT_DIR=${CKPT_DIR:-$HOME/data/ckpts/cell_grpo_overfit32}
export RUN_NAME=${RUN_NAME:-cell_grpo_overfit32}
# Derive from the ACTUAL allocation: `sbatch --gres=gpu:N` overrides the #SBATCH line above, and a
# hardcoded 4 here would then ask Ray for more GPUs than exist and hang at placement-group setup.
export NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}

export LR=${LR:-1.0e-4}   # overridable via sbatch --export for LR sweep
export TEMP=${TEMP:-1.2}
export USE_KL=${USE_KL:-true}   # mild anchor to SFT (overridable)
export KL_COEF=0.005
export MAX_GEN=${MAX_GEN:-512}
export TRAIN_PROJECTION=${TRAIN_PROJECTION:-true}    # unfreeze cell projector: LoRA-only overfit was stable but FLAT
                                # (~0.68, 15 steps) -> hypothesis: frozen projector caps per-prompt
                                # memorization. Co-train + sync the projector so the model can adapt
                                # how it reads cells.
export SKYRL_SYNC_MODALITY_UNDER_LORA=1   # sync trained projector to vLLM each step (on-policy);
                                          # modality IPC path works now that expandable_segments=False
# ANTI-ENTROPY-COLLAPSE config: previous runs climbed ~4 steps then the policy entropy crashed
# (0.17->0.02) and mode-collapsed. Tight PPO clip (eps 0.1) caps per-update movement, and one
# fully-on-policy update per step (MINI_BATCH=TRAIN_BATCH) removes within-step off-policy drift.
export TRAIN_BATCH=${TRAIN_BATCH:-32}
export MINI_BATCH=${MINI_BATCH:-32}
export EPOCHS=${EPOCHS:-80}          # 80 repeats of the SAME batch (override for full runs)
export MICRO_TRAIN=${MICRO_TRAIN:-2}
export MICRO_FWD=${MICRO_FWD:-4}

cd /home/parsaidp/SkyRL/skyrl-train
# disable checkpointing (avoid the ckpt-save OOM that killed earlier runs)
TIS_FLAG=""; [ "${USE_TIS:-false}" = "true" ] && TIS_FLAG="trainer.algorithm.use_tis=true trainer.algorithm.tis_imp_ratio_cap=${TIS_CAP:-2.0}"
# BUGFIX (instability): the negative-advantage branch of the PPO surrogate is UNCLAMPED by eps_clip,
# so a rollout-vs-train logprob gap made exp(logp-old) explode -> policy_loss ~1000, grad_norm ~50k
# (clipped to 1.0) -> tiny noisy updates -> flat reward. dual_clip bounds that branch at |adv|*c.
DUAL_CLIP_FLAG=""; [ "${DUAL_CLIP:-true}" = "true" ] && DUAL_CLIP_FLAG="trainer.algorithm.policy_loss_type=dual_clip trainer.algorithm.clip_ratio_c=${CLIP_C:-3.0}"
# Dr. GRPO (DR_GRPO=true): remove the two GRPO biases — length-normalization in the loss
# (constant-normalizer reduction -> stops length inflation) and /std in the advantage. Targets
# the reasoning model's length-inflation instability so it tolerates a higher/steadier LR.
DR_GRPO_FLAG=""; [ "${DR_GRPO:-false}" = "true" ] && DR_GRPO_FLAG="trainer.algorithm.loss_reduction=seq_mean_token_sum_norm trainer.algorithm.grpo_norm_by_std=false"
# GSPO (GSPO=true): SEQUENCE-level importance sampling (arxiv 2507.18071) — one ratio per response
# instead of per-token, far more stable on long reasoning CoT and it averages out the per-token
# H1/H2 tail. Its own policy loss; REQUIRES sequence_mean reduction; use a SMALL eps_clip (~3e-4,
# pass EPS_CLIP=3e-4). Mutually exclusive with TIS/dual_clip/dr_grpo, so disable those.
GSPO_FLAG=""
if [ "${GSPO:-false}" = "true" ]; then
  # TIS / dual_clip are incompatible (they set a different policy_loss_type).
  TIS_FLAG=""; DUAL_CLIP_FLAG=""
  if [ "${DR_GRPO:-false}" = "true" ]; then
    # "Dr. GSPO" = GSPO's sequence-level IS + Dr. GRPO's two de-biasings. NOTE: GSPO's own
    # sequence_mean reduction STILL divides by response length (same length bias Dr. GRPO
    # removes), so swapping in the constant-normalizer reduction is the point. GSPO only
    # warns about a non-sequence_mean reduction; it is allowed.
    GSPO_FLAG="trainer.algorithm.policy_loss_type=gspo"
    DR_GRPO_FLAG="trainer.algorithm.loss_reduction=seq_mean_token_sum_norm trainer.algorithm.grpo_norm_by_std=false"
  else
    GSPO_FLAG="trainer.algorithm.policy_loss_type=gspo trainer.algorithm.loss_reduction=sequence_mean"
    DR_GRPO_FLAG=""
  fi
fi
bash examples/modalities/run_cell_pathway_grpo_5k.sh trainer.ckpt_interval=${CKPT_INTERVAL:-100000} $TIS_FLAG $DUAL_CLIP_FLAG $DR_GRPO_FLAG $GSPO_FLAG \
  trainer.algorithm.eps_clip_low=${EPS_CLIP:-0.1} trainer.algorithm.eps_clip_high=${EPS_CLIP:-0.1} \
  trainer.resume_mode=${RESUME_MODE:-null} trainer.max_ckpts_to_keep=${MAX_CKPTS:--1}

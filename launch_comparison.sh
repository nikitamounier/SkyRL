#!/bin/bash
# ============================================================================
# Launch GRPO vs REINFORCE++ with Qwen3-Embedding-4B encoder + reward shaping.
# Sweep across memory windows w3, w5, w10. LR=1e-4.
#
# Runs (6 total):
#   tw_grpo_w3    tw_reinforcepp_w3
#   tw_grpo_w5    tw_reinforcepp_w5
#   tw_grpo_w10   tw_reinforcepp_w10
# ============================================================================
set -euo pipefail

PARTITION="${PARTITION:-preemptible}"
TIME="${TIME:-24:00:00}"
LR="1e-4"
CKPT_BASE="${CKPT_BASE:-/large_storage/goodarzilab/parsaidp/MeMo/ckpts}"

echo "========================================="
echo "Comparison: GRPO vs REINFORCE++"
echo "  Partition: $PARTITION"
echo "  Windows:   3, 5, 10"
echo "  LR:        $LR"
echo "  Encoder:   MemoSentenceEmbeddingEncoder (Qwen3-Embedding-4B)"
echo "  Reward:    step_penalty=0.01, eff_bonus=1.0"
echo "  MemCkpt:  ${MEMORY_CHECKPOINT:-none}"
echo "========================================="
echo ""

JOBS=()

for ADV in grpo "reinforce++"; do
  for MW in 3 5 10; do
    ADV_SHORT="${ADV//+/p}"
    NAME="tw_${ADV_SHORT}_w${MW}"

    JOB=$(ADVANTAGE_ESTIMATOR="$ADV" \
      LR="$LR" \
      STEP_PENALTY=0.01 \
      EFFICIENCY_BONUS=1.0 \
      MEMORY_WINDOW=$MW \
      ${MEMORY_CHECKPOINT:+MEMORY_CHECKPOINT="$MEMORY_CHECKPOINT"} \
      RUN_NAME="$NAME" \
      CKPT_DIR="${CKPT_BASE}/${NAME}" \
      sbatch -p "$PARTITION" -t "$TIME" \
        -J "$NAME" \
        -o "${NAME}_%j.out" \
        -e "${NAME}_%j.err" \
        train_textworld_memo.slurm | awk '{print $4}')

    JOBS+=("$JOB $NAME")
    echo "  Submitted: $NAME (job $JOB)"
  done
done

echo ""
echo "========================================="
echo "All ${#JOBS[@]} jobs submitted:"
for j in "${JOBS[@]}"; do
  echo "  $j"
done
echo ""
echo "Monitor on wandb: project=textworld_memo"
echo "========================================="

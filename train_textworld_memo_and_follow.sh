#!/usr/bin/env bash
set -euo pipefail

# Submit the TextWorld MeMo training job and auto-tail logs.
# Usage:
#   ./train_textworld_memo_and_follow.sh
#   ./train_textworld_memo_and_follow.sh --job-name train_textworld_memo

JOB_NAME="train_textworld_memo"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

SBATCH_OUT="$(sbatch --job-name "$JOB_NAME" train_textworld_memo.slurm)"
echo "$SBATCH_OUT"

JOB_ID="$(echo "$SBATCH_OUT" | awk '{print $NF}')"
if [[ -z "${JOB_ID:-}" ]]; then
  echo "Failed to parse job id from sbatch output."
  exit 1
fi

bash ./follow_slurm_logs.sh "$JOB_NAME" "$JOB_ID"

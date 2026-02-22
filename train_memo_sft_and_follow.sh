#!/usr/bin/env bash
set -euo pipefail

# Submit MeMo SFT training and stream Slurm logs until job completion.
# Usage:
#   ./train_memo_sft_and_follow.sh
#   ./train_memo_sft_and_follow.sh --wait-log-start 900
#   ./train_memo_sft_and_follow.sh --sbatch-arg "-p preemptible"

SLURM_SCRIPT="train_memo_sft.slurm"
JOB_NAME="memo_sft"
WAIT_LOG_START="600"   # seconds; 0 means wait forever
POLL_SECONDS="2"
QUEUE_POLL_SECONDS="5"
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --slurm-script <path>      Slurm script to submit (default: train_memo_sft.slurm)
  --job-name <name>          Override job name for submission (default: memo_sft)
  --wait-log-start <sec>     Max wait for .out/.err to appear (default: 600, 0=forever)
  --sbatch-arg <arg>         Additional sbatch arg (repeatable), e.g. --sbatch-arg "-p preemptible"
  -h, --help                 Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slurm-script)
      SLURM_SCRIPT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --wait-log-start)
      WAIT_LOG_START="$2"
      shift 2
      ;;
    --sbatch-arg)
      EXTRA_SBATCH_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

cd "$(dirname "$0")"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Slurm script not found: $SLURM_SCRIPT" >&2
  exit 1
fi

submit_cmd=(sbatch --parsable --job-name "$JOB_NAME")
if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
  # shellcheck disable=SC2206
  extra_tokens=(${EXTRA_SBATCH_ARGS[*]})
  submit_cmd+=("${extra_tokens[@]}")
fi
submit_cmd+=("$SLURM_SCRIPT")

echo "Submitting: ${submit_cmd[*]}"
SBATCH_OUT="$(${submit_cmd[@]})"
JOB_ID="${SBATCH_OUT%%;*}"

if [[ -z "$JOB_ID" || ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "Failed to parse numeric job id from sbatch output: $SBATCH_OUT" >&2
  exit 1
fi

OUT_FILE="${JOB_NAME}_${JOB_ID}.out"
ERR_FILE="${JOB_NAME}_${JOB_ID}.err"

echo "Submitted job: $JOB_ID"
echo "Expected logs:"
echo "  $OUT_FILE"
echo "  $ERR_FILE"

waited=0
last_status=""
while [[ ! -f "$OUT_FILE" || ! -f "$ERR_FILE" ]]; do
  status=""
  if command -v squeue >/dev/null 2>&1; then
    status="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null || true)"
    if [[ -n "$status" && "$status" != "$last_status" ]]; then
      echo "Job $JOB_ID status: $status"
      last_status="$status"
    fi
  fi

  if [[ "$WAIT_LOG_START" -gt 0 && "$waited" -ge "$WAIT_LOG_START" ]]; then
    echo "Timed out waiting for log files after ${WAIT_LOG_START}s" >&2
    break
  fi

  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

if [[ -f "$OUT_FILE" || -f "$ERR_FILE" ]]; then
  echo "Starting live log stream..."
  tail -n 0 -F "$OUT_FILE" "$ERR_FILE" &
  TAIL_PID=$!
else
  echo "No log files found to stream." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${TAIL_PID:-}" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
    kill "$TAIL_PID" 2>/dev/null || true
    wait "$TAIL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for job completion.
while true; do
  if command -v squeue >/dev/null 2>&1; then
    status="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null || true)"
    if [[ -z "$status" ]]; then
      break
    fi
  else
    # Fallback: if squeue is unavailable, keep streaming until interrupted.
    sleep 3600
    continue
  fi
  sleep "$QUEUE_POLL_SECONDS"
done

# Give Slurm a moment to flush final log lines.
sleep 2

echo "Job $JOB_ID is no longer in queue."
if command -v sacct >/dev/null 2>&1; then
  acct_line="$(sacct -j "$JOB_ID" --format=JobIDRaw,State,ExitCode -P -n 2>/dev/null | awk -F'|' -v id="$JOB_ID" '$1==id {print; exit}')"
  if [[ -n "$acct_line" ]]; then
    state="$(echo "$acct_line" | awk -F'|' '{print $2}')"
    exit_code="$(echo "$acct_line" | awk -F'|' '{print $3}')"
    echo "Final state: $state (ExitCode: $exit_code)"
  fi
fi

echo "Log streaming complete."

#!/usr/bin/env bash
set -euo pipefail

# Follow the latest Slurm .out/.err pair for a given prefix.
# Usage:
#   ./follow_slurm_logs.sh train_textworld_memo
#   ./follow_slurm_logs.sh train_modal_text
#   ./follow_slurm_logs.sh train_textworld_memo 123456

PREFIX="${1:-}"
JOB_ID="${2:-}"
MAX_WAIT="${MAX_WAIT:-1800}" # seconds; set to 0 to wait indefinitely

if [[ -z "$PREFIX" ]]; then
  echo "Usage: $0 <prefix> [job_id]"
  exit 1
fi

if [[ -n "$JOB_ID" ]]; then
  OUT_FILE="${PREFIX}_${JOB_ID}.out"
  ERR_FILE="${PREFIX}_${JOB_ID}.err"
else
  OUT_FILE="$(ls -1t ${PREFIX}_*.out 2>/dev/null | head -n 1 || true)"
  ERR_FILE="$(ls -1t ${PREFIX}_*.err 2>/dev/null | head -n 1 || true)"
fi

wait_for_file() {
  local file="$1"
  local max_wait="${2:-$MAX_WAIT}"
  local waited=0
  local last_status=""
  while [[ ! -f "$file" ]]; do
    if [[ "$max_wait" -gt 0 && $waited -ge $max_wait ]]; then
      return 1
    fi
    if [[ -n "$JOB_ID" ]] && command -v squeue >/dev/null 2>&1; then
      local status
      status="$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || true)"
      if [[ -n "$status" && "$status" != "$last_status" ]]; then
        echo "Waiting for $file (job $JOB_ID status: $status)..."
        last_status="$status"
      fi
      if [[ -z "$status" && "$max_wait" -gt 0 && $waited -ge 10 ]]; then
        # Job no longer in queue and no log file yet.
        return 1
      fi
    fi
    sleep 1
    waited=$((waited + 1))
  done
  [[ -f "$file" ]]
}

if [[ -n "$JOB_ID" ]]; then
  if ! wait_for_file "$OUT_FILE" 300; then
    echo "Timed out waiting for $OUT_FILE"
    exit 1
  fi
  if ! wait_for_file "$ERR_FILE" 300; then
    echo "Timed out waiting for $ERR_FILE"
    exit 1
  fi
else
  if [[ -z "${OUT_FILE:-}" || -z "${ERR_FILE:-}" ]]; then
    echo "No log files found for prefix '${PREFIX}'."
    exit 1
  fi
fi

echo "Tailing:"
echo "  ${OUT_FILE}"
echo "  ${ERR_FILE}"
echo "Press Ctrl+C to stop."

tail -n 200 -F "$OUT_FILE" "$ERR_FILE"

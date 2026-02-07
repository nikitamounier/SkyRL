#!/bin/bash
# Capture training logs to a file

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_debug_${TIMESTAMP}.log"

echo "Starting training and capturing logs to: $LOG_FILE"

# Submit job and capture output
sbatch train_textworld_memo.slurm 2>&1 | tee "$LOG_FILE"

# Get job ID
JOB_ID=$(tail -n 1 "$LOG_FILE" | awk '{print $NF}')

echo "Job ID: $JOB_ID" | tee -a "$LOG_FILE"

# Follow the slurm output files
if [ -n "$JOB_ID" ]; then
    echo "Waiting for job to start..." | tee -a "$LOG_FILE"
    sleep 5
    
    # Tail both stdout and stderr
    OUT_FILE="train_textworld_memo_${JOB_ID}.out"
    ERR_FILE="train_textworld_memo_${JOB_ID}.err"
    
    echo "=== Tailing $OUT_FILE ===" | tee -a "$LOG_FILE"
    tail -f "$OUT_FILE" 2>&1 | tee -a "$LOG_FILE" &
    TAIL_PID=$!
    
    # Wait for error or completion
    sleep 60
    kill $TAIL_PID 2>/dev/null || true
    
    # Capture the full output
    if [ -f "$OUT_FILE" ]; then
        echo "=== Full stdout ===" >> "$LOG_FILE"
        cat "$OUT_FILE" >> "$LOG_FILE"
    fi
    if [ -f "$ERR_FILE" ]; then
        echo "=== Full stderr ===" >> "$LOG_FILE"
        cat "$ERR_FILE" >> "$LOG_FILE"
    fi
fi

echo "Logs saved to: $LOG_FILE"

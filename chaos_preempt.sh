#!/bin/bash

# === CONFIG ===
LOGFILE="chaos_log_$(date +%Y%m%d_%H%M%S).txt"
COUNT=0
START_TIME=$(date +%s)
TRAIN_SCRIPT="/home/parsaidp/SkyRL/priority_dummy.slurm"
TARGET_JOB="${TARGET_JOB:-1678902}"  # Your actual training job ID

# === LOGGING ===
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOGFILE"
}

# === CLEANUP ON EXIT ===
cleanup() {
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    log "========================================="
    log "CHAOS SESSION COMPLETE"
    log "========================================="
    log "Total preemptions: $COUNT"
    log "Runtime: $((DURATION / 60))m $((DURATION % 60))s"
    log "Log saved to: $LOGFILE"
    log "========================================="

    if [[ -n "$JOB_ID" ]]; then
        scancel "$JOB_ID" 2>/dev/null
        log "Cleaned up final job: $JOB_ID"
    fi

    exit 0
}

trap cleanup SIGINT SIGTERM

# === CHECK TARGET JOB ===
check_target_job() {
    local state=$(squeue -j "$TARGET_JOB" -h -o "%T" 2>/dev/null)
    if [[ "$state" == "RUNNING" ]]; then
        log "🎉 TARGET JOB $TARGET_JOB IS RUNNING! Mission accomplished!"
        cleanup
    fi
}

# === MAIN LOOP ===
log "========================================="
log "CHAOS LOOP INITIATED"
log "Target job to watch: $TARGET_JOB"
log "Logging to: $LOGFILE"
log "Will auto-stop when target job starts running!"
log "========================================="

while true; do
    # Check if target job is running
    check_target_job
    # Submit job
    OUTPUT=$(sbatch "$TRAIN_SCRIPT" 2>&1)
    JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K\d+')

    if [[ -z "$JOB_ID" ]]; then
        log "ERROR: Failed to submit job. Output: $OUTPUT"
        sleep 5
        continue
    fi

    log "Submitted priority job $JOB_ID, waiting for RUNNING state..."

    # Poll until running
    WAIT_START=$(date +%s)
    while true; do
        STATE=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null)

        if [[ -z "$STATE" ]]; then
            log "Job $JOB_ID vanished from queue (possibly failed)"
            break
        elif [[ "$STATE" == "RUNNING" ]]; then
            WAIT_TIME=$(($(date +%s) - WAIT_START))
            ((COUNT++))
            log ">>> STRIKE #$COUNT - Job $JOB_ID running (waited ${WAIT_TIME}s) - CANCELLING"
            scancel "$JOB_ID"

            # Check target job again right after cancelling
            check_target_job
            break
        fi

        sleep 1
    done

    log "Cooling down 20 seconds..."
    sleep 20
    log "-----------------------------------------"
done

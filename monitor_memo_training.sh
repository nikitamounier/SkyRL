#!/bin/bash
# Monitor MeMo training for the fix verification

JOB_ID=${1:-1679845}
LOG_FILE="train_textworld_memo_${JOB_ID}.err"

echo "========================================="
echo "Monitoring TextWorld MeMo Training"
echo "Job ID: $JOB_ID"
echo "Log file: $LOG_FILE"
echo "========================================="
echo ""

# Check job status
echo "Job Status:"
squeue -u $USER -j $JOB_ID 2>/dev/null || echo "Job not found in queue (may have completed)"
echo ""

# Wait for log file
if [ ! -f "$LOG_FILE" ]; then
    echo "Waiting for log file to appear..."
    sleep 5
fi

echo "========================================="
echo "Key Metrics to Verify Fix"
echo "========================================="
echo ""

echo "1. Generator passing modalities_metadata:"
grep -c "\[GENERATOR\] Passing.*modalities_metadata" "$LOG_FILE" 2>/dev/null || echo "  Not found yet (still initializing)"
echo ""

echo "2. Samples with non-empty payloads:"
grep "\[GENERATOR\] Sample.*non-empty payload" "$LOG_FILE" 2>/dev/null | tail -5
echo ""

echo "3. Experience metadata keys (should include 'modalities_metadata'):"
grep "experience.metadata keys:" "$LOG_FILE" 2>/dev/null | head -1
echo ""

echo "4. Samples with memory documents:"
grep -c "HAS MEMORY DOCS" "$LOG_FILE" 2>/dev/null || echo "  Not found yet (waiting for turns 3+)"
echo ""

echo "5. Backward pass execution:"
grep "BACKWARD.*has_any_memory_docs=True" "$LOG_FILE" 2>/dev/null | head -3
echo ""

echo "========================================="
echo "Recent Log Entries (last 20 lines):"
echo "========================================="
tail -20 "$LOG_FILE"
echo ""

echo "To monitor live: tail -f $LOG_FILE"
echo "To check again: bash $0 $JOB_ID"

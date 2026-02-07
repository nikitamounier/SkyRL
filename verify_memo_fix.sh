#!/bin/bash
# Verification script for MeMo memory document fix

echo "========================================="
echo "Verifying MeMo Memory Document Fix"
echo "========================================="
echo ""

# Run the test training
echo "Running test training (2 epochs)..."
bash skyrl-train/examples/MeMo/run_textworld_memo_train.sh \
  --data_dir /home/parsaidp/data/textworld_memo \
  --ckpt_dir /home/parsaidp/data/ckpts/textworld_memo_test_fix \
  --epochs 2 \
  > train_memo_fix_test.log 2>&1 &

TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Log file: train_memo_fix_test.log"
echo ""

# Wait a bit for training to start
echo "Waiting for training to generate some data..."
sleep 30

# Check the logs
echo ""
echo "========================================="
echo "Checking Logs for Expected Behavior"
echo "========================================="
echo ""

# Check 1: Verify modalities_metadata is being passed
echo "1. Checking if modalities_metadata is passed to trainer..."
grep -c "\[GENERATOR\] Passing.*modalities_metadata" train_memo_fix_test.log || echo "Not found yet"

# Check 2: Verify metadata contains modalities_metadata key
echo "2. Checking if experience.metadata contains modalities_metadata..."
grep "experience.metadata keys:" train_memo_fix_test.log | head -1

# Check 3: Verify samples with memory docs
echo "3. Checking for samples with memory documents..."
grep -c "HAS MEMORY DOCS" train_memo_fix_test.log || echo "Not found yet"

# Check 4: Verify backward pass executes
echo "4. Checking if backward pass executes with memory..."
grep -c "BACKWARD.*has_any_memory_docs=True" train_memo_fix_test.log || echo "Not found yet"

echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo ""
echo "The fix should result in:"
echo "  ✓ [GENERATOR] logs showing metadata passed to trainer"
echo "  ✓ experience.metadata keys including 'modalities_metadata'"
echo "  ✓ [MEMORY CHECK] logs showing 'HAS MEMORY DOCS' for turns 3+"
echo "  ✓ [BACKWARD] logs showing has_any_memory_docs=True"
echo ""
echo "Monitor the log file: tail -f train_memo_fix_test.log"
echo "Kill training when done: kill $TRAIN_PID"

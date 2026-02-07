#!/bin/bash
# Comprehensive cleanup and restart

set -x  # Show commands being run

echo "🧹 COMPREHENSIVE CLEANUP"
echo "========================"

# 1. Stop Ray
echo "Stopping Ray..."
ray stop --force 2>/dev/null || true
sleep 2

# 2. Kill training processes
echo "Killing training processes..."
pkill -f "run_textworld_memo_train.sh" || true
pkill -f "skyrl_train.entrypoints.main_base" || true
if [ -f /home/parsaidp/SkyRL/training.pid ]; then
    PID=$(cat /home/parsaidp/SkyRL/training.pid)
    kill $PID 2>/dev/null || true
    rm /home/parsaidp/SkyRL/training.pid
fi
sleep 2

# 3. Clean ALL caches
echo "Cleaning Ray cache..."
rm -rf /tmp/claude-parsaidp/ray/session_* 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf ~/.cache/ray/* 2>/dev/null || true

echo "Cleaning Python bytecode cache..."
cd /home/parsaidp/SkyRL/skyrl-train
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Cleaning UV/pip cache for this package..."
cd /home/parsaidp/SkyRL/skyrl-train
uv cache clean 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo ""

# 4. Verify source file has correct code
echo "Verifying model_wrapper.py has DTensor fix..."
if grep -q "FSDP2: Convert DTensor to local before loop" /home/parsaidp/SkyRL/skyrl-train/skyrl_train/model_wrapper.py; then
    echo "✅ DTensor fix FOUND in source"
else
    echo "❌ DTensor fix NOT found in source - this should not happen!"
fi

if grep -q "modalities_entry = self.extras.get" /home/parsaidp/SkyRL/skyrl-gym/skyrl_gym/envs/textworld/env.py; then
    echo "✅ Modalities payload fix FOUND in env.py"
else
    echo "❌ Modalities payload fix NOT found - this should not happen!"
fi

echo ""
echo "🚀 Starting fresh training..."
echo ""

# 5. Start training
cd /home/parsaidp/SkyRL
./run_training_background.sh

#!/bin/bash
# Comprehensive cache cleanup script
# Fixes Ray code synchronization issues

set -x

echo "🧹 COMPREHENSIVE CACHE CLEANUP"
echo "================================"

# 1. Stop everything
echo "Stopping Ray and training processes..."
ray stop --force 2>/dev/null || true
pkill -f "skyrl_train" || true
pkill -f "run_textworld_memo_train.sh" || true

# Kill training process if PID file exists
if [ -f /home/parsaidp/SkyRL/training.pid ]; then
    PID=$(cat /home/parsaidp/SkyRL/training.pid)
    kill $PID 2>/dev/null || true
    rm /home/parsaidp/SkyRL/training.pid
fi

sleep 2

# 2. Clean ALL Python bytecode
echo "Cleaning Python bytecode cache..."
find /home/parsaidp/SkyRL -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /home/parsaidp/SkyRL -type f -name "*.pyc" -delete 2>/dev/null || true

# 3. Clean Ray sessions
echo "Cleaning Ray session directories..."
rm -rf /tmp/claude-parsaidp/ray/session_* 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true
rm -rf ~/.cache/ray/* 2>/dev/null || true

# 4. Clean UV cache for this package
echo "Cleaning UV/pip cache..."
cd /home/parsaidp/SkyRL/skyrl-train
uv cache clean 2>/dev/null || true

echo ""
echo "✅ All caches cleaned!"
echo ""
echo "IMPORTANT: Set this environment variable before running training:"
echo "export PYTHONDONTWRITEBYTECODE=1"
echo ""

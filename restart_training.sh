#!/bin/bash
# Clean Ray cache and restart training

echo "🧹 Cleaning up Ray..."

# Stop any running Ray instances
ray stop --force 2>/dev/null || true

# Kill any running training processes
if [ -f /home/parsaidp/SkyRL/training.pid ]; then
    PID=$(cat /home/parsaidp/SkyRL/training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Killing training process $PID..."
        kill $PID 2>/dev/null || true
        sleep 2
    fi
    rm /home/parsaidp/SkyRL/training.pid
fi

# Clean Ray temp files (forces code refresh)
echo "Cleaning Ray temp directories..."
rm -rf /tmp/claude-parsaidp/ray/session_* 2>/dev/null || true
rm -rf /tmp/ray/session_* 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Starting fresh training run..."
echo ""

# Start new training
cd /home/parsaidp/SkyRL
./run_training_background.sh

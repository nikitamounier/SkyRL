#!/bin/bash
# Quick monitor script to watch both chaos jobs and your training job

TARGET_JOB="${1:-1678902}"

watch -n 0.5 "echo '=== YOUR TRAINING JOB ===' && \
squeue -j $TARGET_JOB -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R' 2>/dev/null || echo 'Job not in queue' && \
echo '' && \
echo '=== CHAOS PRIORITY JOBS ===' && \
squeue -u parsaidp --partition=cell_reason_gpu_priority -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R' 2>/dev/null || echo 'No priority jobs' && \
echo '' && \
echo '=== ALL YOUR JOBS ===' && \
squeue -u parsaidp -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"

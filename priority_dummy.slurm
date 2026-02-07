#!/bin/bash
#SBATCH --job-name=priority_preemptor
#SBATCH --output=priority_dummy_%j.out
#SBATCH --error=priority_dummy_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=cell_reason_gpu_priority
#SBATCH --account=cell_reason
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

set -euo pipefail

echo "Priority job started at $(date)"
echo "Hostname: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Just sleep briefly - this job exists only to preempt others
sleep 10

echo "Priority job finished at $(date)"

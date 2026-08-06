#!/bin/bash
#SBATCH --job-name=cm-reason40
#SBATCH --output=/home/parsaidp/SkyRL/cm_reason40_%j.out
#SBATCH --partition=cpu
#SBATCH --account=cell_reason
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=220G --time=0:40:00
set -euo pipefail
export PATH="/home/adibvafa/miniconda/envs/bioreason_cell/bin:$PATH"
export CKPT_POLICY_DIR=/large_storage/goodarzilab/bioreason_cell/rl_ckpts/cell_grpo_nadig_reasoning/global_step_40/policy
export FULL_SD_OUT=/home/parsaidp/cellrl_eval/full_sd_reason40.pt
export BASE_CKPT=/large_storage/goodarzilab/bioreason_cell/checkpoints/sft_gene_pathway_v7_5_50k_9b_converted
export MERGED_OUT=/large_storage/goodarzilab/bioreason_cell/checkpoints/nadig_reason_step40_merged
cd /home/parsaidp/cellrl_eval
OMP_NUM_THREADS=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29531 torchrun --nproc_per_node=3 --nnodes=1 consolidate_generic.py
python merge_generic.py
rm -f $FULL_SD_OUT
echo "CM_DONE"

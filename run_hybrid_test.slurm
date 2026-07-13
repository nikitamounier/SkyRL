#!/bin/bash
#SBATCH --job-name=hybrid_test
#SBATCH --output=hybrid_test_%j.out
#SBATCH --error=hybrid_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#
# Hybrid data gen: Qwen3 plays via vLLM + GPT oracle at memory steps.
#
# Usage:
#   OPENAI_API_KEY=sk-... sbatch run_hybrid_test.slurm
#   OPENAI_API_KEY=sk-... MAX_GAMES=10 sbatch run_hybrid_test.slurm

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd "${SLURM_SUBMIT_DIR:-$HOME/SkyRL}"

export SKYRL_REPO_ROOT="${SKYRL_REPO_ROOT:-$HOME/SkyRL}"
source "${SKYRL_REPO_ROOT}/skyrl-train/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}:$HOME/SkyRL/skyrl-gym"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Export it before submitting: OPENAI_API_KEY=sk-... sbatch run_hybrid_test.slurm"
    exit 1
fi

DATA_FILE="${DATA_FILE:-$HOME/data/textworld_memo/test.parquet}"
MAX_GAMES="${MAX_GAMES:-5}"
MAX_TURNS="${MAX_TURNS:-50}"
MEMORY_WINDOW="${MEMORY_WINDOW:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
GPT_MODEL="${GPT_MODEL:-gpt-5-mini}"
GPT_WORKERS="${GPT_WORKERS:-4}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/data/hybrid_test_output}"

echo "========================================="
echo "Hybrid Data Gen (Qwen3 + GPT Oracle)"
echo "  Data:          $DATA_FILE"
echo "  Qwen3 model:   $MODEL_PATH"
echo "  GPT oracle:    $GPT_MODEL"
echo "  Memory window: $MEMORY_WINDOW"
echo "  Max games:     $MAX_GAMES"
echo "  Max turns:     $MAX_TURNS"
echo "  Batch size:    $BATCH_SIZE"
echo "  GPT workers:   $GPT_WORKERS"
echo "  Output:        $OUTPUT_DIR"
echo "========================================="

# Stage 1: Qwen3 plays
python skyrl-train/examples/MeMo/play_textworld_hybrid.py stage1 \
    --data_file "$DATA_FILE" \
    --model_path "$MODEL_PATH" \
    --memory_window "$MEMORY_WINDOW" \
    --max_turns "$MAX_TURNS" \
    --max_games "$MAX_GAMES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Stage 1 done. Running Stage 2 (GPT oracle)..."
echo ""

# Stage 2: GPT oracle annotates
python skyrl-train/examples/MeMo/play_textworld_hybrid.py stage2 \
    --transcript_dir "$OUTPUT_DIR" \
    --gpt_model "$GPT_MODEL" \
    --gpt_workers "$GPT_WORKERS"

echo ""
echo "========================================="
echo "All done. Output: $OUTPUT_DIR"
echo "========================================="

if [[ -f "$OUTPUT_DIR/summary.json" ]]; then
    echo "Summary:"
    cat "$OUTPUT_DIR/summary.json"
fi

echo ""
if [[ -f "$OUTPUT_DIR/oracle_samples.jsonl" ]]; then
    NSAMPLES=$(wc -l < "$OUTPUT_DIR/oracle_samples.jsonl")
    echo "Oracle samples: $NSAMPLES"
    echo "First sample:"
    head -1 "$OUTPUT_DIR/oracle_samples.jsonl" | python -m json.tool 2>/dev/null || head -1 "$OUTPUT_DIR/oracle_samples.jsonl"
fi

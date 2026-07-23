#!/bin/bash
#SBATCH -J explain_p3_s42_with_evidence_verify
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

RUN_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/headline/p3_eahgt_both_s42
OUT_DIR=$REPO_ROOT/headline_results/evaluate_advancement/explanations_with_evidence_verify
PAIRS=$REPO_ROOT/headline_results/evaluate_advancement/explanations_evidence_free_verify/pairs.csv
RAW_EDGES_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/evidences/edges

mkdir -p "$OUT_DIR"

python explain/cli/explain_advancement.py \
    --config         "$RUN_DIR/config.yaml" \
    --checkpoint     "$RUN_DIR/best_model.pt" \
    --out-dir        "$OUT_DIR" \
    --pairs-csv      "$PAIRS" \
    --raw-edges-dir  "$RAW_EDGES_DIR" \
    --case-studies   1 \
    --case-top-k     20 \
    --n-steps        32

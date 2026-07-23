#!/bin/bash
#SBATCH -J pagelink_ens
#SBATCH -o %x.%A_%a.o
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:0:0
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# PaGE-Link path explanations for EACH of the 5 grouped-ensemble seeds on the
# 26.03 LATEST build. NO relation exclusion: temporal masking already removes
# each pair's own clinical_trial_positive (label) edge at the decision point
# (verified 0/8 own-positive edges survive), so the remaining trial edges are
# legitimate cross-indication clinical precedent, not label leakage. min_mask
# lowered to 0.02 (0.1 pruned real paths). Aggregated across seeds so the
# explanation matches the deployed ensemble.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"
export WANDB_MODE="disabled"

SEEDS=(1 7 42 123 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/lr_grouped_k100_latest/lrgrpk100lat_s${SEED}
PAIRS="${1:-explain_pairs_evfree_paths8.csv}"
OUT_DIR=$REPO_ROOT/headline_results/evaluate_advancement/pagelink_biobridge/s${SEED}

python explain/cli/pagelink_explain.py \
    --config             "$RUN_DIR/config.yaml" \
    --checkpoint         "$RUN_DIR/best_model.pt" \
    --pairs-csv          "$PAIRS" \
    --exclude-relations  "" \
    --min-mask           0.02 \
    --num-paths          5 \
    --mask-epochs        200 \
    --out-dir            "$OUT_DIR" \
    --seed               "$SEED"

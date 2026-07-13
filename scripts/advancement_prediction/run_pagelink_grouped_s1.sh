#!/bin/bash
#SBATCH -J pagelink_grp_s1
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:0:0
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# PaGE-Link path explanations (#4) on the grouped seed-1 26.03 checkpoint.
# Learns a soft edge mask per pair, then enforces target->disease paths.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

# This branch may run from a git worktree, which has no .venv of its own.
# Use the main checkout's venv (same deps + networkx); fall back to a local one.
VENV="$REPO_ROOT/.venv/bin/activate"
[ -f "$VENV" ] || VENV=/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/.venv/bin/activate
source "$VENV"
export WANDB_MODE="disabled"

RUN_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/lr_grouped_k100/lrgrpk100_s1
PAIRS="${1:-explain_pairs_evfree_diverse.csv}"
OUT_DIR=$REPO_ROOT/headline_results/evaluate_advancement/pagelink_grouped_s1

python pagelink_explain.py \
    --config       "$RUN_DIR/config.yaml" \
    --checkpoint   "$RUN_DIR/best_model.pt" \
    --pairs-csv    "$PAIRS" \
    --mask-epochs  200 \
    --lr           0.01 \
    --size-coeff   5e-2 \
    --entropy-coeff 2e-1 \
    --num-paths    5 \
    --min-mask     0.05 \
    --verbose \
    --out-dir      "$OUT_DIR"

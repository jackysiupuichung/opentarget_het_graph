#!/bin/bash
#SBATCH -J fidelity_grp_s1
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

# Faithfulness of the advancement explainer's edge attributions (#1), on the
# grouped seed-1 26.03 checkpoint (the representative model the case studies
# explain). Reports fid+/fid-/characterization/unfaithfulness for
# ig / abs_ig / attention / random, swept over sparsity.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

RUN_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/lr_grouped_k100/lrgrpk100_s1
PAIRS="${1:-explain_pairs_evfree_diverse.csv}"
OUT_DIR=$REPO_ROOT/headline_results/evaluate_advancement/fidelity_grouped_s1

python evaluate_explanation_fidelity.py \
    --config      "$RUN_DIR/config.yaml" \
    --checkpoint  "$RUN_DIR/best_model.pt" \
    --pairs-csv   "$PAIRS" \
    --methods     ig abs_ig attention random \
    --sparsities  0.05 0.1 0.2 0.5 \
    --n-steps     32 \
    --out-dir     "$OUT_DIR"

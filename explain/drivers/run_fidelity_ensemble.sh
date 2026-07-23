#!/bin/bash
#SBATCH -J fidelity_ens
#SBATCH -o %x.%A_%a.o
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:0:0
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Edge-attribution faithfulness (GraphFramEx protocol) for EACH of the 5
# grouped-ensemble seeds on the 26.03 LATEST build, so faithfulness is
# reported for the deployed ensemble rather than one representative seed.
# Each best_model.pt is the val-selected epoch the ensemble uses (verified).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

SEEDS=(1 7 42 123 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
RUN_DIR=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/lr_grouped_k100_latest/lrgrpk100lat_s${SEED}
PAIRS="${1:-explain_pairs_evfree_sample40.csv}"
OUT_DIR=$REPO_ROOT/headline_results/evaluate_advancement/fidelity_ens/s${SEED}

python explain/cli/evaluate_explanation_fidelity.py \
    --config      "$RUN_DIR/config.yaml" \
    --checkpoint  "$RUN_DIR/best_model.pt" \
    --pairs-csv   "$PAIRS" \
    --methods     ig abs_ig attention random \
    --sparsities  0.05 0.1 0.2 0.5 \
    --n-steps     32 \
    --out-dir     "$OUT_DIR"

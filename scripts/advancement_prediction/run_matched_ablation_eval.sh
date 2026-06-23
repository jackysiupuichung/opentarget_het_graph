#!/bin/bash
#SBATCH -J matched_abl_eval
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=32G
# Steps 2+3 of the matched-recipe ablation (see config/experiments/ablation_matched/README.md):
#   2. build the score-only / novelty-only 5-seed rank-fused ensembles
#   3. evaluate them together with the headline ensemble (+ RDG/OTS from the zarr)
#      and regenerate the result plots.
# Single-process eval -> -c 1 (feedback_eval_one_core). Submit with a dependency
# on the training array so it only runs if all 10 seeds succeeded:
#   sbatch --dependency=afterok:<TRAIN_ARRAY_JOBID> scripts/advancement_prediction/run_matched_ablation_eval.sh
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"

# --- Step 2: build the two ablation ensembles (val-selected, rank-fused) ---
python scripts/advancement_prediction/build_matched_ablation_ensembles.py

# --- Step 3: evaluate score / novelty / both ensembles + regenerate figures ---
# RDG and OTS auto-load from the evaluation_dataset.zarr; defaults point at the
# 26.03 graph/zarr. abl_score_ens / abl_novelty_ens are registered in DEFAULT_RUNS.
python evaluate_advancement.py evaluate \
    --only p3_eahgt_both,abl_score_ens,abl_novelty_ens \
    --results_dir headline_results/ablation_matched_eval

echo "Done. Plots in headline_results/ablation_matched_eval/plots/."
echo "Next (manual, step 4): copy the 5 plots into THBKG/figures/results/ and"
echo "restore the ablation rows in Table 4 + ablation prose/captions."

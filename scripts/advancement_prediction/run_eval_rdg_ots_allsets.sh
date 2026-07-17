#!/bin/bash
#SBATCH -J eval_rdgots
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=32G
# Evaluate ONLY the in-zarr baselines RDG (rdg__no_time__positive) + OTS (ots__all)
# across every generated dataset. No --only (no GNN runs) and no --inject, so the
# eval scores just the two base_models present in each zarr.
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"

DATA=/gpfs/scratch/bty414/clinical_advancement_paper/data
OUT=headline_results/rdg_ots_eval

for SET in datasets_26.03_w2 datasets_26.03_w3 datasets_26.03_w4; do
  echo "=========== EVAL $SET (RDG + OTS only) ==========="
  python evaluate_advancement.py evaluate \
      --datasets_dir "$DATA/$SET" \
      --results_dir "$OUT/$SET"
done

echo "Done -> $OUT/{datasets_26.03_w2,w3,w4}/"

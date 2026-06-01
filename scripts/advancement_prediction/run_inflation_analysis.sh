#!/bin/bash
#SBATCH -J inflation_analysis
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=64G

set -euo pipefail

# Activate venv
source .venv/bin/activate
export WANDB_MODE="disabled"

# Edge-count temporal-inflation analysis around advancement edges (Fig 7 companion).
# Outputs land in advancement_data/results/inflation/.
python evaluate_advancement.py inflation_analysis \
  --out_dir=advancement_data/results/inflation \
  --split=first \
  --compute_centrality=True

#!/bin/bash
#SBATCH -J st_k10_sum_h256
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -A pilot
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail
REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"
source .venv/bin/activate
export WANDB_MODE="disabled"
python src/train_advancement_lambdarank.py \
    --config config/experiments/strict_ta_sweep/st_k10_sum_h256.yaml

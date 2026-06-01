#!/bin/bash
#SBATCH -J lf_v2014_s1
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

# Hardcode the repo root. Slurm copies this script to
# /var/spool/slurmd/jobNNN/slurm_script before executing, so BASH_SOURCE
# and SLURM_SUBMIT_DIR don't reliably point at the original location.
REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

# Per-epoch top-100 test snapshot so we can verify disease-concentration
# behaviour is independent of val-window choice.
export SAVE_PER_EPOCH_TOPK=100

python src/train_advancement_lambdarank.py \
    --config config/experiments/leakfix_val_sweep/v2014_s1.yaml

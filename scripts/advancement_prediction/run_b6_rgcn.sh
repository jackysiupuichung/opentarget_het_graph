#!/bin/bash
#SBATCH -J b6_rgcn
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

python src/train_advancement_lambdarank.py --config config/experiments/b6_rgcn.yaml

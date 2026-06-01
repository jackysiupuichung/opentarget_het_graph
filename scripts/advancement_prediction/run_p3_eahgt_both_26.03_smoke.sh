#!/bin/bash
#SBATCH -J p3_eahgt_both_26.03_smoke
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

# First training pass on the rebuilt 26.03 graph. HPs identical to
# p3_eahgt_both_leakfix; the only change is data.graph_file pointing at the
# new graph. Confirms the pipeline trains end-to-end on the new data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"
export SAVE_PER_EPOCH_TOPK=100

python src/train_advancement_lambdarank.py --config config/experiments/p3_eahgt_both_26.03_smoke.yaml

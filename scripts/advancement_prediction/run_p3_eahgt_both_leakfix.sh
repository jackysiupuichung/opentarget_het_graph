#!/bin/bash
#SBATCH -J p3_eahgt_both_leakfix
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

# Retrain p3_eahgt_both with the rev_advancement label-leak closed in
# build_context_graph (src/data/temporal_loader.py). Identical HPs +
# config to the original v2/v3 production runs; only build_context_graph
# differs. Test metrics from this run vs the original tell us whether the
# leak was material to the published performance.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

# Dump top-100 test predictions per epoch into per_epoch_topk.parquet so we
# can inspect whether the model's high-confidence picks are stable across
# epochs (vs. each epoch hitting a different ~9/10 by chance).
export SAVE_PER_EPOCH_TOPK=100

python src/train_advancement_lambdarank.py --config config/experiments/p3_eahgt_both_leakfix.yaml

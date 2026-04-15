#!/bin/bash
#SBATCH -J advancement_lambdarank
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
# Train HGT with LambdaRank loss on clinical trial advancement prediction.
# Usage:
#   bash scripts/advancement_prediction/train_advancement_lambdarank.sh [--config PATH] [--wandb]
#   sbatch scripts/advancement_prediction/train_advancement_lambdarank.sh [--config PATH] [--wandb]

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

source .venv/bin/activate

CONFIG="config/experiments/advancement_lambdarank.yaml"
WANDB_MODE_VAL="online"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";      shift 2 ;;
        --wandb)    WANDB_MODE_VAL="online"; shift ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "Config : $CONFIG"
echo "W&B    : $WANDB_MODE_VAL"
echo "--------------------------------------------"

export WANDB_MODE="$WANDB_MODE_VAL"

python src/train_advancement_lambdarank.py --config "$CONFIG"

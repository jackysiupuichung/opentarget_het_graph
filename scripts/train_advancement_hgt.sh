#!/bin/bash
#SBATCH -J advancement_hgt
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
# Train HGT on clinical trial advancement prediction.
# Usage:
#   bash scripts/train_advancement_hgt.sh [--config PATH] [--wandb]
#   sbatch scripts/train_advancement_hgt.sh [--config PATH] [--wandb]
#
# Defaults:
#   config  = config/experiments/advancement_hgt.yaml
#   wandb   = disabled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

CONFIG="config/experiments/advancement_hgt.yaml"
WANDB_MODE_VAL="disabled"

# ── Parse args ────────────────────────────────────────────────────────────────
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

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_MODE="$WANDB_MODE_VAL"

python src/train_advancement_hgt.py --config "$CONFIG"

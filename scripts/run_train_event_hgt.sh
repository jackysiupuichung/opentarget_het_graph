#!/bin/bash
#SBATCH -J train_event_hgt
#SBATCH -o %x.o%j
#SBATCH -p gpulong
#SBATCH -n 8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

set -euo pipefail

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "================================================================================"
echo "RUNNING EVENT-BASED SELF-SUPERVISED PRETRAINING (HGT)"
echo "================================================================================"

# 1. HGT Pretrain
echo "▶️  Running HGT (Event)..."
python src/train_self_supervised_event.py --config config/experiments/pretrain_event_hgt.yaml

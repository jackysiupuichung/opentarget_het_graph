#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y

set -euo pipefail

# Activate virtual environment
source .venv/bin/activate





echo "================================================================================"
echo "RUNNING STATIC SELF-SUPERVISED PRETRAINING"
echo "================================================================================"

# 1. HGT Pretrain
echo "▶️  Running HGT..."
python src/train_self_supervised_static.py --config config/experiments/pretrain_static_hgt.yaml

# 2. GATv2 Pretrain
echo "▶️  Running GATv2..."
python src/train_self_supervised_static.py --config config/experiments/pretrain_static_gatv2.yaml

# 3. GATv3 Pretrain
echo "▶️  Running GATv3..."
python src/train_self_supervised_static.py --config config/experiments/pretrain_static_gatv3.yaml
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
echo "RUNNING STATIC GATv2 PRETRAINING"
echo "================================================================================"

python src/train_self_supervised_static.py --config config/experiments/static_gatv2.yaml

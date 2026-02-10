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
echo "RUNNING STATIC HGT FINETUNING"
echo "================================================================================"

python src/train_clinical_multitask.py --config config/experiments/static_hgt.yaml

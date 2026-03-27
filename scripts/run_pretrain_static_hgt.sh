#!/bin/bash
#SBATCH -J pretrain_static_hgt
#SBATCH -o %x.o%j
#SBATCH -p gpulong
#SBATCH -n 8
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

set -euo pipefail

# Activate virtual environment
source .venv/bin/activate

echo "================================================================================"
echo "RUNNING STATIC HGT PRETRAINING"
echo "================================================================================"

python src/train_self_supervised_static.py --config config/experiments/static_hgt.yaml

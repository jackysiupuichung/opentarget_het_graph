#!/bin/bash
#SBATCH -J event_hgt_pretrain
#SBATCH -o %x.o%j
#SBATCH -p gpulong
#SBATCH -n 1
#SBATCH -t 72:0:0
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# Event-based HGT Self-Supervised Pretraining with RTE 
# Uses causal temporal sampling with Relative Temporal Encoding

hostname
date

source .venv/bin/activate

echo "Starting Event-based HGT Pretraining with RTE..."
echo "Config: config/experiments/event_hgt.yaml"

python src/train_self_supervised_event.py \
    --config config/experiments/event_hgt.yaml

echo "Event HGT Pretraining Complete!"
date

#!/bin/bash
#SBATCH -J event_hgt_finetune
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 1
#SBATCH -t 24:0:0
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1

# Event-based HGT Clinical Multi-Task Finetuning
# Uses pretrained event HGT encoder with RTE

hostname
date

source .venv/bin/activate

echo "Starting Event-based HGT Finetuning..."
echo "Config: config/experiments/event_hgt.yaml"

python src/train_clinical_multitask.py \
    --config config/experiments/event_hgt.yaml

echo "Event HGT Finetuning Complete!"
date

#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

source .venv/bin/activate

python -m src.node_features.disease_description_to_vector \
  --disease-dir /data/scratch/bty414/opentarget_evidences/23.06/evidenceDated/diseases \
  --output-dir /data/scratch/bty414/opentarget_evidences/23.06/evidenceDated/node_features/disease_description \
  --model-name gpt2 \
  --batch-size 64 \
  --embedding-dim 256

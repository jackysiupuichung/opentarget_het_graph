#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

source .venv/bin/activate

python -m src.node_features.smile_to_morganfingerprint \
  --drug-dir data/drug \
  --output-dir features/drug/morgan_v1 \
  --fp-dim 1024 \
  --radius 2 \
  --run-pca

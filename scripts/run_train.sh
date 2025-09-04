#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -cwd
#$ -j y

set -euo pipefail

source .venv/bin/activate

python -m src.train --config src/configs/test.yaml
#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

source .venv/bin/activate


python -m src.pipeline.build_progression_graph

#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y

set -euo pipefail

source .venv/bin/activate


python -m src.pipeline.build_progression_graph

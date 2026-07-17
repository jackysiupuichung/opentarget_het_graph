#!/bin/bash
#SBATCH -J prospective_p3_eahgt
#SBATCH -o %x.o%j
#SBATCH -p gpushort
#SBATCH -n 8
#SBATCH --cpus-per-gpu=8
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
cd "$REPO_ROOT"

source .venv/bin/activate
export WANDB_MODE="disabled"

OUTPUT_DIR="/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/p3_eahgt_both_lambdarank_v2"
CSV="advancement_data/prospective_diseases.csv"

# Read EFO IDs (column 'EFO_ID') into a bash array
mapfile -t DISEASES < <(python -c "
import pandas as pd
df = pd.read_csv('${CSV}')
for x in df['EFO_ID'].dropna().tolist():
    print(x)
")

echo "Loaded ${#DISEASES[@]} diseases from ${CSV}"

python evaluate_prospective_standalone.py \
    --output_dir "${OUTPUT_DIR}" \
    --cutoff_year 2015 \
    --ks 100 200 500 \
    --batch_size 512 \
    --num_workers 4 \
    --diseases "${DISEASES[@]}"

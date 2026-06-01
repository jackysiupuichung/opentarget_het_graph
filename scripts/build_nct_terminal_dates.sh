#!/bin/bash
#SBATCH -J build_nct_terminal_dates
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=8G

set -euo pipefail

source .venv/bin/activate

echo "=== Stage A: fetch NCT terminal dates from ClinicalTrials.gov v2 ==="
python -m preprocessing.temporal_graph.parsers.build_nct_terminal_dates

echo ""
echo "=== Stage A.5: coverage audit ==="
python -m preprocessing.temporal_graph.parsers.audit_nct_terminal_dates

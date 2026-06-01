#!/bin/bash
# Submit all 3 EAHGT-allrank_grouped runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do
    sbatch "$f"
done

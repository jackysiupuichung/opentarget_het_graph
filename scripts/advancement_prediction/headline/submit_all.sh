#!/bin/bash
# Submit all 15 headline-sweep runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_*.sh; do
    sbatch "$f"
done

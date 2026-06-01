#!/bin/bash
# Submit all 9 leakfix val-window sweep runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for f in run_v*.sh; do
    sbatch "$f"
done

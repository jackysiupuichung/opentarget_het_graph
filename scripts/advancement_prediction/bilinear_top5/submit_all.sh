#!/bin/bash
# Submit the top-5 bilinear_sweep configs as 5 independent GPU training jobs.
# Each retrains the trial's hyperparameters and emits val + test
# rs_ta_mean@{10,30,50,100} per epoch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

for script in run_top1_eager_sweep27.sh \
              run_top2_peachy_sweep28.sh \
              run_top3_lucky_sweep26.sh \
              run_top4_amber_sweep7.sh \
              run_top5_pretty_sweep16.sh; do
    echo "Submitting $script"
    sbatch "$script"
done

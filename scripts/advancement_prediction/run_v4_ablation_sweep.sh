#!/bin/bash
# Submit the V4-ablation sweep: 7 model variants (P1-P3 + B1, B3, B6, B7) × 3 seeds = 21 jobs.
# Each run uses the V4 narrow-adjacent val window (val_min=val_max=2015) and selects the
# best epoch by val_ndcg_ta_mean@30 (max) — see val-window-sweep analysis for rationale.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$REPO_ROOT"

SWEEP_DIR="$SCRIPT_DIR/v4_ablation"

JOBS=("run_p1_eahgt_score_s0.sh" "run_p1_eahgt_score_s1.sh" "run_p1_eahgt_score_s7.sh" "run_p2_eahgt_novelty_s0.sh" "run_p2_eahgt_novelty_s1.sh" "run_p2_eahgt_novelty_s7.sh" "run_p3_eahgt_both_s0.sh" "run_p3_eahgt_both_s1.sh" "run_p3_eahgt_both_s7.sh" "run_b1_hgt_s0.sh" "run_b1_hgt_s1.sh" "run_b1_hgt_s7.sh" "run_b3_gatv2_s0.sh" "run_b3_gatv2_s1.sh" "run_b3_gatv2_s7.sh" "run_b6_rgcn_s0.sh" "run_b6_rgcn_s1.sh" "run_b6_rgcn_s7.sh" "run_b7_compgcn_s0.sh" "run_b7_compgcn_s1.sh" "run_b7_compgcn_s7.sh")

for job in "${JOBS[@]}"; do
    echo "Submitting $job"
    sbatch "$SWEEP_DIR/$job"
done

echo "All ${#JOBS[@]} jobs submitted. Use 'squeue -u $USER' to monitor."

#!/bin/bash
# Two runs of evaluate_advancement.py to keep figures focused:
#   headline:  EAHGT (s42) — main paper claim (RDG/OTS auto-included)
#   ablation:  HGT/GATv2/R-GCN/CompGCN/EAHGT (all s42) — GNN ablation
# Tabular baselines (RDG/OTS) are auto-loaded from the zarr; we pass
# --only="" to disable DEFAULT_RUNS lookup.
#
# Legend order across both runs is canonical (EAHGT first, baselines last)
# via the _slug_categories breaks list inside evaluate_advancement.

set -euo pipefail

REPO_ROOT="/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph"
cd "$REPO_ROOT"

GRAPH="/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt"
MAPPINGS="/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt"
BASE="/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/headline"
SEED=42

run_eval () {
    local label="$1"; shift
    local archs=("$@")
    local out_dir="${REPO_ROOT}/headline_results/${label}"
    mkdir -p "$out_dir"
    local inject
    inject=$(uv run python - "$BASE" "$SEED" "${archs[@]}" <<'PY'
import json, os, sys
base = sys.argv[1]; seed = sys.argv[2]; archs = sys.argv[3:]
entries = []
for a in archs:
    p = f"{base}/{a}_s{seed}/test_predictions.parquet"
    if os.path.exists(p):
        entries.append({"path": p, "model_name": a})
print(json.dumps(entries))
PY
)
    echo "=== ${label}: ${archs[*]} ==="
    uv run python evaluate_advancement.py evaluate \
        --graph_file "$GRAPH" \
        --mappings_file "$MAPPINGS" \
        --results_dir "$out_dir" \
        --only "" \
        --inject "$inject"
}

run_eval "headline" p3_eahgt_both
run_eval "ablation" b1_hgt b3_gatv2 b6_rgcn b7_compgcn p3_eahgt_both

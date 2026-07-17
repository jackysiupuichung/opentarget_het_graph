#!/bin/bash
#SBATCH -J rebuild_graph_w3
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=72G
#SBATCH -t 2:0:0
# Rebuild the 26.03 datatype graph with 26.03 w3 pipeline advancement labels
# (instead of the 23.06 CSVs the canonical graph uses). Writes to NEW _w3 paths
# so the original 23.06-label graph is untouched.
#   Step 1: build_event_graph -> temporal_graph_datatype_w3.pt (+ _mappings.pt)
#   Step 2: attach_features    -> hetero_graph_with_features_datatype_w3.pt
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate

OT_VERSION=26.03
BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
EVENT_DIR="${BASE}/progression"
FEATURE_DIR="${BASE}/features/processed"
GRAPH_DIR="${BASE}/graph"
ADV_TRAIN="data/clinical_trial_advancement/26.03_w3/train_dataset.csv"
ADV_TEST="data/clinical_trial_advancement/26.03_w3/test_dataset.csv"

echo "[$(date)] STEP 1: build event graph (datatype) with w3 advancement labels"
python preprocessing/temporal_graph/pipeline/build_event_graph.py \
  --input "${EVENT_DIR}/events_datatype.parquet" \
  --output "${EVENT_DIR}/temporal_graph_datatype_w3.pt" \
  --static-edges "${BASE}/evidences/static_edges" \
  --raw-edges "${BASE}/evidences/edges" \
  --advancement-train-csv "$ADV_TRAIN" \
  --advancement-test-csv "$ADV_TEST" \
  --edge-type-mode relation_only
echo "✅ event graph: ${EVENT_DIR}/temporal_graph_datatype_w3.pt (+ _mappings.pt)"

echo "[$(date)] STEP 2: attach features"
python preprocessing/temporal_graph/pipeline/attach_features.py \
  --graph-file "${EVENT_DIR}/temporal_graph_datatype_w3.pt" \
  --output-file "${GRAPH_DIR}/hetero_graph_with_features_datatype_w3.pt" \
  --feature-dir "$FEATURE_DIR"
echo "✅ final graph: ${GRAPH_DIR}/hetero_graph_with_features_datatype_w3.pt"
echo "[$(date)] DONE. mappings: ${EVENT_DIR}/temporal_graph_datatype_w3_mappings.pt"

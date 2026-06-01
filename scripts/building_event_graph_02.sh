#!/bin/bash
#SBATCH -J building_event_graph_02
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=32G

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OT_VERSION="${OT_VERSION:-26.03}"
OUTPUT_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"

# Advancement labels — mixed-version setup: 26.03 graph + 23.06 advancement
# CSVs. Compatibility verified: 98.2% train / 97.9% test endpoints exist in
# the 26.03 graph. See data/clinical_trial_advancement/.
ADV_TRAIN="data/clinical_trial_advancement/23.06/train_dataset.csv"
ADV_TEST="data/clinical_trial_advancement/23.06/test_dataset.csv"

# === Build Graph Structures (datasource-level and datatype-level) ===

# --- Datasource-level ---
echo "🚀 Building Event Graph (datasource-level)..."
python preprocessing/temporal_graph/pipeline/build_event_graph.py \
  --input "${EVENT_OUTPUT_DIR}/events_datasource.parquet" \
  --output "${EVENT_OUTPUT_DIR}/temporal_graph_datasource.pt" \
  --static-edges "$STATIC_EDGES_DIR" \
  --raw-edges "$RAW_EDGES_DIR" \
  --advancement-train-csv "$ADV_TRAIN" \
  --advancement-test-csv "$ADV_TEST" \
  --edge-type-mode relation_only

echo "✅ Datasource graph: ${EVENT_OUTPUT_DIR}/temporal_graph_datasource.pt"

# --- Datatype-level ---
echo ""
echo "🚀 Building Event Graph (datatype-level)..."
python preprocessing/temporal_graph/pipeline/build_event_graph.py \
  --input "${EVENT_OUTPUT_DIR}/events_datatype.parquet" \
  --output "${EVENT_OUTPUT_DIR}/temporal_graph_datatype.pt" \
  --static-edges "$STATIC_EDGES_DIR" \
  --raw-edges "$RAW_EDGES_DIR" \
  --advancement-train-csv "$ADV_TRAIN" \
  --advancement-test-csv "$ADV_TEST" \
  --edge-type-mode relation_only

echo "✅ Datatype graph: ${EVENT_OUTPUT_DIR}/temporal_graph_datatype.pt"

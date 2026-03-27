#!/bin/bash
#SBATCH -J assembling_graph_04
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=16G

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OUTPUT_BASE="/data/scratch/bty414/opentarget_evidences/23.06"
# OUTPUT_BASE="output"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"
FINAL_GRAPH_DIR="${OUTPUT_BASE}/graph"

# === Attach Features to Graph (datasource-level and datatype-level) ===

# --- Datasource-level ---
echo "🚀 Attaching Features (datasource-level)..."
python preprocessing/temporal_graph/pipeline/attach_features.py \
  --graph-file "${EVENT_OUTPUT_DIR}/temporal_graph_datasource.pt" \
  --output-file "${FINAL_GRAPH_DIR}/hetero_graph_with_features_datasource.pt" \
  --feature-dir "$FEATURE_OUTPUT_DIR"
echo "✅ ${FINAL_GRAPH_DIR}/hetero_graph_with_features_datasource.pt"

# --- Datatype-level ---
echo ""
echo "🚀 Attaching Features (datatype-level)..."
python preprocessing/temporal_graph/pipeline/attach_features.py \
  --graph-file "${EVENT_OUTPUT_DIR}/temporal_graph_datatype.pt" \
  --output-file "${FINAL_GRAPH_DIR}/hetero_graph_with_features_datatype.pt" \
  --feature-dir "$FEATURE_OUTPUT_DIR"
echo "✅ ${FINAL_GRAPH_DIR}/hetero_graph_with_features_datatype.pt"

echo ""
echo "✅ Graph Assembly Complete!"

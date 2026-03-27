#!/bin/bash
#SBATCH -J collecting_node_features_03
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 4
#SBATCH -t 12:0:0
#SBATCH --mem-per-cpu=32G

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
# OUTPUT_BASE="/data/scratch/bty414/opentarget_evidences/23.06"
# MAPPINGS_FILE="/data/scratch/bty414/opentarget_evidences/23.06/progression/temporal_graph_mappings.pt"
# EVIDENCE_DIR="/data/scratch/bty414/opentarget_evidences/23.06/evidenceDated"
# FEATURE_DATA_DIR="/data/scratch/bty414/opentarget_evidences/23.06/features/raw"
# FEATURE_OUTPUT_DIR="/data/scratch/bty414/opentarget_evidences/23.06/features/processed"
# TEMP_NODE_DIR="/data/scratch/bty414/opentarget_evidences/23.06/features/temp_nodes"

OUTPUT_BASE="output"
MAPPINGS_FILE="${OUTPUT_BASE}/progression/temporal_graph_sample_mappings.pt"
EVIDENCE_DIR="data/evidenceDated_subset/23.06"
FEATURE_DATA_DIR="data/node_features"
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"
TEMP_NODE_DIR="${OUTPUT_BASE}/features/temp_nodes"

# === Build Node Features ===
echo "🚀 Building Node Features from Graph Mappings..."
echo "   Mappings: $MAPPINGS_FILE"
echo "   Evidence: $EVIDENCE_DIR"
echo "   Raw Feature Data: $FEATURE_DATA_DIR"
echo "   Output: $FEATURE_OUTPUT_DIR"

python -m src.node_features.build_all_features \
  --mappings-file "$MAPPINGS_FILE" \
  --evidence-dir "$EVIDENCE_DIR" \
  --feature-data-dir "$FEATURE_DATA_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR" \
  --temp-dir "$TEMP_NODE_DIR" \
  --random-init

echo ""
echo "✅ Node Feature Collection Complete!"
echo "   Features saved to: $FEATURE_OUTPUT_DIR"
echo "   Temp nodes: $TEMP_NODE_DIR"

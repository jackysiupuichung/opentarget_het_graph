#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=32G
#$ -l h_rt=12:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OUTPUT_BASE="output"
RAW_NODES_DIR="${OUTPUT_BASE}/evidences/nodes"
FEATURE_RAW_DIR="data/node_features"
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"

# === Build Node Features ===
echo "🚀 Building Node Features..."
echo "   Input Nodes: $RAW_NODES_DIR"
echo "   Raw Feature Data: $FEATURE_RAW_DIR"
echo "   Output: $FEATURE_OUTPUT_DIR"

# This invokes target_features, disease_description, molecule_structure
python -m src.node_features.build_all_features \
  --node-dir "$RAW_NODES_DIR" \
  --feature-data-dir "$FEATURE_RAW_DIR" \
  --output-dir "$FEATURE_OUTPUT_DIR"

echo "✅ Node Feature Collection Complete!"
echo "   Features saved to: $FEATURE_OUTPUT_DIR"

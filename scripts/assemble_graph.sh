#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=2:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OUTPUT_BASE="output"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
FEATURE_OUTPUT_DIR="${OUTPUT_BASE}/features/processed"
FINAL_GRAPH_DIR="${OUTPUT_BASE}/graph"

GRAPH_STRUCT_FILE="${EVENT_OUTPUT_DIR}/temporal_graph.pt"
FINAL_GRAPH_FILE="${FINAL_GRAPH_DIR}/hetero_graph_with_features.pt"

# === Attach Features to Graph ===
echo "🚀 [1/2] Attaching Features to Graph..."
echo "   Graph Structure: $GRAPH_STRUCT_FILE"
echo "   Features: $FEATURE_OUTPUT_DIR"
echo "   Output: $FINAL_GRAPH_FILE"

python -m src.pipeline.attach_features \
  --graph-file "$GRAPH_STRUCT_FILE" \
  --output-file "$FINAL_GRAPH_FILE" \
  --feature-dir "$FEATURE_OUTPUT_DIR"

# === Analysis ===
echo "🚀 [2/2] Analyzing Final Graph..."
python -m src.data.analyze_graph \
  --file "$FINAL_GRAPH_FILE" \
  --output "${OUTPUT_BASE}/analysis"

echo "✅ Graph Assembly Complete!"
echo "   Final Graph: $FINAL_GRAPH_FILE"

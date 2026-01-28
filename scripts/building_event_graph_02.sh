#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
OUTPUT_BASE="/data/scratch/bty414/opentarget_evidences/23.06"
# OUTPUT_BASE="output"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
EVENTS_FILE="${EVENT_OUTPUT_DIR}/events.parquet"

GRAPH_STRUCT_FILE="${EVENT_OUTPUT_DIR}/temporal_graph.pt"

# === Build Graph Structure ===
echo "🚀 Building Event Graph Structure..."
python -m src.pipeline.build_event_graph \
  --input "$EVENTS_FILE" \
  --output "$GRAPH_STRUCT_FILE" \
  --static-edges "$STATIC_EDGES_DIR"

echo ""
echo "✅ Event Graph Built!"
echo "   Graph: $GRAPH_STRUCT_FILE"
echo "   Mappings: ${GRAPH_STRUCT_FILE/_graph.pt/_graph_mappings.pt}"

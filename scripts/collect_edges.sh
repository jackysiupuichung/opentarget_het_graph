#!/bin/bash
#$ -pe smp 8
#$ -l h_vmem=64G
#$ -l h_rt=24:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
CONFIG="config/event_graph_config.yaml"

# --- Input ---
INPUT_EVIDENCE_DIR="data/evidenceDated_subset/23.06"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"

# --- Output ---
OUTPUT_BASE="output"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
RAW_NODES_DIR="${KG_OUTPUT_DIR}/nodes"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
EVENTS_FILE="${EVENT_OUTPUT_DIR}/events.parquet"
GRAPH_STRUCT_FILE="${EVENT_OUTPUT_DIR}/temporal_graph.pt"

# === 0. KG Pipeline (Raw Evidence -> Nodes/Edges) ===
echo "🚀 [1/3] Running KG Pipeline..."
echo "   Input: $INPUT_EVIDENCE_DIR"
echo "   Output: $KG_OUTPUT_DIR"

python -m src.pipeline.kg_pipeline \
  --input "$INPUT_EVIDENCE_DIR" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --static-edge-schema "$STATIC_EDGE_SCHEMA" \
  --node-output "$RAW_NODES_DIR" \
  --edge-output "$RAW_EDGES_DIR" \
  --static-edge-output "$STATIC_EDGES_DIR"

# === 1. Build Event List ===
echo "🚀 [2/3] Building Event List..."
if [ ! -f "$CONFIG" ]; then echo "❌ Config $CONFIG not found!"; exit 1; fi

python -m src.pipeline.build_event_list \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_FILE" \
  --aggregation-method "harmonic_sum"

# === 2. Build Graph Structure ===
echo "🚀 [3/3] Building Graph Structure (Nodes + Edges)..."
python -m src.pipeline.build_event_graph \
  --input "$EVENTS_FILE" \
  --output "$GRAPH_STRUCT_FILE" \
  --static-edges "$STATIC_EDGES_DIR"

echo "✅ Edge Collection Complete!"
echo "   Graph Structure: $GRAPH_STRUCT_FILE"

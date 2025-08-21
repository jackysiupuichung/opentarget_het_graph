#!/bin/bash
set -euo pipefail

# === Config ===
NODE_INPUT="ot_files_complete/nodes"
EDGE_INPUT="ot_files_complete/evidence"
NODE_SCHEMA="node_schema.yaml"
EDGE_SCHEMA="edge_schema.yaml"
NODE_OUTPUT="kg_output/nodes"
EDGE_OUTPUT="kg_output/edges"

# === Run pipeline ===
echo "🚀 Running Knowledge Graph pipeline..."
python kg_pipeline.py \
  --node-input "$NODE_INPUT" \
  --edge-input "$EDGE_INPUT" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --node-output "$NODE_OUTPUT" \
  --edge-output "$EDGE_OUTPUT"

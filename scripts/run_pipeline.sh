#!/bin/bash
set -euo pipefail

# activate venv (adjust path if needed)
source .venv/bin/activate

# === Config ===
INPUT="evidenceDated_subset"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"
NODE_OUTPUT="kg_output/nodes"
EDGE_OUTPUT="kg_output/edges"

# === Run pipeline ===
echo "🚀 Running Knowledge Graph pipeline..."
python -m src.pipeline.kg_pipeline \
  --input "$INPUT" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --static-edge-schema "$STATIC_EDGE_SCHEMA" \
  --node-output "$NODE_OUTPUT" \
  --edge-output "$EDGE_OUTPUT"

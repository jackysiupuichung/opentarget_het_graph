#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=8G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# activate venv (adjust path if needed)
source .venv/bin/activate

# === Config ===
INPUT="data/evidenceDated_subset"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"
NODE_OUTPUT="data/kg_output/nodes"
EDGE_OUTPUT="data/kg_output/edges"
STATIC_EDGE_OUTPUT="data/kg_output/static_edges"

# === Run pipeline ===
echo "🚀 Running Knowledge Graph pipeline..."
python -m src.pipeline.kg_pipeline \
  --input "$INPUT" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --static-edge-schema "$STATIC_EDGE_SCHEMA" \
  --node-output "$NODE_OUTPUT" \
  --edge-output "$EDGE_OUTPUT" \
  --static-edge-output "$STATIC_EDGE_OUTPUT"

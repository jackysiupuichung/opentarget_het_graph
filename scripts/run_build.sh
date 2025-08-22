#!/bin/bash
set -euo pipefail

source .venv/bin/activate

EDGE_DIR="kg_output/edges"
NODE_DIR="kg_output/nodes"
CUTOFF=2015
RELATION_MODE="datatype"
OUT="kg_output/hetero_graph.pt"

echo "🚀 Building PyG HeteroData..."
python build_hetero_graph.py \
  --edge-dir "$EDGE_DIR" \
  --node-dir "$NODE_DIR" \
  --cutoff "$CUTOFF" \
  --relation-mode "$RELATION_MODE" \
  --out "$OUT"
echo "✅ HeteroData saved to $OUT"
#!/bin/bash
set -euo pipefail

source .venv/bin/activate

EDGE_DIR="kg_output/edges"
NODE_DIR="kg_output/nodes"
CUTOFF=2010
TEST_HORIZON=5
RELATION_MODE="datatype"
OUT="kg_output/hetero_graph_${CUTOFF}_${TEST_HORIZON}.pt"

echo "🚀 Building PyG HeteroData..."
python -m src.pipeline.build_hetero_graph \
  --edge-dir "$EDGE_DIR" \
  --node-dir "$NODE_DIR" \
  --cutoff "$CUTOFF" \
  --test-horizon "$TEST_HORIZON" \
  --relation-mode "$RELATION_MODE" \
  --out "$OUT"
echo "✅ HeteroData saved to $OUT"

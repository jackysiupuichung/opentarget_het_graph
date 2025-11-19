#!/bin/bash
set -euo pipefail

source .venv/bin/activate

EDGE_DIR="data/kg_output/edges"
NODE_DIR="data/kg_output/nodes"
CUTOFF=2015
TEST_HORIZON=5
RELATION_MODE="datatype"
OUT="data/kg_output/hetero_graph_${CUTOFF}_${TEST_HORIZON}.pt"

python -m src.pipeline.build_progression_graph

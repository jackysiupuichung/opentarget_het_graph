#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=32G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Configuration ===
CONFIG="config/event_graph_config.yaml"

# --- Input ---
INPUT_EVIDENCE_DIR="/data/scratch/bty414/opentarget_evidences/23.06/evidenceDated/"
# INPUT_EVIDENCE_DIR="data/evidenceDated_subset/23.06"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"

# --- Output ---
OUTPUT_BASE="/data/scratch/bty414/opentarget_evidences/23.06"
# OUTPUT_BASE="output"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
RAW_NODES_DIR="${KG_OUTPUT_DIR}/nodes"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"
EVENTS_FILE="${EVENT_OUTPUT_DIR}/events.parquet"

# === 0. KG Pipeline (Raw Evidence -> Nodes/Edges) ===
echo "🚀 [1/2] Running KG Pipeline..."
echo "   Input: $INPUT_EVIDENCE_DIR"
echo "   Output: $KG_OUTPUT_DIR"

python preprocessing/temporal_graph/pipeline/kg_pipeline.py \
  --input "$INPUT_EVIDENCE_DIR" \
  --node-schema "$NODE_SCHEMA" \
  --edge-schema "$EDGE_SCHEMA" \
  --static-edge-schema "$STATIC_EDGE_SCHEMA" \
  --node-output "$RAW_NODES_DIR" \
  --edge-output "$RAW_EDGES_DIR" \
  --static-edge-output "$STATIC_EDGES_DIR"

# === 1. Build Event Lists ===
echo "🚀 [2/3] Building Event Lists (datasource-level and datatype-level)..."
if [ ! -f "$CONFIG" ]; then echo "❌ Config $CONFIG not found!"; exit 1; fi

EVENTS_DATASOURCE_FILE="${EVENT_OUTPUT_DIR}/events_datasource.parquet"
EVENTS_DATATYPE_FILE="${EVENT_OUTPUT_DIR}/events_datatype.parquet"

# --- Datasource-level ---
echo "   [datasource-level]"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_DATASOURCE_FILE" \
  --aggregation-method "harmonic_sum"

# --- Datatype-level ---
echo "   [datatype-level]"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_DATATYPE_FILE" \
  --aggregation-method "harmonic_sum" \
  --datatype-mapping "config/datatype_mapping.yaml"

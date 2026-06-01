#!/bin/bash
#SBATCH -J collecting_edges_01
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=64G

set -euo pipefail

# Activate venv
source .venv/bin/activate

# === Debug mode: set to "--debug" to read only 1 file per datasource subdir ===
DEBUG_FLAG=""
# DEBUG_FLAG="--debug"

# === Configuration ===
CONFIG="config/event_graph_config.yaml"

# === OpenTargets release (23.06 | 26.03) ===
OT_VERSION="${OT_VERSION:-26.03}"

# --- Input ---
EVIDENCE_SUBDIR="evidenceDated"
[ "$OT_VERSION" = "23.06" ] && EVIDENCE_SUBDIR="evidenceDated"
INPUT_EVIDENCE_DIR="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}/${EVIDENCE_SUBDIR}/"
NODE_INPUT_DIR="${INPUT_EVIDENCE_DIR}"
NODE_SCHEMA="config/node_schema.yaml"
EDGE_SCHEMA="config/edge_schema_${OT_VERSION}.yaml"
STATIC_EDGE_SCHEMA="config/static_edge_schema.yaml"

# --- Output ---
OUTPUT_BASE="/gpfs/scratch/bty414/opentarget_evidences/${OT_VERSION}"
# OUTPUT_BASE="output"
KG_OUTPUT_DIR="${OUTPUT_BASE}/evidences"
RAW_EDGES_DIR="${KG_OUTPUT_DIR}/edges"
RAW_NODES_DIR="${KG_OUTPUT_DIR}/nodes"
STATIC_EDGES_DIR="${KG_OUTPUT_DIR}/static_edges"
EVENT_OUTPUT_DIR="${OUTPUT_BASE}/progression"

# === 0. KG Pipeline (Raw Evidence -> Nodes/Edges) === [already completed]
# echo "🚀 [1/2] Running KG Pipeline..."
# echo "   Input: $INPUT_EVIDENCE_DIR"
# echo "   Output: $KG_OUTPUT_DIR"

# python preprocessing/temporal_graph/pipeline/kg_pipeline.py \
#   --input "$INPUT_EVIDENCE_DIR" \
#   --node-input "$NODE_INPUT_DIR" \
#   --node-schema "$NODE_SCHEMA" \
#   --ot-version "$OT_VERSION" \
#   --static-edge-schema "$STATIC_EDGE_SCHEMA" \
#   --node-output "$RAW_NODES_DIR" \
#   --edge-output "$RAW_EDGES_DIR" \
#   --static-edge-output "$STATIC_EDGES_DIR" \
#   $DEBUG_FLAG

# === 1. Build Event Lists ===
echo "🚀 [2/3] Building Event Lists (datasource-level and datatype-level)..."
if [ ! -f "$CONFIG" ]; then echo "❌ Config $CONFIG not found!"; exit 1; fi

EVENTS_DATATYPE_FILE="${EVENT_OUTPUT_DIR}/events_datatype.parquet"

# --- Datatype-level ---
echo "   [datatype-level]"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_DATATYPE_FILE" \
  --datatype-mapping "config/datatype_mapping.yaml"

# --- Datasource-level ---
echo "   [datasource-level]"
python preprocessing/temporal_graph/pipeline/build_event_list.py \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "${EVENT_OUTPUT_DIR}/events_datasource.parquet"

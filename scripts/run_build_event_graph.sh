#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=2:0:0
#$ -cwd
#$ -j y

set -euo pipefail

# Activate virtual environment
source .venv/bin/activate

# === Configuration ===
CONFIG="config/event_graph_config.yaml"
RAW_EDGES_DIR="/data/scratch/bty414/opentarget_evidences/23.06/evidenceDated/kg_output/edges" # Example path, adjust as needed
EVENTS_OUTPUT="output/progression/events.parquet"
GRAPH_OUTPUT="output/progression/temporal_graph.pt"

# === Step 1: Build Event List (Aggregated Events) ===
echo "🚀 Building Event List from ${RAW_EDGES_DIR}..."
echo "Config: ${CONFIG}"

# Check if config exists (it relies on renaming 'progression_config.yaml')
if [ ! -f "$CONFIG" ]; then
    echo "⚠️ Config file $CONFIG not found! Please ensure progression_config.yaml is renamed."
    exit 1
fi

python -m src.pipeline.build_event_list \
  --input-dir "$RAW_EDGES_DIR" \
  --config "$CONFIG" \
  --output "$EVENTS_OUTPUT" \
  --aggregation-method "harmonic_sum"

# === Step 2: Build Event Graph (HeteroData) ===
echo "🚀 Building HeteroData Graph..."

python -m src.pipeline.build_event_graph \
  --input "$EVENTS_OUTPUT" \
  --output "$GRAPH_OUTPUT"

echo "✅ Event Graph Build Pipeline Complete!"

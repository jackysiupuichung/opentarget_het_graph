#!/bin/bash
# Step 05: Build TFT Dataset
# Inherits output paths from collecting_edges_01.sh and collecting_node_features_03.sh.
# Run AFTER Steps 01 and 03 have completed.
#
# Usage:
#   bash scripts/build_tft_dataset_05.sh
#   bash scripts/build_tft_dataset_05.sh --sample-ratio 0.01   # fast test run

#SBATCH -J build_tft_dataset_05
#SBATCH -o %x.o%j
#SBATCH -p compute
#SBATCH -n 1
#SBATCH -t 4:0:0
#SBATCH --mem-per-cpu=32G

set -euo pipefail

source .venv/bin/activate

# === Shared paths (match collecting_edges_01.sh and collecting_node_features_03.sh) ===
OUTPUT_BASE="output"
# OUTPUT_BASE="/data/scratch/bty414/opentarget_evidences/23.06"  # server path

RAW_EDGES_DIR="${OUTPUT_BASE}/evidences/edges"
FEATURE_DIR="${OUTPUT_BASE}/features/processed"
TFT_OUTPUT_DIR="${OUTPUT_BASE}/tft_dataset"

# === TFT-specific parameters ===
TRAIN_MAX=2014
VAL_MAX=2015
TEST_MAX=2022
OUTCOME_MAX=2024
LOOKBACK=10

# Optional: pass --sample-ratio 0.01 for quick test
# EXTRA_ARGS="${@}"

# ── Step 05a: Build longitudinal parquet ──
echo "🚀 [05a] Building TFT longitudinal dataset..."
echo "   Raw edges:  $RAW_EDGES_DIR"
echo "   Output dir: $TFT_OUTPUT_DIR"

python preprocessing/temporal_series/build_tft_dataset.py \
  --raw-edges-dir "$RAW_EDGES_DIR" \
  --output-dir    "$TFT_OUTPUT_DIR" \
  --train-max     "$TRAIN_MAX" \
  --val-max       "$VAL_MAX" \
  --test-max      "$TEST_MAX" \
  --outcome-max   "$OUTCOME_MAX" \
  --lookback      "$LOOKBACK"

echo ""
echo "✅ TFT longitudinal dataset saved to: $TFT_OUTPUT_DIR"

# ── Step 05b: Assemble TFT tensors ──
echo ""
echo "🚀 [05b] Assembling TFT tensors (static + temporal)..."
echo "   Feature dir: $FEATURE_DIR"

python preprocessing/temporal_series/assemble_tft_tensors.py \
  --tft-dir     "$TFT_OUTPUT_DIR" \
  --feature-dir "$FEATURE_DIR" \
  --output      "${TFT_OUTPUT_DIR}/tft_tensors.pt" \
  --lookback    "$LOOKBACK"

echo ""
echo "✅ TFT Dataset Construction Complete!"
echo "   Longitudinal: ${TFT_OUTPUT_DIR}/tft_longitudinal.parquet"
echo "   Anchors:      ${TFT_OUTPUT_DIR}/tft_anchors.parquet"
echo "   Tensors:      ${TFT_OUTPUT_DIR}/tft_tensors.pt"

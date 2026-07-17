#!/bin/bash
#SBATCH -J w3_retrain
#SBATCH -o %x.o%j
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH -t 240:0:0
# Chain ALL 35 w3-retrain runs sequentially inside ONE GPU allocation, so the
# GPU is held for the whole sweep (no re-queueing between runs).
#   20 encoder baselines (~30 min each) + 15 edge-feature ablations (~131 min each)
#   ~= 43h total, well under the 240h cap.
# Encoders run first (fast, cheap wins); ablations after.
set -uo pipefail   # NOT -e: one run failing must not abort the rest
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"
export SAVE_PER_EPOCH_PREDS=1                 # required for ensemble val-selection
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG_ROOT=config/experiments/w3_retrain
SEEDS=(1 7 42 123 2024)

# Ordered config list: encoders first, then ablations.
CONFIGS=()
for enc in hgt gatv2 rgcn compgcn; do
  for s in "${SEEDS[@]}"; do
    CONFIGS+=("$CFG_ROOT/encoder_baselines/$enc/enc_${enc}_w3_s${s}.yaml")
  done
done
for v in score novelty both; do
  for s in "${SEEDS[@]}"; do
    CONFIGS+=("$CFG_ROOT/ablation_matched/$v/abl_${v}_w3_s${s}.yaml")
  done
done

echo "[$(date)] w3 retrain: ${#CONFIGS[@]} runs on one GPU ($(hostname))"
FAIL=0
for i in "${!CONFIGS[@]}"; do
  CFG="${CONFIGS[$i]}"
  echo "======================================================================"
  echo "[$(date)] RUN $((i+1))/${#CONFIGS[@]}: $CFG"
  echo "======================================================================"
  if python src/train_advancement_lambdarank.py --config "$CFG"; then
    echo "[$(date)] OK: $CFG"
  else
    echo "[$(date)] FAILED (continuing): $CFG"
    FAIL=$((FAIL+1))
  fi
done
echo "[$(date)] ALL DONE. failures=$FAIL / ${#CONFIGS[@]}"

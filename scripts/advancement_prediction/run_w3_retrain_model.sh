#!/bin/bash
#SBATCH -J w3_%x
#SBATCH -o w3_retrain_%x.o%j
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH -t 240:0:0
# Chain the 5 seeds of ONE model on one GPU. Submit 7 of these (one per model);
# with 2 GPUs, 2 run at a time. Usage:
#   sbatch --job-name=<model> run_w3_retrain_model.sh <model>
# where <model> is one of: hgt gatv2 rgcn compgcn score novelty both
#   encoders (hgt/gatv2/rgcn/compgcn): ~30 min/seed -> ~2.5h chain
#   ablations (score/novelty/both):    ~131 min/seed -> ~11h chain
set -uo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"
export SAVE_PER_EPOCH_PREDS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${1:?usage: run_w3_retrain_model.sh <hgt|gatv2|rgcn|compgcn|score|novelty|both>}"
CFG_ROOT=config/experiments/w3_retrain
SEEDS=(1 7 42 123 2024)

case "$MODEL" in
  hgt|gatv2|rgcn|compgcn)
    DIR="$CFG_ROOT/encoder_baselines/$MODEL"; PREFIX="enc_${MODEL}_w3" ;;
  score|novelty|both)
    DIR="$CFG_ROOT/ablation_matched/$MODEL"; PREFIX="abl_${MODEL}_w3" ;;
  gatv2_score|gatv2_novelty|gatv2_both)
    V="${MODEL#gatv2_}"; DIR="$CFG_ROOT/gatv2_ablation/$V"; PREFIX="gatv2_${V}_w3" ;;
  *) echo "unknown model: $MODEL"; exit 2 ;;
esac

echo "[$(date)] w3 retrain MODEL=$MODEL, 5 seeds on $(hostname)"
FAIL=0
for s in "${SEEDS[@]}"; do
  CFG="$DIR/${PREFIX}_s${s}.yaml"
  echo "=============================================================="
  echo "[$(date)] $MODEL seed $s -> $CFG"
  echo "=============================================================="
  if python src/train_advancement_lambdarank.py --config "$CFG"; then
    echo "[$(date)] OK: $MODEL s$s"
  else
    echo "[$(date)] FAILED (continuing): $MODEL s$s"; FAIL=$((FAIL+1))
  fi
done
echo "[$(date)] MODEL=$MODEL DONE. failures=$FAIL/5"

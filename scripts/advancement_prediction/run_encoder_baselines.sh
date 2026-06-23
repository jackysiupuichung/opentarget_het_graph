#!/bin/bash
#SBATCH -J enc_base
#SBATCH -o %x.%A_%a.o
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
# Encoder-family baselines for the THBKG ablation (26.03 build).
# 20 runs = 4 encoders {hgt-no-edgefeat, gatv2, rgcn, compgcn} x 5 seeds.
# Each encoder keeps its INDIVIDUAL best params from the prior 23.06 runs
# (model arch + lr + weight_decay + num_neighbors, read from
# 23.06/runs/headline/{e}_s1/config.yaml), but runs under the SAME ensemble
# recipe as the headline (grouped allRank, ndcg_k=100, group_all_tas, 5 seeds,
# val-selected by val_rs_ta_median@50, percentile-rank fused). This isolates
# the encoder family vs the edge-aware EAHGT at each encoder's best settings.
# SAVE_PER_EPOCH_PREDS=1 required for the ensemble build.
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_PER_EPOCH_PREDS=1

CONFIGS=(
  config/experiments/encoder_baselines/hgt/enc_hgt_s1.yaml
  config/experiments/encoder_baselines/hgt/enc_hgt_s7.yaml
  config/experiments/encoder_baselines/hgt/enc_hgt_s42.yaml
  config/experiments/encoder_baselines/hgt/enc_hgt_s123.yaml
  config/experiments/encoder_baselines/hgt/enc_hgt_s2024.yaml
  config/experiments/encoder_baselines/gatv2/enc_gatv2_s1.yaml
  config/experiments/encoder_baselines/gatv2/enc_gatv2_s7.yaml
  config/experiments/encoder_baselines/gatv2/enc_gatv2_s42.yaml
  config/experiments/encoder_baselines/gatv2/enc_gatv2_s123.yaml
  config/experiments/encoder_baselines/gatv2/enc_gatv2_s2024.yaml
  config/experiments/encoder_baselines/rgcn/enc_rgcn_s1.yaml
  config/experiments/encoder_baselines/rgcn/enc_rgcn_s7.yaml
  config/experiments/encoder_baselines/rgcn/enc_rgcn_s42.yaml
  config/experiments/encoder_baselines/rgcn/enc_rgcn_s123.yaml
  config/experiments/encoder_baselines/rgcn/enc_rgcn_s2024.yaml
  config/experiments/encoder_baselines/compgcn/enc_compgcn_s1.yaml
  config/experiments/encoder_baselines/compgcn/enc_compgcn_s7.yaml
  config/experiments/encoder_baselines/compgcn/enc_compgcn_s42.yaml
  config/experiments/encoder_baselines/compgcn/enc_compgcn_s123.yaml
  config/experiments/encoder_baselines/compgcn/enc_compgcn_s2024.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[task $SLURM_ARRAY_TASK_ID] training $CFG"
python src/train_advancement_lambdarank.py --config "$CFG"

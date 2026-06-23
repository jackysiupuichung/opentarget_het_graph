#!/bin/bash
#SBATCH -J matched_abl
#SBATCH -o %x.%A_%a.o
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 4
#SBATCH --cpus-per-gpu=4
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
# Resource ask trimmed from 16c/320G (copied from the headline script) to
# 4c/64G: the headline run's MaxRSS was ~45GB, so 320G was ~7x over-provisioned
# and made the single-GPU tasks hard to schedule into the busy 'mix' nodes.
# Matched-recipe edge-feature ablation for the THBKG paper.
# 10 runs = 5 seeds x {score-only, novelty-only}, SAME recipe as the headline
# 5-seed ensemble (grouped allRank, ndcg_k=100, group_all_tas, 40 ep, ES off).
# Only edge_feat_cols/dim differ ([0]=score, [1]=novelty vs the headline [0,1]).
# After all 10 finish: build per-variant 5-seed rank-fused ensembles, then
# evaluate score/novelty/both together (see build_matched_ablation_ensembles.py).
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# REQUIRED: dump per-epoch test scores so the ensemble build can val-select
# each seed's checkpoint (build_matched_ablation_ensembles.py reads
# per_epoch_preds/epoch_*.parquet). Without this the ensemble step fails.
export SAVE_PER_EPOCH_PREDS=1

CONFIGS=(
  config/experiments/ablation_matched/score/abl_score_s1.yaml
  config/experiments/ablation_matched/score/abl_score_s7.yaml
  config/experiments/ablation_matched/score/abl_score_s42.yaml
  config/experiments/ablation_matched/score/abl_score_s123.yaml
  config/experiments/ablation_matched/score/abl_score_s2024.yaml
  config/experiments/ablation_matched/novelty/abl_novelty_s1.yaml
  config/experiments/ablation_matched/novelty/abl_novelty_s7.yaml
  config/experiments/ablation_matched/novelty/abl_novelty_s42.yaml
  config/experiments/ablation_matched/novelty/abl_novelty_s123.yaml
  config/experiments/ablation_matched/novelty/abl_novelty_s2024.yaml
)
CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "[task $SLURM_ARRAY_TASK_ID] training $CFG"
python src/train_advancement_lambdarank.py --config "$CFG"

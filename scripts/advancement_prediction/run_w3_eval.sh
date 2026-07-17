#!/bin/bash
#SBATCH -J w3_eval
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=48G
# Phase 3: evaluate the w3-retrain ensembles against the 26.03 w3 eval zarr.
# Injects the 7 w3 ensembles (4 encoders + score/novelty/both); RDG/OTS come
# from the w3 zarr. Strata computed from the w3 graph (so masking/context match
# the labels the models trained on).
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"

W3RUNS=/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/w3_retrain
W3ZARR=/gpfs/scratch/bty414/clinical_advancement_paper/data/datasets_26.03_w3
W3GRAPH=/gpfs/scratch/bty414/opentarget_evidences/26.03/graph/hetero_graph_with_features_datatype_w3.pt
W3MAP=/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/temporal_graph_datatype_w3_mappings.pt

# Build the fire --inject JSON list of the 7 w3 ensembles.
INJECT="["
INJECT+="{\"path\":\"$W3RUNS/encoder_baselines/hgt/ensemble/test_predictions.parquet\",\"model_name\":\"enc_hgt_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/encoder_baselines/gatv2/ensemble/test_predictions.parquet\",\"model_name\":\"enc_gatv2_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/encoder_baselines/rgcn/ensemble/test_predictions.parquet\",\"model_name\":\"enc_rgcn_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/encoder_baselines/compgcn/ensemble/test_predictions.parquet\",\"model_name\":\"enc_compgcn_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/ablation_matched/score/ensemble/test_predictions.parquet\",\"model_name\":\"abl_score_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/ablation_matched/novelty/ensemble/test_predictions.parquet\",\"model_name\":\"abl_novelty_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/ablation_matched/both/ensemble/test_predictions.parquet\",\"model_name\":\"p3_eahgt_both_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/gatv2_ablation/score/ensemble/test_predictions.parquet\",\"model_name\":\"gatv2_score_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/gatv2_ablation/novelty/ensemble/test_predictions.parquet\",\"model_name\":\"gatv2_novelty_w3\"},"
INJECT+="{\"path\":\"$W3RUNS/gatv2_ablation/both/ensemble/test_predictions.parquet\",\"model_name\":\"gatv2_both_w3\"}"
INJECT+="]"

python evaluate_advancement.py evaluate \
    --inject "$INJECT" \
    --datasets_dir "$W3ZARR" \
    --graph_file "$W3GRAPH" \
    --mappings_file "$W3MAP" \
    --results_dir headline_results/w3_retrain_eval

echo "Done -> headline_results/w3_retrain_eval/"

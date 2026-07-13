#!/bin/bash
#SBATCH -J enc_base_eval
#SBATCH -o %x.o%j
#SBATCH -p computeshort
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 1:0:0
#SBATCH --mem-per-cpu=32G
# Steps 2+3 of the encoder-family baselines. Single-process eval -> -c 1.
# Submit with a dependency on the encoder training array:
#   sbatch --dependency=afterok:<ENC_ARRAY_JOBID> scripts/advancement_prediction/run_encoder_baselines_eval.sh
set -euo pipefail
cd /data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph
source .venv/bin/activate
export WANDB_MODE="disabled"

# Step 2: build the 4 encoder-family ensembles (val-selected, rank-fused)
python scripts/advancement_prediction/build_encoder_baseline_ensembles.py

# Step 3: evaluate the FULL Table-4 set together (encoders + edge-feature
# variants + headline; RDG/OTS auto-load from the zarr). NOTE: evaluate_advancement.py
# is a fire CLI with subcommands -> use `evaluate`, not a bare flag.
python evaluate_advancement.py evaluate \
    --only enc_hgt_ens,enc_gatv2_ens,enc_rgcn_ens,enc_compgcn_ens,abl_score_ens,abl_novelty_ens,p3_eahgt_both \
    --datasets_dir /gpfs/scratch/bty414/clinical_advancement_paper/data/datasets \
    --results_dir headline_results/full_ablation_eval

echo "Done. Plots in headline_results/full_ablation_eval/plots/."
echo "Next (manual): fold encoder rows into Table 4 + ablation prose; copy figures."

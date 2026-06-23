# Matched-recipe edge-feature ablation

Produces the score-only / novelty-only ablation that is compared **like-for-like**
with the headline EA-HGT ensemble (`p3_eahgt_both` = `grouped_ensemble_s5`):
same grouped allRank recipe (ndcg_k=100, `group_all_tas`, 40 ep, ES off),
same 5 seeds (1/7/42/123/2024), same val-selection + percentile-rank fusion.
Only `edge_feat_cols` differs — `[0]` (score) or `[1]` (novelty) vs the
headline `[0,1]`.

Configs here were generated from `config/experiments/lr_grouped_k100/lrgrpk100_s*.yaml`.

## Run order

1. **Train (10 runs = 5 seeds × 2 variants), GPU array:**
   ```
   sbatch scripts/advancement_prediction/run_matched_ablation.sh
   ```
   Sets `SAVE_PER_EPOCH_PREDS=1` (required — the ensemble build needs
   `per_epoch_preds/epoch_*.parquet`). Outputs under
   `.../26.03/runs/ablation_matched/{score,novelty}/abl_*_s*/`.

2. **Build the two 5-seed ensembles** (compute node, after all 10 finish):
   ```
   python scripts/advancement_prediction/build_matched_ablation_ensembles.py
   ```
   Writes `.../ablation_matched/{score,novelty}/ensemble/test_predictions.parquet`.

3. **Evaluate all three together + regenerate figures** (single core; see
   feedback_eval_one_core), injecting the new registry names:
   ```
   sbatch ... python evaluate_advancement.py \
     --only p3_eahgt_both,abl_score_ens,abl_novelty_ens \
     --results_dir headline_results/ablation_matched_eval
   ```
   (`abl_score_ens` / `abl_novelty_ens` are registered in
   `evaluate_advancement.py` DEFAULT_RUNS; RDG/OTS auto-load from the zarr.)

4. **Copy the 5 plots** from `headline_results/ablation_matched_eval/plots/`
   into `THBKG/figures/results/` under the paper filenames (see the mapping in
   `progress/2026-06-16/FINAL_EVALUATION.md`) and restore the ablation rows in
   Table 4 + the ablation prose / figure captions in `thbkg_20260420.tex`.

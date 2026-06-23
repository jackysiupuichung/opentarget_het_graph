#!/usr/bin/env python3
"""Build the 4 encoder-family baseline ensembles (THBKG ablation, 26.03).

Mirrors build_grouped_ensemble.py / build_matched_ablation_ensembles.py exactly
(val-selection by val_rs_ta_median@50, percentile-rank fusion over the same 5
seeds), for the encoder-family runs from run_encoder_baselines.sh. Run on a
compute node AFTER all 20 training runs finish.
"""
import pandas as pd, numpy as np, glob, os

RUNS = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs"
ref = pd.read_parquet(f"{RUNS}/ndcgk_corr/ndcgk100/test_predictions.parquet").reset_index(drop=True)
SEEDS = [1, 7, 42, 123, 2024]
ENCODERS = ["hgt", "gatv2", "rgcn", "compgcn"]


def val_selected(rd):
    fs = sorted(glob.glob(f"{rd}/per_epoch_preds/epoch_*.parquet"))
    sc = [pd.read_parquet(f)["score"].values for f in fs]
    em = pd.read_csv(f"{rd}/epoch_metrics.csv")
    ep = int(em.loc[em["val_rs_ta_median@50"].idxmax(), "epoch"])
    return sc[ep - 1], ep


for enc in ENCODERS:
    pcts = []
    for s in SEEDS:
        rd = f"{RUNS}/encoder_baselines/{enc}/enc_{enc}_s{s}"
        v, ep = val_selected(rd)
        pcts.append(pd.Series(v).rank(pct=True).values)
        print(f"{enc} s{s}: val-selected epoch {ep}")
    ens = np.mean(np.stack(pcts), axis=0)
    out = f"{RUNS}/encoder_baselines/{enc}/ensemble/test_predictions.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"target_id": ref.target_id, "disease_id": ref.disease_id,
                  "score": ens, "label": ref.label.astype(int)}).to_parquet(out, index=False)
    print(f"wrote {out}\n")

#!/usr/bin/env python3
"""Build the matched-recipe edge-feature ablation ensembles for the THBKG paper.

Mirrors build_grouped_ensemble.py EXACTLY (same val-selection by
val_rs_ta_median@50, same percentile-rank fusion over the same 5 seeds), but for
the score-only and novelty-only variants trained by run_matched_ablation.sh.
Produces one test_predictions.parquet per variant so that score / novelty / both
are compared like-for-like with the headline ensemble.

Run on a compute node AFTER all 10 ablation training runs finish.
"""
import pandas as pd, numpy as np, glob, os

RUNS = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs"
# reference pair ordering (same as the headline ensemble build)
ref = pd.read_parquet(f"{RUNS}/ndcgk_corr/ndcgk100/test_predictions.parquet").reset_index(drop=True)
SEEDS = [1, 7, 42, 123, 2024]


def val_selected(rd):
    fs = sorted(glob.glob(f"{rd}/per_epoch_preds/epoch_*.parquet"))
    sc = [pd.read_parquet(f)["score"].values for f in fs]
    em = pd.read_csv(f"{rd}/epoch_metrics.csv")
    ep = int(em.loc[em["val_rs_ta_median@50"].idxmax(), "epoch"])
    return sc[ep - 1], ep


for variant in ["score", "novelty"]:
    pcts = []
    for s in SEEDS:
        rd = f"{RUNS}/ablation_matched/{variant}/abl_{variant}_s{s}"
        v, ep = val_selected(rd)
        pcts.append(pd.Series(v).rank(pct=True).values)
        print(f"{variant} s{s}: val-selected epoch {ep}")
    ens = np.mean(np.stack(pcts), axis=0)
    out = f"{RUNS}/ablation_matched/{variant}/ensemble/test_predictions.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"target_id": ref.target_id, "disease_id": ref.disease_id,
                  "score": ens, "label": ref.label.astype(int)}).to_parquet(out, index=False)
    print(f"wrote {out}\n")

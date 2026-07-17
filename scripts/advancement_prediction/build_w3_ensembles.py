#!/usr/bin/env python3
"""Build the w3-retrain ensembles (encoder baselines + score/novelty/both
ablation) from runs/w3_retrain. Same recipe as the originals: per-seed
val-selection by val_rs_ta_median@50, percentile-rank fusion over the 5 seeds.

Reference pair ordering = a w3 run's own test_predictions.parquet (7193 pairs),
NOT the old 9094 ndcgk reference.

Run AFTER all 35 w3 training runs finish. Skips any model whose seeds are
incomplete (logs it) so partial progress can still ensemble.
"""
import pandas as pd, numpy as np, glob, os, sys

RUNS = "/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/w3_retrain"
SEEDS = [1, 7, 42, 123, 2024]

ENCODERS = ["hgt", "gatv2", "rgcn", "compgcn"]
ABLATIONS = ["score", "novelty", "both"]
GATV2_ABL = ["gatv2_score", "gatv2_novelty", "gatv2_both"]


def run_dir(model, seed):
    if model in ENCODERS:
        return f"{RUNS}/encoder_baselines/{model}/enc_{model}_w3_s{seed}"
    if model in GATV2_ABL:
        v = model.split("_", 1)[1]
        return f"{RUNS}/gatv2_ablation/{v}/gatv2_{v}_w3_s{seed}"
    return f"{RUNS}/ablation_matched/{model}/abl_{model}_w3_s{seed}"


def val_selected(rd):
    fs = sorted(glob.glob(f"{rd}/per_epoch_preds/epoch_*.parquet"))
    if not fs or not os.path.exists(f"{rd}/epoch_metrics.csv"):
        return None
    sc = [pd.read_parquet(f)["score"].values for f in fs]
    em = pd.read_csv(f"{rd}/epoch_metrics.csv")
    if "val_rs_ta_median@50" not in em.columns:
        return None
    ep = int(em.loc[em["val_rs_ta_median@50"].idxmax(), "epoch"])
    return sc[ep - 1], ep


# reference pair order: first complete w3 run we can find
ref = None
for m in ENCODERS + ABLATIONS:
    for s in SEEDS:
        p = f"{run_dir(m, s)}/test_predictions.parquet"
        if os.path.exists(p):
            ref = pd.read_parquet(p).reset_index(drop=True); break
    if ref is not None: break
if ref is None:
    sys.exit("No w3 test_predictions found yet — training not complete.")
print(f"reference: {len(ref)} pairs, pos-rate {ref.label.mean():.4f}\n")

def ens_out(model):
    if model in ENCODERS:
        return f"{RUNS}/encoder_baselines/{model}/ensemble/test_predictions.parquet"
    if model in GATV2_ABL:
        v = model.split("_", 1)[1]
        return f"{RUNS}/gatv2_ablation/{v}/ensemble/test_predictions.parquet"
    return f"{RUNS}/ablation_matched/{model}/ensemble/test_predictions.parquet"


built = []
for model in ENCODERS + ABLATIONS + GATV2_ABL:
    pcts, used = [], []
    for s in SEEDS:
        r = val_selected(run_dir(model, s))
        if r is None:
            print(f"  {model} s{s}: MISSING (skip)"); continue
        v, ep = r
        pcts.append(pd.Series(v).rank(pct=True).values); used.append(s)
        print(f"  {model} s{s}: val-selected epoch {ep}")
    if not pcts:
        print(f"{model}: no seeds ready, skipping\n"); continue
    ens = np.mean(np.stack(pcts), axis=0)
    out = ens_out(model)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame({"target_id": ref.target_id, "disease_id": ref.disease_id,
                  "score": ens, "label": ref.label.astype(int)}).to_parquet(out, index=False)
    print(f"{model}: ensembled {len(used)} seeds -> {out}\n")
    built.append(model)

print(f"Built {len(built)} ensembles: {built}")

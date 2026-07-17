#!/usr/bin/env python3
"""RDG + OTS ONLY performance across all generated eval datasets.

Reuses evaluate_advancement's own metric functions on just the two in-zarr base
models (rdg__no_time__positive, ots__all) — no GNN injection, no DEFAULT_RUNS.
Reports RS@{10,50,100} (TA-mean over primary TAs + pooled) and classification
(AUROC / average-precision) for 26.03 w2/w3/w4.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import evaluate_advancement as ea

TA_PARQUET = REPO / "advancement_data/features/therapeutic_areas.parquet"
PRIMARY_JSON = REPO / "advancement_data/results/primary_therapeutic_areas.json"
DATA = Path("/gpfs/scratch/bty414/clinical_advancement_paper/data")
SETS = ["datasets_26.03_w2", "datasets_26.03_w3", "datasets_26.03_w4"]
MODELS = ["rdg__no_time__positive", "ots__all"]
LIMITS = [10, 50, 100]

ta = pd.read_parquet(TA_PARQUET)
primary = set(json.load(open(PRIMARY_JSON)))

rows = []
for s in SETS:
    ds = ea._load_dataset(DATA / s / "evaluation_dataset.zarr")
    have = [m for m in MODELS if m in ds.coords["models"].values]
    by_ta = ea._compute_rs_by_ta(ds, ta, have, LIMITS)
    by_lim = ea._compute_relative_success_by_limit(ds, have, LIMITS)  # pooled/global
    cls = ea._compute_classification_metrics(ds, ta, have)
    for m in have:
        rec = {"dataset": s, "model": "RDG" if m.startswith("rdg") else "OTS"}
        # TA-mean RS over primary TAs
        for L in LIMITS:
            v = by_ta[(by_ta.model_name == m) & (by_ta.limit == L) &
                      (by_ta.therapeutic_area_name.isin(primary))]["relative_success"].dropna()
            rec[f"RS@{L}_TAmean"] = round(v.mean(), 2) if len(v) else np.nan
        # pooled RS (global list)
        for L in LIMITS:
            pr = by_lim[by_lim.limit == L]
            pr = pr[pr.model_name == m] if "model_name" in pr.columns else pr
            rec[f"RS@{L}_pooled"] = round(float(pr["relative_success"].iloc[0]), 2) if len(pr) else np.nan
        # classification on the 'all' TA
        ca = cls[(cls.models == m) & (cls.therapeutic_area_name == "all")]
        rec["AUROC"] = round(float(ca.roc_auc.iloc[0]), 3) if len(ca) else np.nan
        rec["AP"] = round(float(ca.average_precision.iloc[0]), 3) if len(ca) else np.nan
        rows.append(rec)

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
out.to_csv(REPO / "headline_results/rdg_ots_summary.csv", index=False)
print("\nwrote headline_results/rdg_ots_summary.csv")

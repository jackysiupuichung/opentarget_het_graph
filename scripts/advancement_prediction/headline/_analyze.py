#!/usr/bin/env python3
"""Produce headline tables + figures from the 15-run sweep + zarr baselines.

Outputs go under `headline_results/`:
  table1_metrics.csv          5 GNNs (mean ± std × 3 seeds) + tabular baselines
  table1_metrics.tex          LaTeX form
  table3_disease_concentration.csv   Top-K unique-disease counts per model
  figure1_rr_at_k.png         RR@K curves with seed-variance bands
  figure2_per_ta_boxplot.png  Per-TA rs_ta_mean@30 boxplots
  figure3_val_test_scatter.png   val_rs_ta_mean@50 vs test_rs_ta_mean@30
  figure4_collapse_trajectory.png   test_rs@10 + #unique-diseases over epochs

Run after all 15 jobs finish:
    uv run python scripts/advancement_prediction/headline/_analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
HEADLINE_DIR = Path("/gpfs/scratch/bty414/opentarget_evidences/26.03/runs/headline")
ZARR = REPO / "advancement_data" / "datasets" / "evaluation_dataset.zarr"
TA_PARQUET = REPO / "advancement_data" / "features" / "therapeutic_areas.parquet"
PRIMARY_TAS_JSON = REPO / "advancement_data" / "results" / "primary_therapeutic_areas.json"
OUT_DIR = REPO / "headline_results"
OUT_DIR.mkdir(exist_ok=True)

GNN_ARCHS = ["b1_hgt", "b3_gatv2", "b6_rgcn", "b7_compgcn", "p3_eahgt_both"]
SEEDS = [1, 7, 42]
TABULAR_MODELS = [
    ("rdg__no_time__positive", "RDG (Czech)"),
    ("gbm__no_time__positive", "GBM (Czech)"),
    ("ots__all", "OTS"),
    ("baseline__most_frequent", "Baseline (most frequent)"),
]
PRETTY_GNN = {
    "b1_hgt": "HGT",
    "b3_gatv2": "GATv2",
    "b6_rgcn": "RGCN",
    "b7_compgcn": "CompGCN",
    "p3_eahgt_both": "EA-HGT (ours)",
}

KEY_TEST_METRICS = [
    "test_rs@10", "test_rs@30", "test_rs@50",
    "test_rs_ta_mean@10", "test_rs_ta_mean@30", "test_rs_ta_mean@50",
    "test_ndcg_ta_mean@30",
    "test_roc_auc", "test_average_precision",
]


# ---------------- Table 1: headline metric table ----------------

def load_gnn_results():
    rows = []
    for arch in GNN_ARCHS:
        for seed in SEEDS:
            d = HEADLINE_DIR / f"{arch}_s{seed}"
            if not (d / "results.yaml").exists():
                print(f"WARN: missing {d}/results.yaml")
                continue
            import yaml
            with open(d / "results.yaml") as f:
                r = yaml.safe_load(f)
            test_block = r.get("test", r)
            row = {"arch": arch, "seed": seed, "best_epoch": r.get("best_epoch")}
            for k in KEY_TEST_METRICS:
                row[k] = test_block.get(k, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_baseline_metrics():
    """Compute rs@K, rs_ta_mean@K, ndcg_ta_mean@K, ROC, AP for each tabular
    baseline by scoring its predictions against the evaluation zarr."""
    import xarray as xr
    import torch
    from src.train_advancement_hgt import compute_metrics
    from src.benchmark.metrics import ndcg_ta_mean_at_k, rs_ta_mean_at_k

    ds = xr.open_zarr(str(ZARR)).load()
    label = ds["outcome"].squeeze("outcomes").values.astype(int)
    disease_ids = ds.coords["disease_id"].values
    target_ids = ds.coords["target_id"].values

    # TA mapping
    ta_df = pd.read_parquet(TA_PARQUET)
    with open(PRIMARY_TAS_JSON) as f:
        primary_tas = json.load(f)
    primary_tas = [t for t in primary_tas if t != "all"]
    # Per-item list of primary TAs
    ta_by_dis = ta_df.groupby("disease_id")["therapeutic_area_name"].apply(
        lambda s: [t for t in s.tolist() if t in primary_tas]
    ).to_dict()
    ta_per_item = [ta_by_dis.get(d, []) for d in disease_ids]

    rows = []
    for slug, display in TABULAR_MODELS:
        scores = ds["prediction"].sel(models=slug, classes="positive").values.astype(float)
        m = compute_metrics(label, scores)
        # TA-mean metrics
        scores_t = torch.from_numpy(scores)
        label_t = torch.from_numpy(label).float()
        for k in (10, 30, 50, 100):
            m[f"ndcg_ta_mean@{k}"] = ndcg_ta_mean_at_k(
                scores_t, label_t, ta_per_item, k, primary_tas=primary_tas
            )
            m[f"rs_ta_mean@{k}"] = rs_ta_mean_at_k(
                scores_t, label_t, ta_per_item, k, primary_tas=primary_tas
            )
        row = {"arch": slug, "display": display, "seed": None, "best_epoch": None}
        # prefix metrics with test_ to match GNN cols
        for k in KEY_TEST_METRICS:
            base_k = k.replace("test_", "")
            row[k] = m.get(base_k, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def build_table1(gnn_df, base_df):
    """Mean ± std per arch (GNNs) plus single values (baselines), one row per
    model, columns = key test metrics."""
    fmt = lambda m, s: (f"{m:.2f} ± {s:.2f}" if (s is not None and not np.isnan(s)) else f"{m:.2f}")

    out = []
    for arch in GNN_ARCHS:
        sub = gnn_df[gnn_df.arch == arch]
        if len(sub) == 0:
            continue
        row = {"model": PRETTY_GNN[arch], "n_seeds": len(sub),
               "best_epoch": fmt(sub.best_epoch.mean(), sub.best_epoch.std())}
        for k in KEY_TEST_METRICS:
            row[k] = fmt(sub[k].mean(), sub[k].std())
        out.append(row)
    for _, r in base_df.iterrows():
        row = {"model": r["display"], "n_seeds": 1, "best_epoch": "-"}
        for k in KEY_TEST_METRICS:
            row[k] = f"{r[k]:.2f}"
        out.append(row)
    return pd.DataFrame(out)


# ---------------- Table 3: disease concentration ----------------

def build_table3_disease_concentration(base_df):
    """For each GNN seed: unique diseases in top-K test predictions.
    For each tabular baseline: same, from zarr predictions.
    Aggregate GNN as mean ± std across seeds; baselines as single values.
    """
    import xarray as xr
    ds = xr.open_zarr(str(ZARR)).load()
    disease_ids = ds.coords["disease_id"].values

    rows = []
    # GNNs
    for arch in GNN_ARCHS:
        per_seed = []
        for seed in SEEDS:
            d = HEADLINE_DIR / f"{arch}_s{seed}"
            pred_path = d / "test_predictions.parquet"
            if not pred_path.exists():
                continue
            pred = pd.read_parquet(pred_path)
            stats = {}
            for k in (10, 30, 100):
                top = pred.nlargest(k, "score")
                stats[f"unique_dst@{k}"] = top.disease_id.nunique()
                stats[f"top1_share@{k}"] = top.disease_id.value_counts().iloc[0] / k
            per_seed.append(stats)
        if per_seed:
            ag = pd.DataFrame(per_seed)
            row = {"model": PRETTY_GNN[arch], "n_seeds": len(per_seed)}
            for c in ag.columns:
                row[c] = f"{ag[c].mean():.1f} ± {ag[c].std():.1f}"
            rows.append(row)

    # Tabular baselines from zarr
    for slug, display in TABULAR_MODELS:
        scores = ds["prediction"].sel(models=slug, classes="positive").values
        pred = pd.DataFrame({"disease_id": disease_ids, "score": scores})
        row = {"model": display, "n_seeds": 1}
        for k in (10, 30, 100):
            top = pred.nlargest(k, "score")
            row[f"unique_dst@{k}"] = f"{top.disease_id.nunique()}"
            row[f"top1_share@{k}"] = f"{top.disease_id.value_counts().iloc[0]/k:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------- Figures ----------------

def figure1_rr_at_k(gnn_df, base_df):
    """RR@K curves with seed-variance bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ks = [10, 30, 50]
    metric_pattern = "test_rs_ta_mean@{k}"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors

    for i, arch in enumerate(GNN_ARCHS):
        sub = gnn_df[gnn_df.arch == arch]
        if len(sub) == 0:
            continue
        means = [sub[metric_pattern.format(k=k)].mean() for k in Ks]
        stds = [sub[metric_pattern.format(k=k)].std() for k in Ks]
        ax.plot(Ks, means, "-o", color=colors[i], label=PRETTY_GNN[arch], linewidth=2)
        ax.fill_between(Ks,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color=colors[i], alpha=0.15)

    for j, (slug, display) in enumerate(TABULAR_MODELS):
        r = base_df[base_df.arch == slug]
        if len(r) == 0:
            continue
        vals = [r[metric_pattern.format(k=k)].iloc[0] for k in Ks]
        ax.plot(Ks, vals, "--s", color=colors[5 + j], label=display, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("K")
    ax.set_ylabel("Relative Success @ K (per-TA mean)")
    ax.set_title("Test RS@K across models (val_year=2013, ES patience=10, rs_ta_mean@50)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure1_rr_at_k.png", dpi=150)
    plt.close(fig)


def figure4_collapse_trajectory():
    """For one representative GNN seed (b1_hgt_s7 by default), overlay
    test_rs@10 trajectory with #unique-diseases-in-top-10 across epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pick = HEADLINE_DIR / "p3_eahgt_both_s7"
    if not (pick / "epoch_metrics.csv").exists():
        return
    epoch_df = pd.read_csv(pick / "epoch_metrics.csv")
    topk_path = pick / "per_epoch_topk.parquet"
    if topk_path.exists():
        tk = pd.read_parquet(topk_path)
        top10 = tk[tk["rank"] < 10]
        unique_per_epoch = top10.groupby("epoch")["dst_idx"].nunique()
    else:
        unique_per_epoch = None

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epoch_df.epoch, epoch_df["test_rs@10"], "-o", color="C0", label="test rs@10")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("test rs@10", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    if unique_per_epoch is not None:
        ax2 = ax1.twinx()
        ax2.plot(unique_per_epoch.index, unique_per_epoch.values,
                 "--s", color="C1", label="#unique diseases in top-10")
        ax2.set_ylabel("#unique diseases in top-10", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(0, 11)
    ax1.set_title(f"Collapse trajectory: {pick.name}")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure4_collapse_trajectory.png", dpi=150)
    plt.close(fig)


# ---------------- Main ----------------

def main():
    print(f"Loading GNN headline runs from {HEADLINE_DIR}")
    gnn = load_gnn_results()
    print(f"  loaded {len(gnn)}/15 runs")
    print(f"\nLoading tabular baselines from {ZARR}")
    base = compute_baseline_metrics()

    print("\nBuilding Table 1 (headline metrics)…")
    t1 = build_table1(gnn, base)
    t1.to_csv(OUT_DIR / "table1_metrics.csv", index=False)
    print(t1.to_string(index=False))

    print("\nBuilding Table 3 (disease concentration)…")
    t3 = build_table3_disease_concentration(base)
    t3.to_csv(OUT_DIR / "table3_disease_concentration.csv", index=False)
    print(t3.to_string(index=False))

    print("\nBuilding Figure 1 (RR@K curves)…")
    figure1_rr_at_k(gnn, base)
    print("\nBuilding Figure 4 (collapse trajectory)…")
    figure4_collapse_trajectory()

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    main()

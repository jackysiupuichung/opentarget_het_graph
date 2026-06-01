#!/usr/bin/env python3
"""Test the per-disease score centering hypothesis.

Hypothesis (from diagnose_collapse.py): the EAHGT decoder produces scores
that are dominated by an additive per-disease offset:
    score(t, d) ≈ b(d) + small_noise(t, d)
with between-disease variation ~14× larger than within-disease variation.

Fix at inference: subtract the per-disease mean before ranking.
    score'(t, d) := score(t, d) − mean_t score(t, d)
This kills the disease-only signal, forcing top-K to be chosen by the
within-disease ranking. If the diagnosis is correct, this should:
    - REDUCE pooled RS@K (because pooled RS rewards collapsing on hot
      diseases and centering removes that ability), AND
    - INCREASE per-TA RS spread (lower n_zero, higher median) by
      forcing the model to pick targets across many TAs.

If neither happens, the collapse is not just an additive offset.

Reads test_predictions.parquet from a run dir, applies centering, runs
the same per-TA + pooled RS computations evaluate_advancement uses, and
prints a side-by-side comparison.

Usage:
    uv run python scripts/diagnose_centering.py \\
        --run /gpfs/scratch/.../runs/headline/p3_eahgt_both_s42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate_advancement as ev  # for _load_dataset, _inject_predictions, _compute_*

ZARR = ROOT / "advancement_data" / "datasets" / "evaluation_dataset.zarr"
TA_PARQUET = ROOT / "advancement_data" / "features" / "therapeutic_areas.parquet"
PRIMARY_TAS_JSON = ROOT / "advancement_data" / "results" / "primary_therapeutic_areas.json"


def _per_disease_centering(scores_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a copy of scores_df with `score` replaced by the chosen
    centering. `mode` ∈ {raw, mean_centered, mean_std_normalised}."""
    df = scores_df.copy()
    if mode == "raw":
        return df
    grouped = df.groupby("disease_id")["score"]
    if mode == "mean_centered":
        df["score"] = df["score"] - grouped.transform("mean")
    elif mode == "mean_std_normalised":
        # z-score within disease: score' = (score - mean_d) / std_d
        df["score"] = (df["score"] - grouped.transform("mean")) / (grouped.transform("std").replace(0, np.nan))
        df["score"] = df["score"].fillna(0.0)
    else:
        raise ValueError(f"unknown mode {mode}")
    return df


def _summarise(rs_by_ta: pd.DataFrame, label: str, limits=(10, 20, 50, 100)) -> pd.DataFrame:
    rows = []
    for L in limits:
        sub = rs_by_ta[rs_by_ta["limit"] == L]
        v = sub["relative_success"].dropna().to_numpy()
        rows.append({
            "mode": label, "N": L,
            "mean": float(np.mean(v)) if len(v) else float("nan"),
            "median": float(np.median(v)) if len(v) else float("nan"),
            "std": float(np.std(v)) if len(v) else float("nan"),
            "n_zero": int((v == 0).sum()),
            "n_gt5": int((v > 5).sum()),
            "n_tas": len(v),
        })
    return pd.DataFrame(rows)


def main(run_dir: Path):
    pred = pd.read_parquet(run_dir / "test_predictions.parquet")
    print(f"Loaded {len(pred):,} predictions from {run_dir.name}")
    print(f"  score range: [{pred.score.min():.4f}, {pred.score.max():.4f}], "
          f"std={pred.score.std():.4f}")
    print(f"  per-disease score-mean range: "
          f"[{pred.groupby('disease_id').score.mean().min():.3f}, "
          f"{pred.groupby('disease_id').score.mean().max():.3f}]")

    # Build the three score variants
    variants = {
        "raw":                  _per_disease_centering(pred, "raw"),
        "mean_centered":        _per_disease_centering(pred, "mean_centered"),
        "mean_std_normalised":  _per_disease_centering(pred, "mean_std_normalised"),
    }

    ta_df = pd.read_parquet(TA_PARQUET)
    with open(PRIMARY_TAS_JSON) as f:
        primary = set(json.load(f)) - {"all"}

    # Use evaluate_advancement's RS computations. Each variant injects a
    # differently-scored prediction set into a fresh zarr copy.
    summaries = []
    per_ta_summaries = []
    for label, df_v in variants.items():
        print(f"\n=== variant: {label} ===")
        out_dir = run_dir / f"centering_{label}"
        out_dir.mkdir(exist_ok=True)
        scored = run_dir / f"_centering_{label}.parquet"
        df_v.to_parquet(scored, index=False)

        ds = ev._load_dataset(ZARR)
        ds = ev._inject_predictions(ds, [{"path": str(scored), "model_name": "eahgt"}])

        # Pooled RS by limit
        rs_pooled = ev._compute_relative_success_by_limit(
            ds, ["eahgt"], limits=[10, 30, 50, 100], confidence=0.9,
        )
        pooled_row = {"variant": label}
        for L in (10, 30, 50, 100):
            r = rs_pooled[rs_pooled["limit"] == L]
            if len(r):
                pooled_row[f"pooled_rs@{L}"] = float(r["relative_success"].iloc[0])
            else:
                pooled_row[f"pooled_rs@{L}"] = float("nan")
        summaries.append(pooled_row)

        # Per-TA RS by limit
        rs_ta = ev._compute_rs_by_ta(
            ds, ta_df, ["eahgt"], limits=[10, 20, 50, 100], confidence=0.9,
        )
        rs_ta = rs_ta[rs_ta["therapeutic_area_name"].isin(primary)]
        per_ta = _summarise(rs_ta, label)
        per_ta_summaries.append(per_ta)

        # Persist per-variant artifacts
        scored.rename(out_dir / "test_predictions.parquet")

    print("\n" + "=" * 70)
    print("Pooled RS@K (whole test set)")
    print("=" * 70)
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 70)
    print("Per-TA RS distribution (across 13 primary TAs)")
    print("=" * 70)
    full = pd.concat(per_ta_summaries, ignore_index=True)
    print(full.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Highlight the key comparisons
    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)
    raw = full[full["mode"] == "raw"].set_index("N")
    mc  = full[full["mode"] == "mean_centered"].set_index("N")
    print(f"{'N':>4}  {'pooled_raw':>11}  {'pooled_mc':>10}   "
          f"{'mean_raw':>9}  {'mean_mc':>9}   "
          f"{'#zero_raw':>10}  {'#zero_mc':>10}")
    pooled_raw_df = pd.DataFrame(summaries).set_index("variant")
    for L in (10, 30, 50, 100):
        if L not in raw.index:
            continue
        print(f"{L:>4}  "
              f"{pooled_raw_df.loc['raw', f'pooled_rs@{L}']:>11.3f}  "
              f"{pooled_raw_df.loc['mean_centered', f'pooled_rs@{L}']:>10.3f}   "
              f"{raw.loc[L,'mean']:>9.3f}  {mc.loc[L,'mean']:>9.3f}   "
              f"{raw.loc[L,'n_zero']:>10d}  {mc.loc[L,'n_zero']:>10d}")
    print("""
If centering helps:
  - pooled RS@K DROPS sharply (loss of the disease-collapse advantage)
  - per-TA mean stays similar OR rises
  - per-TA #zero DROPS (more TAs covered)

If centering doesn't help (or hurts both):
  - the collapse is not just an additive offset
  - bilinear decoder retrain is required
""")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    main(p.parse_args().run)

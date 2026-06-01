#!/usr/bin/env python3
"""Average the 3-seed predictions for each headline architecture into a
single test_predictions.parquet per architecture, named with the canonical
slug evaluate_advancement.py expects (e.g. p3_eahgt_both, b1_hgt, ...).

Writes to /gpfs/scratch/bty414/opentarget_evidences/23.06/runs/headline_agg/
"""
from pathlib import Path
import pandas as pd

BASE = Path("/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/headline")
OUT = Path("/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/headline_agg")
OUT.mkdir(parents=True, exist_ok=True)

ARCHS = ["b1_hgt", "b3_gatv2", "b6_rgcn", "b7_compgcn", "p3_eahgt_both"]
SEEDS = [1, 7, 42]

for arch in ARCHS:
    frames = []
    for seed in SEEDS:
        p = BASE / f"{arch}_s{seed}" / "test_predictions.parquet"
        if not p.exists():
            print(f"WARN missing {p}")
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        continue
    # Sanity: all frames should index over the same (target_id, disease_id) set.
    ref = frames[0][["target_id", "disease_id", "label"]]
    for i, f in enumerate(frames[1:], start=1):
        if not f[["target_id", "disease_id"]].equals(ref[["target_id", "disease_id"]]):
            raise RuntimeError(f"seed {SEEDS[i]} for {arch} has different pair order")
    # Mean score per pair across seeds
    score_cols = [f.rename(columns={"score": f"score_{i}"})[["target_id", "disease_id", f"score_{i}"]]
                  for i, f in enumerate(frames)]
    merged = score_cols[0]
    for sc in score_cols[1:]:
        merged = merged.merge(sc, on=["target_id", "disease_id"])
    score_col_names = [c for c in merged.columns if c.startswith("score_")]
    merged["score"] = merged[score_col_names].mean(axis=1)
    out = ref.merge(merged[["target_id", "disease_id", "score"]],
                    on=["target_id", "disease_id"])
    out_dir = OUT / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"  {arch}: {len(out)} rows, score range [{out.score.min():.3f}, {out.score.max():.3f}], "
          f"agg of {len(frames)} seeds → {out_dir}/test_predictions.parquet")

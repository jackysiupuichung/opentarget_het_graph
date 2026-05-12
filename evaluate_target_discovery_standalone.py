#!/usr/bin/env python3
"""Run the full-pool target-discovery eval against a saved checkpoint.

Reuses ``<output_dir>/config.yaml`` and ``<output_dir>/best_model.pt`` to
rebuild the exact model/context used at training time, then scores the full
candidate pool per eval disease and writes ``<output_dir>/discovery/{per_disease.csv,
macro.csv,predictions.parquet}`` — without retraining.

Example:
  python evaluate_target_discovery_standalone.py --output_dir runs/target_discovery
  python evaluate_target_discovery_standalone.py --output_dir runs/td \
      --eval_diseases EFO_0000676 EFO_0003767
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_advancement_hgt import build_context_graph, ADV_ETYPE
from src.models.utils import build_model
from src.eval.prospective import (
    first_trial_year_by_pair,
    build_split_positive_sets,
)
from src.train_target_discovery import _full_pool_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--eval_diseases", nargs="*", default=None,
                    help="EFO IDs to evaluate (overrides config; default = all with a test positive)")
    ap.add_argument("--eval_batch_size", type=int, default=None)
    ap.add_argument("--eval_num_workers", type=int, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    cfg_path = output_dir / "config.yaml"
    ckpt_path = output_dir / "best_model.pt"
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing config.yaml or best_model.pt in {output_dir}")
    cfg = OmegaConf.load(cfg_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    from src.data.temporal_loader import load_event_graph
    to_undirected = bool(cfg.data.get("undirected", False))
    print(f"Loading graph from {cfg.data.graph_file} (undirected={to_undirected})")
    data = load_event_graph(cfg.data.graph_file, to_undirected=to_undirected)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    disease_map = mappings["node_mapping"]["disease"]
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_target = {v: k for k, v in mappings["node_mapping"]["target"].items()}
    num_targets = data["target"].num_nodes

    d_cfg = cfg.discovery
    split_ranges = {k: tuple(d_cfg.split[k]) for k in ("train", "val", "test")}
    first_year = first_trial_year_by_pair(data)
    pos_sets = build_split_positive_sets(first_year, split_ranges)
    train_pos, val_pos, test_pos = pos_sets["train"], pos_sets["val"], pos_sets["test"]
    excl_test = set(train_pos) | set(val_pos)
    test_cutoff = int(split_ranges["test"][1]) if split_ranges["test"][1] is not None else 9999

    eval_ids = args.eval_diseases if args.eval_diseases is not None else list(d_cfg.get("eval_diseases", []) or [])
    if eval_ids:
        eval_diseases = [disease_map[e] for e in eval_ids if e in disease_map]
    else:
        eval_diseases = sorted({d for (_t, d) in test_pos})
    print(f"Eval diseases: {len(eval_diseases)}")

    context = build_context_graph(data)
    assert ADV_ETYPE not in context.edge_types

    use_recency = bool(cfg.model.get("use_recency", False))
    time_dim = int(cfg.model.get("time_dim", 0))
    t_min_val, t_max_val = 0.0, 1.0
    if use_recency:
        lo = split_ranges["train"][0]
        t_min_val = float(lo) if lo is not None else 1990.0
        t_max_val = float(split_ranges["train"][1]) if split_ranges["train"][1] is not None else 2012.0
    model = build_model(
        model_name=cfg.model.name, data=context,
        hidden_dim=cfg.model.hidden_dim, out_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads, num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout, use_rte=cfg.model.get("use_rte", False),
        use_edge_features=cfg.model.get("use_edge_features", False),
        edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
        use_recency=use_recency, time_dim=time_dim, t_min=t_min_val, t_max=t_max_val,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    eval_bs = args.eval_batch_size or d_cfg.get("eval_batch_size", None)
    eval_bs = int(eval_bs) if eval_bs else int(cfg.train.batch_size) * 4
    eval_nw = args.eval_num_workers if args.eval_num_workers is not None else int(d_cfg.get("eval_num_workers", 4))

    per_df, macro_df, pred_df = _full_pool_eval(
        model, context,
        eval_diseases=eval_diseases, num_targets=num_targets,
        excluded_pairs=excl_test, test_positive_set=test_pos,
        cutoff_year=test_cutoff,
        num_neighbors=list(cfg.train.num_neighbors),
        batch_size=eval_bs, device=device,
        edge_feat_cols=edge_feat_cols, num_workers=eval_nw,
    )
    out_dir = output_dir / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    if per_df is None or per_df.empty:
        print("No metrics produced.")
        return
    per_df = per_df.copy()
    per_df["disease_id"] = per_df["disease_idx"].map(inv_disease)
    per_df.to_csv(out_dir / "per_disease.csv", index=False)
    macro_df.to_csv(out_dir / "macro.csv", index=False)
    if pred_df is not None and not pred_df.empty:
        pred_df = pred_df.copy()
        pred_df["disease_id"] = pred_df["disease_idx"].map(inv_disease)
        pred_df["target_id"] = pred_df["target_idx"].map(inv_target)
        pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    print("\nMacro discovery metrics (vs. random baseline):")
    for _, mr in macro_df.iterrows():
        k = int(mr["K"])
        rand = per_df[per_df["K"] == k]["random_precision_at_k"].mean()
        print(f"  K={k:4d}  P@K={mr['precision_at_k_macro']:.4f}  "
              f"R@K={mr['recall_at_k_macro']:.4f}  (random P@K≈{rand:.5f})")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

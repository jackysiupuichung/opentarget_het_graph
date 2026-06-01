#!/usr/bin/env python3
"""Diagnose representation collapse in EAHGT.

Loads a trained checkpoint, encodes the full graph, then asks:
  (1) Disease embedding norm distribution — are some diseases dominating?
  (2) Per-disease score variance — for each disease, how much does the
      decoder output change across different targets? Low variance =
      collapse (the disease projection saturates the logit).
  (3) Target embedding norm distribution — for context, are targets
      similarly variable?
  (4) Decoder per-disease "characteristic score" — mean over targets,
      tells us which diseases have systematically high scores regardless
      of target.

Run:
  uv run python scripts/diagnose_collapse.py \\
      --checkpoint runs/headline/p3_eahgt_both_s42 \\
      --top-diseases 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import (
    ADV_ETYPE, build_context_graph, build_edge_time_dict as _build_edge_time_dict,
    load_event_graph, split_advancement_edges,
)
from src.models.utils import build_model


def main(checkpoint_dir: Path, top_diseases: int):
    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print(f"Loading graph: {cfg.data.graph_file}")
    data = load_event_graph(cfg.data.graph_file)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    inv_dis = {v: k for k, v in mappings["node_mapping"]["disease"].items()}
    inv_tgt = {v: k for k, v in mappings["node_mapping"]["target"].items()}

    context = build_context_graph(data)
    model = build_model(
        model_name=cfg.model.name,
        data=context,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        use_rte=cfg.model.get("use_rte", False),
        use_edge_features=cfg.model.get("use_edge_features", False),
        edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
        use_recency=cfg.model.get("use_recency", False),
        time_dim=cfg.model.get("time_dim", 0),
        decoder_kind=str(cfg.model.get("decoder_kind", "mlp")),
        decoder_dropout=float(cfg.model.get("decoder_dropout", cfg.model.get("dropout", 0.1))),
    ).to(device)
    sd = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(sd)
    model.eval()

    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    # ---- Encode the WHOLE graph in one pass (no neighbor sampling) ----
    # This is the cleanest representation of what the model thinks each
    # node's embedding is.
    context = context.to(device)
    x_dict = {nt: context[nt].x for nt in context.node_types}
    edge_index_dict = {et: context[et].edge_index for et in context.edge_types}
    edge_time_dict = _build_edge_time_dict(context, exclude_etype=ADV_ETYPE)
    edge_feat_dict = {
        et: context[et].edge_attr[:, edge_feat_cols]
        for et in context.edge_types
        if et != ADV_ETYPE and hasattr(context[et], "edge_attr")
        and context[et].edge_attr is not None
    }

    print("Encoding all nodes (single forward pass on the full context graph)...")
    with torch.no_grad():
        z = model.encode(x_dict, edge_index_dict, edge_time_dict, edge_feat_dict)
    z_target = z["target"].detach().cpu()
    z_disease = z["disease"].detach().cpu()
    print(f"  z_target: {tuple(z_target.shape)}, z_disease: {tuple(z_disease.shape)}")

    # ---- (1) Disease embedding norm distribution ----
    print()
    print("=" * 70)
    print("(1) Disease embedding L2 norms")
    print("=" * 70)
    d_norms = torch.linalg.vector_norm(z_disease, dim=1).numpy()
    t_norms = torch.linalg.vector_norm(z_target, dim=1).numpy()
    print(f"  disease norms:  min={d_norms.min():.3f}  max={d_norms.max():.3f}  "
          f"mean={d_norms.mean():.3f}  std={d_norms.std():.3f}  "
          f"max/median ratio={d_norms.max()/np.median(d_norms):.2f}x")
    print(f"  target norms:   min={t_norms.min():.3f}  max={t_norms.max():.3f}  "
          f"mean={t_norms.mean():.3f}  std={t_norms.std():.3f}  "
          f"max/median ratio={t_norms.max()/np.median(t_norms):.2f}x")
    # Top-K disease by norm
    print(f"\n  Top {top_diseases} diseases by ||z_disease||:")
    top_idx = np.argsort(d_norms)[::-1][:top_diseases]
    for rank, i in enumerate(top_idx[:top_diseases], 1):
        print(f"    {rank:>3}. {inv_dis.get(int(i), f'idx={i}'):<25}  ||z||={d_norms[i]:.4f}")

    # ---- (2) Per-disease score variance across targets ----
    # The decoder takes [z_target; z_disease] -> scalar logit.
    # For each disease d, score everything against all targets, and
    # measure the variance of that score across targets. Low variance =
    # disease projection dominates -> collapse.
    print()
    print("=" * 70)
    print("(2) Per-disease score variance (across all targets)")
    print("=" * 70)
    n_t = z_target.shape[0]
    n_d = z_disease.shape[0]
    print(f"  Scoring {n_d:,} diseases × {n_t:,} targets via the decoder ...")
    score_stats = []
    with torch.no_grad():
        # Score in chunks of 200 diseases to fit memory.
        chunk = 200
        for d_start in range(0, n_d, chunk):
            d_end = min(d_start + chunk, n_d)
            zd_chunk = z_disease[d_start:d_end].to(device)  # [chunk, H]
            # For each disease in the chunk, score against every target.
            # decoder expects [B, H] z_src, [B, H] z_dst.
            for d_local, d_global in enumerate(range(d_start, d_end)):
                zd_rep = zd_chunk[d_local : d_local + 1].expand(n_t, -1)
                zt = z_target.to(device)
                logits = model.decode(zt, zd_rep).detach().cpu().numpy()
                score_stats.append({
                    "disease_idx": d_global,
                    "score_min": float(logits.min()),
                    "score_max": float(logits.max()),
                    "score_mean": float(logits.mean()),
                    "score_std": float(logits.std()),
                })
    df = pd.DataFrame(score_stats)
    df["disease_id"] = df["disease_idx"].map(inv_dis).fillna("unknown")

    # Global stats
    print(f"  Per-disease score-mean range:  "
          f"[{df.score_mean.min():.3f}, {df.score_mean.max():.3f}]  "
          f"(std={df.score_mean.std():.3f})")
    print(f"  Per-disease score-std range:   "
          f"[{df.score_std.min():.4f}, {df.score_std.max():.4f}]  "
          f"(mean={df.score_std.mean():.4f})")
    print(f"  Per-disease score-range (max-min) mean: "
          f"{(df.score_max - df.score_min).mean():.3f}")

    # Top diseases by score_mean (hot diseases — the model assigns them high
    # scores regardless of target)
    print(f"\n  Top {top_diseases} diseases by mean-over-targets score:")
    df_sorted = df.sort_values("score_mean", ascending=False).head(top_diseases)
    print(df_sorted[["disease_id", "score_mean", "score_std", "score_min", "score_max"]].to_string(index=False))

    # Bottom diseases (cold ones)
    print(f"\n  Bottom {top_diseases//2} diseases by mean-over-targets score (cold):")
    print(df.sort_values("score_mean").head(top_diseases//2)[["disease_id", "score_mean", "score_std"]].to_string(index=False))

    # ---- (3) Indicator: is per-disease score variance small relative to ----
    # the between-disease score range? If yes -> collapse.
    print()
    print("=" * 70)
    print("(3) Collapse indicator")
    print("=" * 70)
    between_disease_range = df.score_mean.max() - df.score_mean.min()
    median_within_disease_std = df.score_std.median()
    ratio = between_disease_range / median_within_disease_std if median_within_disease_std > 0 else float("inf")
    print(f"  Between-disease range (max mean − min mean):  {between_disease_range:.3f}")
    print(f"  Median within-disease score std:              {median_within_disease_std:.4f}")
    print(f"  Ratio (between / within):                     {ratio:.1f}")
    print()
    if ratio > 5:
        print("  ▶ Strong collapse: between-disease variation >> within-disease.")
        print("    A 'hot' disease produces high scores for almost any target;")
        print("    the model's top-K is dominated by disease selection.")
    elif ratio > 2:
        print("  ▶ Moderate collapse: disease identity matters more than per-pair.")
    else:
        print("  ▶ No obvious collapse: within-disease variation dominates.")

    # Save for downstream plots
    out_csv = checkpoint_dir / "diagnose_collapse.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote per-disease score stats to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to a trained run dir (must contain best_model.pt + config.yaml)")
    p.add_argument("--top-diseases", type=int, default=20)
    args = p.parse_args()
    main(args.checkpoint, args.top_diseases)

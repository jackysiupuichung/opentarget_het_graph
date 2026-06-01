#!/usr/bin/env python3
"""Ablate molecule (drug) information at inference and measure the impact
on per-pair and per-TA RS@K.

The hypothesis: the graph contains molecule nodes with 1024-dim Morgan-style
features (45k `target -> modulated_by -> molecule` edges), but the EAHGT
predictions may not actually depend on this signal because:
  (i) disease embeddings only access molecules via 2+ hops
  (ii) the 2-layer HGT may not propagate molecule signal far enough
  (iii) disease's own 256-dim text features may dominate over propagated
       molecule signal

Method: load the trained s42 checkpoint, then re-score the test set under
three conditions:
  (1) full graph (baseline)
  (2) molecule edges removed (modulated_by + rev_modulated_by both wiped)
  (3) molecule features randomised (edges intact, features replaced with
      gaussian noise)

If predictions barely change between (1) and (2) -> molecule edges aren't
contributing. If they change between (1) and (3) but not (1) and (2) ->
the model uses molecule edges as graph structure but not the features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import (
    ADV_ETYPE, REV_ADV_ETYPE, build_context_graph,
    build_edge_time_dict as _build_edge_time_dict,
    load_event_graph, split_advancement_edges,
)
from src.models.utils import build_model

MOLECULE_EDGE_TYPES = [
    ("target", "modulated_by", "molecule"),
    ("molecule", "rev_modulated_by", "target"),
]


def load_model(run_dir: Path, device):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    data = load_event_graph(cfg.data.graph_file)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
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
    sd = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return cfg, data, context, mappings, model


def score_pairs(model, context, cfg, edge_label_index, edge_feat_cols, device,
                 disable_molecule_edges=False, randomise_molecule_features=False,
                 rng_seed=42):
    """Encode the full graph (no neighbor sampling), then score every test edge."""
    ctx = context.clone()

    if randomise_molecule_features and "molecule" in ctx.node_types:
        torch.manual_seed(rng_seed)
        ctx["molecule"].x = torch.randn_like(ctx["molecule"].x) * ctx["molecule"].x.std()

    edge_index_dict = {}
    for et in ctx.edge_types:
        if disable_molecule_edges and et in MOLECULE_EDGE_TYPES:
            continue
        edge_index_dict[et] = ctx[et].edge_index.to(device)

    edge_feat_dict = {
        et: ctx[et].edge_attr[:, edge_feat_cols].to(device)
        for et in edge_index_dict.keys()
        if et != ADV_ETYPE and hasattr(ctx[et], "edge_attr")
        and ctx[et].edge_attr is not None
    }
    edge_time_dict = {}
    for et in edge_index_dict.keys():
        if et == ADV_ETYPE:
            continue
        store = ctx[et]
        n = store.edge_index.size(1)
        if hasattr(store, "edge_time") and store.edge_time is not None:
            edge_time_dict[et] = store.edge_time.to(device)
        else:
            edge_time_dict[et] = torch.zeros(n, dtype=torch.long, device=device)

    x_dict = {nt: ctx[nt].x.to(device) for nt in ctx.node_types}

    with torch.no_grad():
        z = model.encode(x_dict, edge_index_dict, edge_time_dict, edge_feat_dict)
        # decode in chunks to fit memory
        scores = []
        chunk = 4096
        for i in range(0, edge_label_index.size(1), chunk):
            ei = edge_label_index[:, i:i+chunk].to(device)
            z_src = z["target"][ei[0]]
            z_dst = z["disease"][ei[1]]
            s = model.decode(z_src, z_dst).cpu().numpy()
            scores.append(s)
    return np.concatenate(scores)


def compute_metrics(scores, labels, ta_df, target_ids, disease_ids, primary):
    rows = []
    for k in (10, 30, 50, 100):
        if len(scores) <= k:
            rows.append({"K": k}); continue
        order = np.argsort(scores)[::-1]
        thr = scores[order][k-1]
        exposed = labels[scores >= thr]; control = labels[scores < thr]
        if len(exposed)==0 or len(control)==0:
            pooled = np.nan
        else:
            pe = exposed.sum() / len(exposed)
            pc = control.sum() / max(len(control),1)
            pooled = pe / pc if pc > 0 else np.nan
        rows.append({"K": k, "pooled": pooled})
    # per-TA
    pred = pd.DataFrame({"target_id": target_ids, "disease_id": disease_ids,
                          "score": scores, "label": labels}).merge(ta_df, on="disease_id")
    perta_mean = {}
    for k in (10, 30, 50, 100):
        vals = []
        for ta in primary:
            sub = pred[pred.therapeutic_area_name == ta]
            if len(sub) <= k: continue
            order = np.argsort(sub.score.to_numpy())[::-1]
            thr = sub.score.to_numpy()[order][k-1]
            exposed = sub.label.to_numpy()[sub.score.to_numpy() >= thr]
            control = sub.label.to_numpy()[sub.score.to_numpy() < thr]
            if len(exposed)==0 or len(control)==0: continue
            pe = exposed.sum() / len(exposed)
            pc = control.sum() / max(len(control),1)
            if pc > 0: vals.append(pe / pc)
        perta_mean[k] = float(np.mean(vals)) if vals else np.nan
    return rows, perta_mean


def main(run_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, data, context, mappings, model = load_model(run_dir, device)

    # Build test edge_label_index
    train_mask, val_mask, test_mask, _ = split_advancement_edges(
        data,
        val_min_year=cfg.data.get("val_min_year"),
        val_max_year=cfg.data.get("val_max_year"),
    )
    ei = data[ADV_ETYPE].edge_index
    ea = data[ADV_ETYPE].edge_attr
    test_idx = test_mask.nonzero(as_tuple=False).flatten()
    test_ei = ei[:, test_idx]
    test_labels = ea[test_idx, 0].cpu().numpy().astype(int)

    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    # Build aligned test_target_ids / test_disease_ids for per-TA stats
    inv_t = {v:k for k,v in mappings['node_mapping']['target'].items()}
    inv_d = {v:k for k,v in mappings['node_mapping']['disease'].items()}
    target_ids = [inv_t[int(i)] for i in test_ei[0].cpu().numpy()]
    disease_ids = [inv_d[int(i)] for i in test_ei[1].cpu().numpy()]

    ta_df = pd.read_parquet(ROOT/"advancement_data/features/therapeutic_areas.parquet")
    ta_df = ta_df[['disease_id','therapeutic_area_name']].drop_duplicates()
    primary = set(json.load(open(ROOT/"advancement_data/results/primary_therapeutic_areas.json"))) - {"all"}

    print(f"Scoring {test_ei.size(1):,} test edges under three conditions...\n")

    for label, kw in [
        ("baseline (full graph)",       {}),
        ("ablate molecule edges",       dict(disable_molecule_edges=True)),
        ("randomise molecule features", dict(randomise_molecule_features=True)),
    ]:
        scores = score_pairs(model, context, cfg, test_ei, edge_feat_cols, device, **kw)
        pooled_rows, perta = compute_metrics(scores, test_labels, ta_df, target_ids, disease_ids, primary)
        print(f"=== {label} ===")
        print(f"  score stats: min={scores.min():.4f}  max={scores.max():.4f}  std={scores.std():.4f}")
        for r in pooled_rows:
            k = r["K"]
            print(f"  K={k:3d}  pooled_rs={r.get('pooled', float('nan')):.3f}  perTA_mean={perta.get(k, float('nan')):.3f}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    main(p.parse_args().run)

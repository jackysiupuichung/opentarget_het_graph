#!/usr/bin/env python3
"""Train + evaluate an HGT *target-discovery* recommender with LambdaRank.

Unlike ``train_advancement_lambdarank.py`` (which ranks among already-trialed
(target, disease) pairs), this trainer ranks the full (target, disease)
candidate population: for a chronological year-range split, a pair's label in a
split is "did this pair acquire its first trial-related edge inside that split's
time window?" Splits are mutually exclusive (a pair positive in train is never a
val/test candidate, etc.).

Pipeline
--------
* Context graph = the input graph with **all** ``("target","advancement","disease")``
  edges removed (``build_context_graph``). Candidate pairs enter only as
  ``edge_label_index`` — never as message-passing edges — so there is no label
  leakage through the graph structure. Temporal neighbor sampling
  (``time_attr="edge_time"``, ``edge_label_time = split cutoff``) additionally
  hides context edges newer than the cutoff.
* Train: per-disease ranking slates (all positives + ``neg_ratio`` sampled
  negatives per positive), rebuilt every epoch with fresh negatives. LambdaRank
  loss applied per-disease within each batch, averaged.
* Early stopping: macro NDCG@k over the val slates.
* Final eval: score the **full** candidate pool per eval-disease, report macro
  precision@K / recall@K vs. a random baseline, plus a per-disease CSV.

Reuses ``build_context_graph`` / ``predict_test`` / ``_build_edge_time_dict``
from ``train_advancement_hgt`` (which key ``ADV_ETYPE`` as the label relation —
that is fine, ``LinkNeighborLoader`` accepts label pairs for an edge type that
is not present in the data, which is exactly our case), the model builders, and
the discovery helpers in ``src/eval/prospective.py``.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.special import expit
from torch_geometric.loader import LinkNeighborLoader
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_advancement_hgt import (
    ADV_ETYPE,
    TeeLogger,
    build_context_graph,
    _build_edge_time_dict,
    predict_test,
)
from src.losses.lambdaLoss_allrank import lambdaLoss
from src.benchmark.metrics import ndcg_at_k
from src.models.utils import build_model
from src.eval.prospective import (
    first_trial_year_by_pair,
    build_split_positive_sets,
    build_discovery_slates,
    compute_discovery_metrics,
)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def _make_flat_loss(cfg):
    """Return a (logits_1d, labels_1d) -> scalar ranking loss for one slate."""
    lr_cfg = cfg.train.lambdarank
    impl = str(lr_cfg.get("impl", "allrank")).lower()
    sigma = float(lr_cfg.get("sigma", 1.0))
    ndcg_k = lr_cfg.get("ndcg_k", 50)
    ndcg_k = int(ndcg_k) if ndcg_k is not None else None
    if impl in ("allrank", "allrank_grouped"):
        # train_target_discovery already batches per-disease and applies the
        # flat slate loss per group; "allrank_grouped" at the config level is
        # therefore equivalent here. Both map to the flat [1, slate] call.
        weighing_scheme = str(lr_cfg.get("weighing_scheme", "lambdaRank_scheme"))
        reduction = str(lr_cfg.get("reduction", "sum"))
        reduction_log = str(lr_cfg.get("reduction_log", "binary"))
        def loss_fn(logits, labels):
            return lambdaLoss(
                logits.view(1, -1), labels.view(1, -1),
                sigma=sigma, k=ndcg_k,
                weighing_scheme=weighing_scheme,
                reduction=reduction, reduction_log=reduction_log,
            )
        return loss_fn, (
            f"allRank lambdaLoss flat (sigma={sigma}, k={ndcg_k}, "
            f"weighing_scheme={weighing_scheme})"
        )
    raise ValueError(f"Unknown lambdarank.impl: {impl!r}. Expected 'allrank' or 'allrank_grouped'.")


def _grouped_batch_loss(logits, labels, disease_idx, flat_loss_fn):
    """Per-disease lambdarank within a batch, averaged over valid groups."""
    uniq = torch.unique(disease_idx)
    total = logits.new_zeros(())
    n = 0
    for d in uniq.tolist():
        m = disease_idx == d
        if m.sum() < 2:
            continue
        gl, gy = logits[m], labels[m]
        s = gy.sum()
        if s == 0 or s == gy.numel():
            continue
        total = total + flat_loss_fn(gl, gy)
        n += 1
    if n == 0:
        return 0.0 * logits.sum()
    return total / n


# --------------------------------------------------------------------------- #
# Slate loaders
# --------------------------------------------------------------------------- #
def _make_slate_loader(context, edge_label_index, edge_label, edge_label_time,
                       num_neighbors, batch_size, shuffle):
    return LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_label_index),
        edge_label=edge_label,
        edge_label_time=edge_label_time,
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _forward_logits(model, batch, edge_feat_cols):
    edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
    return model(
        batch.x_dict,
        batch.edge_index_dict,
        batch[ADV_ETYPE].edge_label_index,
        src_type="target",
        dst_type="disease",
        edge_time_dict=edge_time_dict,
        edge_feat_dict={
            et: batch[et].edge_attr[:, edge_feat_cols]
            for et in batch.edge_types
            if et != ADV_ETYPE and hasattr(batch[et], "edge_attr")
            and batch[et].edge_attr is not None
        },
        edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
    )


def run_epoch(model, loader, optimizer, device, edge_feat_cols, flat_loss_fn, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            logits = _forward_logits(model, batch, edge_feat_cols)
            labels = batch[ADV_ETYPE].edge_label.float()
            disease_local = batch[ADV_ETYPE].edge_label_index[1]
            disease_idx = batch["disease"].n_id[disease_local]
            loss = _grouped_batch_loss(logits, labels, disease_idx, flat_loss_fn)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_slates(model, loader, device, edge_feat_cols, flat_loss_fn, ks):
    """Macro NDCG@k over val slates, plus the slate-level ranking loss."""
    model.eval()
    all_logits, all_labels, all_disease = [], [], []
    for batch in loader:
        batch = batch.to(device)
        logits = _forward_logits(model, batch, edge_feat_cols)
        all_logits.append(logits.cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())
        disease_local = batch[ADV_ETYPE].edge_label_index[1]
        all_disease.append(batch["disease"].n_id[disease_local].cpu())
    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels).float()
    disease_t = torch.cat(all_disease)

    # Slate-level loss (group by disease over the whole val set).
    val_loss = float(_grouped_batch_loss(logits_t, labels_t, disease_t, flat_loss_fn).item())

    scores_t = torch.from_numpy(expit(logits_t.numpy()))
    metrics = {"val_loss": val_loss}
    # Macro NDCG@k: per-disease NDCG, averaged.
    for k in ks:
        per = []
        for d in torch.unique(disease_t).tolist():
            m = disease_t == d
            if m.sum() < 2 or labels_t[m].sum() == 0:
                continue
            per.append(ndcg_at_k(scores_t[m], labels_t[m], k))
        metrics[f"ndcg@{k}"] = float(np.mean(per)) if per else 0.0
    return metrics


# --------------------------------------------------------------------------- #
# Full-pool eval
# --------------------------------------------------------------------------- #
def _full_pool_eval(model, context, *, eval_diseases, num_targets, excluded_pairs,
                    test_positive_set, cutoff_year, num_neighbors, batch_size,
                    device, edge_feat_cols, num_workers):
    """Score every candidate target for each eval disease, return per-disease/macro."""
    src_list, dst_list = [], []
    offsets = []  # (disease_idx, start, n)
    full_counts: dict[int, int] = {}
    for d in eval_diseases:
        excl_d = {t for (t, dd) in excluded_pairs if dd == d}
        cand = [t for t in range(num_targets) if t not in excl_d]
        if not cand:
            continue
        offsets.append((d, len(src_list), len(cand)))
        src_list.extend(cand)
        dst_list.extend([d] * len(cand))
        full_counts[d] = len(cand)
    if not src_list:
        return None, None, None
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    n = edge_index.shape[1]
    scores, _ = predict_test(
        model, context,
        edge_index=edge_index,
        edge_labels=torch.zeros(n, dtype=torch.float),
        edge_times=torch.full((n,), int(cutoff_year), dtype=torch.long),
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        device=device,
        edge_feat_cols=edge_feat_cols,
        num_workers=num_workers,
    )
    per_df, macro_df, pred_df = compute_discovery_metrics(
        scores=np.asarray(scores),
        disease_idx=np.asarray(dst_list),
        target_idx=np.asarray(src_list),
        positive_set=test_positive_set,
        full_candidate_counts=full_counts,
        ks=[100, 200, 500],
    )
    return per_df, macro_df, pred_df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg):
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    tee = TeeLogger(output_dir / "train.log")
    sys.stdout = tee

    wandb.init(
        project="target_discovery",
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(output_dir),
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Graph + mappings -------------------------------------------------- #
    from src.data.temporal_loader import load_event_graph
    to_undirected = bool(cfg.data.get("undirected", False))
    print(f"Loading graph from {cfg.data.graph_file} (undirected={to_undirected})")
    data = load_event_graph(cfg.data.graph_file, to_undirected=to_undirected)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    disease_map = mappings["node_mapping"]["disease"]
    inv_disease = {v: k for k, v in disease_map.items()}
    inv_target = {v: k for k, v in mappings["node_mapping"]["target"].items()}
    num_targets = data["target"].num_nodes

    # ---- Chronological split by first-trial-year --------------------------- #
    d_cfg = cfg.discovery
    split_ranges = {
        k: tuple(d_cfg.split[k]) for k in ("train", "val", "test")
    }
    print(f"Discovery split (first-trial-year windows): {split_ranges}")
    first_year = first_trial_year_by_pair(data)
    pos_sets = build_split_positive_sets(first_year, split_ranges)
    train_pos, val_pos, test_pos = pos_sets["train"], pos_sets["val"], pos_sets["test"]
    print(f"  train positives: {len(train_pos)} | val: {len(val_pos)} | test: {len(test_pos)}")
    assert train_pos.isdisjoint(val_pos) and train_pos.isdisjoint(test_pos) and val_pos.isdisjoint(test_pos)

    # Excluded-pair sets: for split S, exclude pairs positive in any earlier split.
    excl_train: set = set()
    excl_val: set = set(train_pos)
    excl_test: set = set(train_pos) | set(val_pos)

    # Cutoff years per split (upper bound of the window; None -> use a large year).
    def _cutoff(rng_tuple):
        hi = rng_tuple[1]
        return int(hi) if hi is not None else 9999
    train_cutoff = _cutoff(split_ranges["train"])
    val_cutoff = _cutoff(split_ranges["val"])
    test_cutoff = _cutoff(split_ranges["test"])

    # ---- Disease lists ----------------------------------------------------- #
    def _resolve_disease_list(cfg_list, positive_set):
        ids = list(cfg_list or [])
        if ids:
            idxs = [disease_map[e] for e in ids if e in disease_map]
            missing = [e for e in ids if e not in disease_map]
            if missing:
                print(f"  [warn] {len(missing)} configured disease IDs not in mapping: {missing[:5]}...")
            return idxs
        # all diseases with >=1 positive in this split
        return sorted({d for (_t, d) in positive_set})

    train_diseases = _resolve_disease_list(d_cfg.get("train_diseases", []), train_pos)
    eval_diseases = _resolve_disease_list(d_cfg.get("eval_diseases", []), test_pos)
    print(f"  train diseases: {len(train_diseases)} | eval diseases: {len(eval_diseases)}")
    neg_ratio = int(d_cfg.get("neg_ratio", 50))
    ks = [int(k) for k in d_cfg.get("ks", [100, 200, 500])]

    # ---- Context graph (advancement edges removed) ------------------------- #
    print("Building context graph (dropping all ('target','advancement','disease') edges)...")
    context = build_context_graph(data)
    assert ADV_ETYPE not in context.edge_types, "advancement edges leaked into context"

    # ---- Model ------------------------------------------------------------- #
    use_recency = bool(cfg.model.get("use_recency", False))
    time_dim = int(cfg.model.get("time_dim", 0))
    t_min_val, t_max_val = (0.0, 1.0)
    if use_recency:
        # No advancement edge_time available; use the train split window.
        lo = split_ranges["train"][0]
        t_min_val = float(lo) if lo is not None else 1990.0
        t_max_val = float(train_cutoff)
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
        use_recency=use_recency,
        time_dim=time_dim,
        t_min=t_min_val,
        t_max=t_max_val,
    ).to(device)
    print(f"Model: {cfg.model.name}")
    edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    flat_loss_fn, loss_desc = _make_flat_loss(cfg)
    print(f"LambdaRank loss: {loss_desc} (applied per-disease within batch)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.get("cosine_t_max", cfg.train.num_epochs)),
        eta_min=cfg.train.get("eta_min", 1e-6),
    )
    num_neighbors = list(cfg.train.num_neighbors)
    batch_size = int(cfg.train.batch_size)

    # ---- Fixed val slates (negatives sampled once, deterministically) ------ #
    val_eli, val_lbl, val_t, _val_grp, _val_d = build_discovery_slates(
        diseases=sorted({d for (_t, d) in val_pos}),
        num_targets=num_targets,
        positive_set=val_pos,
        excluded_pairs=excl_val,
        neg_ratio=neg_ratio,
        rng=np.random.default_rng(seed + 1),
        cutoff_year=val_cutoff,
    )
    val_loader = _make_slate_loader(
        context, val_eli, val_lbl, val_t, num_neighbors, batch_size, shuffle=False
    )
    print(f"Val slates: {val_eli.shape[1]} rows over "
          f"{len(set(_val_d.tolist()))} diseases.")

    # ---- Training loop ----------------------------------------------------- #
    es_cfg = cfg.train.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 5)) if es_enabled else int(1e9)
    es_metric = str(es_cfg.get("metric", "ndcg@50"))
    print(f"Early stopping on val/{es_metric}, patience={patience}")
    ckpt_path = output_dir / "best_model.pt"

    best_val = -1.0
    best_epoch = 0
    patience_ctr = 0
    epoch_rows = []
    train_disease_pool = train_diseases  # constant; negatives reshuffled each epoch

    for epoch in range(1, cfg.train.num_epochs + 1):
        # Rebuild train slates with fresh negatives each epoch.
        tr_eli, tr_lbl, tr_t, _tr_grp, _tr_d = build_discovery_slates(
            diseases=train_disease_pool,
            num_targets=num_targets,
            positive_set=train_pos,
            excluded_pairs=excl_train,
            neg_ratio=neg_ratio,
            rng=rng,
            cutoff_year=train_cutoff,
        )
        train_loader = _make_slate_loader(
            context, tr_eli, tr_lbl, tr_t, num_neighbors, batch_size, shuffle=True
        )
        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            edge_feat_cols=edge_feat_cols, flat_loss_fn=flat_loss_fn, train=True,
        )
        scheduler.step()

        val_metrics = evaluate_slates(
            model, val_loader, device,
            edge_feat_cols=edge_feat_cols, flat_loss_fn=flat_loss_fn, ks=(10, 30, 50, 100),
        )
        val_score = val_metrics.get(es_metric, float("nan"))
        if np.isnan(val_score):
            val_score = -1.0

        row = {"epoch": epoch, "train_loss": train_loss,
               "n_train_rows": int(tr_eli.shape[1])}
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        epoch_rows.append(row)
        wandb.log({"train/loss": train_loss} | {f"val/{k}": v for k, v in val_metrics.items()},
                  step=epoch)
        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} "
            f"| val ndcg@10/50/100: {val_metrics['ndcg@10']:.4f}/"
            f"{val_metrics['ndcg@50']:.4f}/{val_metrics['ndcg@100']:.4f} "
            f"| val_loss {val_metrics['val_loss']:.4f}"
        )

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    print(f"Best epoch: {best_epoch} | best val {es_metric}: {best_val:.4f}")
    if ckpt_path.exists():
        print(f"Loading best checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    pd.DataFrame(epoch_rows).to_csv(output_dir / "epoch_metrics.csv", index=False)
    print(f"Epoch metrics saved to {output_dir / 'epoch_metrics.csv'}")

    # ---- Full-pool test eval ---------------------------------------------- #
    eval_bs = d_cfg.get("eval_batch_size", None)
    eval_bs = int(eval_bs) if eval_bs else batch_size * 4
    eval_nw = int(d_cfg.get("eval_num_workers", 4))
    print(f"Full-pool eval over {len(eval_diseases)} diseases "
          f"(batch_size={eval_bs}, num_workers={eval_nw})...")
    per_df, macro_df, pred_df = _full_pool_eval(
        model, context,
        eval_diseases=eval_diseases,
        num_targets=num_targets,
        excluded_pairs=excl_test,
        test_positive_set=test_pos,
        cutoff_year=test_cutoff,
        num_neighbors=num_neighbors,
        batch_size=eval_bs,
        device=device,
        edge_feat_cols=edge_feat_cols,
        num_workers=eval_nw,
    )

    out_dir = output_dir / "discovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "best_epoch": int(best_epoch),
        f"best_val_{es_metric}": float(best_val),
        "n_train_positives": len(train_pos),
        "n_val_positives": len(val_pos),
        "n_test_positives": len(test_pos),
        "n_eval_diseases": len(eval_diseases),
    }
    if per_df is not None and not per_df.empty:
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
        # Mean random baseline across diseases, per K, for context.
        for _, mr in macro_df.iterrows():
            k = int(mr["K"])
            rand = per_df[per_df["K"] == k]["random_precision_at_k"].mean()
            print(f"  K={k:4d}  P@K={mr['precision_at_k_macro']:.4f}  "
                  f"R@K={mr['recall_at_k_macro']:.4f}  "
                  f"(random P@K≈{rand:.5f}, n_diseases={int(mr['n_diseases'])})")
            results[f"test_precision_at_{k}_macro"] = float(mr["precision_at_k_macro"])
            results[f"test_recall_at_{k}_macro"] = float(mr["recall_at_k_macro"])
            results[f"random_precision_at_{k}_mean"] = float(rand)
        wandb.log({f"test/{k}": v for k, v in results.items()
                   if isinstance(v, (int, float))})
        print(f"Per-disease metrics saved to {out_dir / 'per_disease.csv'}")
    else:
        print("Full-pool eval produced no metrics (no eval diseases with positives?).")

    OmegaConf.save(OmegaConf.create(results), output_dir / "results.yaml")
    print(f"Results saved to {output_dir / 'results.yaml'}")

    wandb.finish()
    sys.stdout = tee._terminal
    tee.close()
    print(f"Log saved to {output_dir / 'train.log'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/target_discovery.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)

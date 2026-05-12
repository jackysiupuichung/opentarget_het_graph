#!/usr/bin/env python3
"""Train HGT for advancement link prediction with LambdaRank loss.

Mirrors src/train_advancement_hgt.py but swaps the pointwise BCE/focal
objective for a batch-level LambdaRank loss and uses NDCG@K for early
stopping. Graph construction, splits, model, and sampling are identical so
runs can be compared against the BCE baseline.
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from scipy.special import expit
from torch_geometric.loader import LinkNeighborLoader
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_advancement_hgt import (
    ADV_ETYPE,
    TeeLogger,
    split_advancement_edges,
    build_context_graph,
    _build_edge_time_dict,
    compute_metrics,
    predict_test,
)
from src.losses.lambdaLoss_allrank import lambdaLoss
from src.benchmark.metrics import ndcg_at_k, ndcg_ta_mean_at_k, rr_ta_mean_at_k
from src.models.utils import build_model


def _build_grouped_slates(logits, labels, ta_groups_per_item, primary_tas, padded_value=-1.0):
    """Reshape flat batch → [n_groups, max_slate_length] padded with -1.

    For each primary TA seen in this batch, gather the items belonging to it
    (one item can sit in multiple TAs; it's replicated across those TA slates).
    Pad the shorter slates with `padded_value` so they share a row length.
    Returns y_pred, y_true, or None if no TA had ≥2 items with at least one
    positive (allRank's lambdaLoss needs both for valid pairs).
    """
    by_ta_pred: dict = {}
    by_ta_true: dict = {}
    for i, tas in enumerate(ta_groups_per_item):
        if not tas:
            continue
        for ta in tas:
            if ta not in primary_tas:
                continue
            by_ta_pred.setdefault(ta, []).append(logits[i])
            by_ta_true.setdefault(ta, []).append(labels[i])

    if not by_ta_pred:
        return None, None

    valid_groups = [
        ta for ta, lab_list in by_ta_true.items()
        if len(lab_list) >= 2 and torch.stack(lab_list).sum() > 0
        and torch.stack(lab_list).sum() < len(lab_list)
    ]
    if not valid_groups:
        return None, None

    max_len = max(len(by_ta_pred[ta]) for ta in valid_groups)
    n_groups = len(valid_groups)
    device = logits.device

    y_pred = logits.new_full((n_groups, max_len), padded_value)
    y_true = labels.new_full((n_groups, max_len), padded_value)
    for r, ta in enumerate(valid_groups):
        slate_pred = torch.stack(by_ta_pred[ta])
        slate_true = torch.stack(by_ta_true[ta])
        y_pred[r, : slate_pred.shape[0]] = slate_pred
        y_true[r, : slate_true.shape[0]] = slate_true
    return y_pred, y_true


def _make_loss_fn(cfg, sigma=None, ndcg_k=None, ta_by_disease_idx=None, primary_tas=None):
    """Resolve `cfg.train.lambdarank.impl` into a callable (logits, labels) -> scalar loss.

    Supported impls (the in-house lambdarank_loss has been retired — it was a
    less-tested implementation of the same flat-slate math allRank provides):
      * "allrank" (default): vendored allRank lambdaLoss treating the whole
        training batch as one slate of shape [1, B].
      * "allrank_grouped": vendored allRank lambdaLoss with one slate per
        primary TA — items replicated across each of their primary TAs,
        padded with -1. Optimises per-TA ranking instead of one global
        ranking. Requires the TA mapping to be loaded.

    Optional `sigma` / `ndcg_k` overrides let callers (e.g. the Optuna tuner)
    inject per-trial values without mutating cfg.

    Both impls also accept:
      * cfg.train.lambdarank.weighing_scheme (default "lambdaRank_scheme")
      * cfg.train.lambdarank.reduction       (default "sum")
      * cfg.train.lambdarank.reduction_log   (default "binary")
    """
    lr_cfg = cfg.train.lambdarank
    impl = str(lr_cfg.get("impl", "allrank")).lower()
    if sigma is None:
        sigma = float(lr_cfg.get("sigma", 1.0))
    sigma = float(sigma)
    # The loss truncation is fixed at NDCG@50 — it is the operating point we
    # report and select on. Config may still override but the project default
    # is 50, not the historical 100.
    if ndcg_k is None:
        ndcg_k = lr_cfg.get("ndcg_k", 50)
    if ndcg_k is not None:
        ndcg_k = int(ndcg_k)

    if impl == "allrank":
        weighing_scheme = str(lr_cfg.get("weighing_scheme", "lambdaRank_scheme"))
        reduction = str(lr_cfg.get("reduction", "sum"))
        reduction_log = str(lr_cfg.get("reduction_log", "binary"))
        def loss_fn(logits, labels, disease_idx=None):
            y_pred = logits.view(1, -1)
            y_true = labels.view(1, -1)
            return lambdaLoss(
                y_pred, y_true,
                sigma=sigma, k=ndcg_k,
                weighing_scheme=weighing_scheme,
                reduction=reduction,
                reduction_log=reduction_log,
            )
        desc = (
            f"allRank lambdaLoss flat (sigma={sigma}, k={ndcg_k}, "
            f"weighing_scheme={weighing_scheme}, reduction={reduction})"
        )

    elif impl == "allrank_grouped":
        if ta_by_disease_idx is None or primary_tas is None:
            raise ValueError(
                "lambdarank.impl='allrank_grouped' requires the TA mapping to be "
                "loaded (advancement_data/features/therapeutic_areas.parquet + "
                "primary_therapeutic_areas.json). Loading appears to have failed."
            )
        weighing_scheme = str(lr_cfg.get("weighing_scheme", "lambdaRank_scheme"))
        reduction = str(lr_cfg.get("reduction", "sum"))
        reduction_log = str(lr_cfg.get("reduction_log", "binary"))
        primary_set = set(primary_tas)
        def loss_fn(logits, labels, disease_idx=None):
            if disease_idx is None:
                # No grouping info available (e.g. tuner forgot to thread it
                # through). Fall back to flat-slate behaviour with a warning
                # the first time.
                y_pred = logits.view(1, -1)
                y_true = labels.view(1, -1)
            else:
                ta_groups_per_item = [
                    ta_by_disease_idx.get(int(d), []) for d in disease_idx.tolist()
                ]
                y_pred, y_true = _build_grouped_slates(
                    logits, labels, ta_groups_per_item, primary_set,
                )
                if y_pred is None:
                    # No valid TA group in this batch (rare; tiny batch or
                    # all positives/negatives in same TA). Skip step.
                    return 0.0 * logits.sum()
            return lambdaLoss(
                y_pred, y_true,
                sigma=sigma, k=ndcg_k,
                weighing_scheme=weighing_scheme,
                reduction=reduction,
                reduction_log=reduction_log,
            )
        desc = (
            f"allRank lambdaLoss grouped-by-TA (sigma={sigma}, k={ndcg_k}, "
            f"weighing_scheme={weighing_scheme}, reduction={reduction})"
        )

    else:
        raise ValueError(
            f"Unknown lambdarank.impl: {impl!r}. Expected 'allrank' or 'allrank_grouped'."
        )

    return loss_fn, desc


def _load_disease_ta_map(
    ta_parquet_path: str,
    primary_tas_json_path: str,
    disease_mapping: dict,
):
    """Build a per-disease-index list of primary TA names plus the primary set.

    Returns
    -------
    ta_by_disease_idx : dict[int, list[str]]
        Maps internal disease node index → list of primary TA names.
    primary_tas : list[str]
        Primary TA whitelist (with the synthetic "all" entry dropped).
    """
    import json
    import pandas as pd

    ta_df = pd.read_parquet(ta_parquet_path)
    with open(primary_tas_json_path) as f:
        primary_tas_raw = json.load(f)
    primary_tas = [t for t in primary_tas_raw if t != "all"]
    primary_set = set(primary_tas)

    ta_df = ta_df[ta_df["therapeutic_area_name"].isin(primary_set)]
    grouped = (
        ta_df.groupby("disease_id")["therapeutic_area_name"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    ta_by_disease_idx: dict = {}
    for disease_id, idx in disease_mapping.items():
        tas = grouped.get(disease_id)
        if tas:
            ta_by_disease_idx[int(idx)] = tas
    return ta_by_disease_idx, primary_tas


def run_epoch_lambdarank(model, loader, optimizer, device, edge_feat_cols, loss_fn, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[ADV_ETYPE].edge_label_index,
                src_type="target",
                dst_type="disease",
                edge_time_dict=edge_time_dict,
                edge_feat_dict={
                    et: batch[et].edge_attr[:, edge_feat_cols]
                    for et in batch.edge_types
                    if et != ADV_ETYPE and hasattr(batch[et], 'edge_attr')
                    and batch[et].edge_attr is not None
                },
                edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
            )
            labels = batch[ADV_ETYPE].edge_label.float()
            logits = out
            disease_local = batch[ADV_ETYPE].edge_label_index[1]
            disease_idx = batch["disease"].n_id[disease_local]
            loss = loss_fn(logits, labels, disease_idx=disease_idx)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_lambdarank(
    model, loader, device, edge_feat_cols, loss_fn,
    ta_by_disease_idx=None, primary_tas=None,
):
    model.eval()
    all_logits, all_labels, all_disease_idx = [], [], []

    for batch in loader:
        batch = batch.to(device)
        edge_time_dict = _build_edge_time_dict(batch, ADV_ETYPE)
        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target",
            dst_type="disease",
            edge_time_dict=edge_time_dict,
            edge_feat_dict={
                et: batch[et].edge_attr[:, edge_feat_cols]
                for et in batch.edge_types
                if et != ADV_ETYPE and hasattr(batch[et], 'edge_attr')
                and batch[et].edge_attr is not None
            },
            edge_label_time=getattr(batch[ADV_ETYPE], "edge_label_time", None),
        )
        all_logits.append(out.cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())
        if ta_by_disease_idx is not None:
            disease_local = batch[ADV_ETYPE].edge_label_index[1]
            disease_global = batch["disease"].n_id[disease_local]
            all_disease_idx.append(disease_global.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels).float()

    val_disease_idx = torch.cat(all_disease_idx) if all_disease_idx else None
    val_loss = loss_fn(logits_t, labels_t, disease_idx=val_disease_idx).item()

    logits = logits_t.numpy()
    labels = (labels_t > 0).numpy().astype(int)
    nan_mask = np.isnan(logits)
    if nan_mask.any():
        print(f"WARNING: {nan_mask.sum()} NaN logits detected, dropping them.")
        logits = logits[~nan_mask]
        labels = labels[~nan_mask]
    scores = expit(logits)

    metrics = compute_metrics(labels, scores)
    metrics["val_loss"] = val_loss

    scores_t = torch.from_numpy(scores)
    labels_bin_t = torch.from_numpy(labels).float()
    for kk in (10, 30, 50, 100):
        metrics[f"ndcg@{kk}"] = ndcg_at_k(scores_t, labels_bin_t, kk)

    if ta_by_disease_idx is not None and all_disease_idx:
        disease_idx_t = torch.cat(all_disease_idx)
        if nan_mask.any():
            disease_idx_t = disease_idx_t[~torch.from_numpy(nan_mask)]
        ta_per_item = [
            ta_by_disease_idx.get(int(d), [])
            for d in disease_idx_t.tolist()
        ]
        for kk in (10, 30, 50, 100):
            metrics[f"ndcg_ta_mean@{kk}"] = ndcg_ta_mean_at_k(
                scores_t, labels_bin_t, ta_per_item, kk,
                primary_tas=primary_tas,
            )
            metrics[f"rr_ta_mean@{kk}"] = rr_ta_mean_at_k(
                scores_t, labels_bin_t, ta_per_item, kk,
                primary_tas=primary_tas,
            )

    return metrics


def main(cfg):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    tee = TeeLogger(output_dir / "train.log")
    sys.stdout = tee

    wandb.init(
        project="advancement_lambdarank",
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(output_dir),
    )

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

    train_mask, val_mask, test_mask, cutoff_year = split_advancement_edges(data)
    print(f"  Cutoff year: {cutoff_year}")
    print(f"  Train edges: {train_mask.sum().item()}")
    print(f"  Val   edges: {val_mask.sum().item()}")
    print(f"  Test  edges: {test_mask.sum().item()}")

    edge_index = data[ADV_ETYPE].edge_index
    edge_attr  = data[ADV_ETYPE].edge_attr
    edge_time  = data[ADV_ETYPE].edge_time

    train_labels_all = edge_attr[train_mask, 0]
    n_pos = train_labels_all.sum().item()
    n_neg = len(train_labels_all) - n_pos
    print(f"Train label balance: n_pos={int(n_pos)}, n_neg={int(n_neg)}")
    n_pos_val = edge_attr[val_mask, 0].sum().item()
    n_neg_val = len(edge_attr[val_mask, 0]) - n_pos_val
    print(f"Val label balance: n_pos={int(n_pos_val)}, n_neg={int(n_neg_val)}")
    n_pos_test = edge_attr[test_mask, 0].sum().item()
    n_neg_test = len(edge_attr[test_mask, 0]) - n_pos_test
    print(f"Test label balance: n_pos={int(n_pos_test)}, n_neg={int(n_neg_test)}")


    print("Building context graph...")
    context = build_context_graph(data)

    use_recency = bool(cfg.model.get("use_recency", False))
    time_dim = int(cfg.model.get("time_dim", 0))
    if use_recency:
        train_times = edge_time[train_mask].float()
        t_min_val = float(train_times.min().item())
        t_max_val = float(train_times.max().item())
        print(f"Recency encoder: time_dim={time_dim}, t_min={t_min_val}, t_max={t_max_val}")
    else:
        t_min_val, t_max_val = 0.0, 1.0

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

    _edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    repo_root = Path(__file__).resolve().parents[1]
    ta_parquet_path = str(cfg.data.get(
        "ta_parquet",
        repo_root / "advancement_data/features/therapeutic_areas.parquet",
    ))
    primary_tas_json_path = str(cfg.data.get(
        "primary_tas_json",
        repo_root / "advancement_data/results/primary_therapeutic_areas.json",
    ))
    ta_by_disease_idx, primary_tas = None, None
    if Path(ta_parquet_path).exists() and Path(primary_tas_json_path).exists():
        mappings_for_ta = torch.load(cfg.data.mappings_file, weights_only=False)
        disease_mapping = mappings_for_ta["node_mapping"]["disease"]
        ta_by_disease_idx, primary_tas = _load_disease_ta_map(
            ta_parquet_path, primary_tas_json_path, disease_mapping
        )
        print(
            f"TA-grouped NDCG enabled: {len(ta_by_disease_idx)} diseases mapped, "
            f"{len(primary_tas)} primary TAs."
        )
    else:
        print(
            f"TA-grouped NDCG disabled (missing {ta_parquet_path} or "
            f"{primary_tas_json_path})."
        )

    loss_fn, loss_desc = _make_loss_fn(
        cfg,
        ta_by_disease_idx=ta_by_disease_idx,
        primary_tas=primary_tas,
    )
    print(f"LambdaRank loss: {loss_desc}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.train.get("cosine_t_max", cfg.train.num_epochs)),
        eta_min=cfg.train.get("eta_min", 1e-6),
    )

    num_neighbors = list(cfg.train.num_neighbors)

    train_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, train_mask]),
        edge_label=edge_attr[train_mask, 0],
        edge_label_time=edge_time[train_mask],
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, val_mask]),
        edge_label=edge_attr[val_mask, 0],
        edge_label_time=edge_time[val_mask],
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    epoch_rows = []
    ckpt_path = output_dir / "best_model.pt"
    es_cfg = cfg.train.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 10)) if es_enabled else int(1e9)
    es_metric = str(es_cfg.get("metric", "ndcg@50"))
    print(f"Early stopping on val/{es_metric}, patience={patience}")

    test_edge_index = edge_index[:, test_mask]
    test_edge_labels = edge_attr[test_mask, 0]
    test_edge_times  = edge_time[test_mask]

    test_ta_per_item = None
    if ta_by_disease_idx is not None:
        test_disease_idx = test_edge_index[1].tolist()
        test_ta_per_item = [
            ta_by_disease_idx.get(int(d), []) for d in test_disease_idx
        ]

    best_val = -1.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss = run_epoch_lambdarank(
            model, train_loader, optimizer, device,
            edge_feat_cols=_edge_feat_cols, loss_fn=loss_fn, train=True,
        )
        scheduler.step()

        val_metrics = evaluate_lambdarank(
            model, val_loader, device,
            edge_feat_cols=_edge_feat_cols, loss_fn=loss_fn,
            ta_by_disease_idx=ta_by_disease_idx, primary_tas=primary_tas,
        )
        val_score = val_metrics.get(es_metric, float("nan"))
        if np.isnan(val_score):
            val_score = -1.0

        # Per-epoch test rr@K tracking (diagnostic only; does not affect ES).
        test_scores_ep, test_labels_ep = predict_test(
            model, context,
            edge_index=test_edge_index,
            edge_labels=test_edge_labels,
            edge_times=test_edge_times,
            num_neighbors=num_neighbors,
            batch_size=cfg.train.batch_size,
            device=device,
            edge_feat_cols=_edge_feat_cols,
        )
        test_metrics_ep = compute_metrics(test_labels_ep, test_scores_ep)
        test_rr = {
            "rr@10":  float(test_metrics_ep["rr@10"]),
            "rr@50":  float(test_metrics_ep["rr@50"]),
            "rr@100": float(test_metrics_ep["rr@100"]),
        }
        if test_ta_per_item is not None:
            test_scores_t = torch.from_numpy(test_scores_ep)
            test_labels_t = torch.from_numpy(test_labels_ep).float()
            for kk in (10, 30, 50, 100):
                test_rr[f"ndcg_ta_mean@{kk}"] = ndcg_ta_mean_at_k(
                    test_scores_t, test_labels_t, test_ta_per_item, kk,
                    primary_tas=primary_tas,
                )
                test_rr[f"rr_ta_mean@{kk}"] = rr_ta_mean_at_k(
                    test_scores_t, test_labels_t, test_ta_per_item, kk,
                    primary_tas=primary_tas,
                )

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_rr.items()})
        epoch_rows.append(row)

        wandb.log(
            {"train/loss": train_loss}
            | {f"val/{k}": v for k, v in val_metrics.items()}
            | {f"test/{k}": v for k, v in test_rr.items()},
            step=epoch,
        )
        ta_str = ""
        if "ndcg_ta_mean@50" in val_metrics:
            ta_str = (
                f"| val ndcg_ta_mean@10/50: "
                f"{val_metrics['ndcg_ta_mean@10']:.3f}/{val_metrics['ndcg_ta_mean@50']:.3f} "
                f"| val rr_ta_mean@10/50: "
                f"{val_metrics['rr_ta_mean@10']:.3f}/{val_metrics['rr_ta_mean@50']:.3f} "
                f"| test rr_ta_mean@10/50: "
                f"{test_rr['rr_ta_mean@10']:.3f}/{test_rr['rr_ta_mean@50']:.3f} "
            )
        print(
            f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} "
            f"| val ndcg@10/50/100: {val_metrics['ndcg@10']:.3f}/{val_metrics['ndcg@50']:.3f}/{val_metrics['ndcg@100']:.3f} "
            f"{ta_str}"
            f"| val rr@10/50/100: {val_metrics['rr@10']:.3f}/{val_metrics['rr@50']:.3f}/{val_metrics['rr@100']:.3f} "
            f"| test rr@10/50/100: {test_rr['rr@10']:.3f}/{test_rr['rr@50']:.3f}/{test_rr['rr@100']:.3f}"
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
    print(f"Loading best checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    print(f"Epoch metrics saved to {output_dir / 'epoch_metrics.csv'}")

    test_scores, test_labels = predict_test(
        model, context,
        edge_index=test_edge_index,
        edge_labels=test_edge_labels,
        edge_times=test_edge_times,
        num_neighbors=num_neighbors,
        batch_size=cfg.train.batch_size,
        device=device,
        edge_feat_cols=_edge_feat_cols,
    )
    test_metrics = compute_metrics(test_labels, test_scores)

    scores_t = torch.from_numpy(test_scores)
    labels_t = torch.from_numpy(test_labels).float()
    for kk in (10, 30, 50, 100):
        test_metrics[f"ndcg@{kk}"] = ndcg_at_k(scores_t, labels_t, kk)

    if test_ta_per_item is not None:
        for kk in (10, 30, 50, 100):
            test_metrics[f"ndcg_ta_mean@{kk}"] = ndcg_ta_mean_at_k(
                scores_t, labels_t, test_ta_per_item, kk,
                primary_tas=primary_tas,
            )
            test_metrics[f"rr_ta_mean@{kk}"] = rr_ta_mean_at_k(
                scores_t, labels_t, test_ta_per_item, kk,
                primary_tas=primary_tas,
            )

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    print(
        f"\nTest | roc_auc: {test_metrics['roc_auc']:.4f} "
        f"| ap: {test_metrics['average_precision']:.4f} "
        f"| ndcg@100: {test_metrics['ndcg@100']:.4f} "
        f"| rr@100: {test_metrics['rr@100']:.4f} "
        f"| p@10: {test_metrics['precision@10']:.4f}"
    )

    results = {
        "train_edges": int(train_mask.sum().item()),
        "val_edges":   int(val_mask.sum().item()),
        "test_edges":  int(test_mask.sum().item()),
        "best_epoch":  int(best_epoch),
        f"best_val_{es_metric}": float(best_val),
        "test": {f"test_{k}": float(v) for k, v in test_metrics.items()},
    }
    OmegaConf.save(OmegaConf.create(results), output_dir / "results.yaml")
    print(f"Results saved to {output_dir / 'results.yaml'}")

    mappings    = torch.load(cfg.data.mappings_file, weights_only=False)
    inv_target  = {v: k for k, v in mappings["node_mapping"]["target"].items()}
    inv_disease = {v: k for k, v in mappings["node_mapping"]["disease"].items()}

    eval_cfg = cfg.get("eval", None)
    prosp_cfg = eval_cfg.get("prospective", None) if eval_cfg is not None else None
    if prosp_cfg is not None and list(prosp_cfg.get("diseases", []) or []):
        from src.eval.prospective import run_prospective_eval
        run_prospective_eval(model, data, context, mappings, cfg, output_dir)

    wandb.finish()

    src_ids = [inv_target[i]  for i in test_edge_index[0].tolist()]
    dst_ids = [inv_disease[i] for i in test_edge_index[1].tolist()]

    pred_df = pd.DataFrame({
        "target_id":  src_ids,
        "disease_id": dst_ids,
        "score":      test_scores,
        "label":      test_labels.astype(int),
    })
    pred_path = output_dir / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"Test predictions saved to {pred_path}")
    print(f"  Positive pairs: {int(test_labels.sum())} | Negative pairs: {int((test_labels == 0).sum())}")

    sys.stdout = tee._terminal
    tee.close()
    print(f"Log saved to {output_dir / 'train.log'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/experiments/advancement_lambdarank.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)

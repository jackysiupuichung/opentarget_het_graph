#!/usr/bin/env python3
"""Train HGT for advancement link prediction with LambdaRank loss.

Mirrors src/train_advancement_hgt.py but swaps the pointwise BCE/focal
objective for a batch-level LambdaRank loss and uses NDCG@K for early
stopping. Graph construction, splits, model, and sampling are identical so
runs can be compared against the BCE baseline.
"""

import os
import sys
import random
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
from src.benchmark.metrics import ndcg_at_k, ndcg_ta_mean_at_k, rs_ta_mean_at_k, rs_ta_median_at_k, _rs_ta_values_at_k
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
    group_all_tas: bool = False,
):
    """Build a per-disease-index list of TA names plus the TA whitelist.

    By default restricts to the primary TAs (the 13 in primary_therapeutic_areas
    .json). With ``group_all_tas=True`` the map and whitelist instead cover EVERY
    therapeutic area present in the parquet, except the synthetic "all" catch-all
    (which spans ~all diseases and would collapse grouping into one giant slate).

    Returns
    -------
    ta_by_disease_idx : dict[int, list[str]]
        Maps internal disease node index → list of TA names (primary-only, or all
        real TAs when group_all_tas).
    ta_whitelist : list[str]
        The TA names used for grouping (with the synthetic "all" entry dropped).
    """
    import json
    import pandas as pd

    ta_df = pd.read_parquet(ta_parquet_path)
    # The EVAL whitelist is ALWAYS the 13 primary TAs (so reported RS_ta metrics
    # stay comparable across runs regardless of the grouping scope).
    with open(primary_tas_json_path) as f:
        primary_tas_raw = json.load(f)
    eval_primary_tas = [t for t in primary_tas_raw if t != "all"]

    if group_all_tas:
        # GROUPING covers every real TA in the parquet except the synthetic "all"
        # catch-all (which spans ~all diseases and would collapse into one slate).
        group_tas = sorted(
            t for t in ta_df["therapeutic_area_name"].unique() if t != "all"
        )
    else:
        group_tas = list(eval_primary_tas)
    group_set = set(group_tas)

    ta_df = ta_df[ta_df["therapeutic_area_name"].isin(group_set)]
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
    return ta_by_disease_idx, group_tas, eval_primary_tas


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
            metrics[f"rs_ta_mean@{kk}"] = rs_ta_mean_at_k(
                scores_t, labels_bin_t, ta_per_item, kk,
                primary_tas=primary_tas,
            )
            metrics[f"rs_ta_median@{kk}"] = rs_ta_median_at_k(
                scores_t, labels_bin_t, ta_per_item, kk,
                primary_tas=primary_tas,
            )

    return metrics


def main(cfg):
    seed = cfg.get("seed", 42)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    from src.data.temporal_loader import load_event_graph, build_num_neighbors
    to_undirected = bool(cfg.data.get("undirected", False))
    print(f"Loading graph from {cfg.data.graph_file} (undirected={to_undirected})")
    data = load_event_graph(cfg.data.graph_file, to_undirected=to_undirected)

    _data_cfg = cfg.get("data", {})
    train_mask, val_mask, test_mask, cutoff_year = split_advancement_edges(
        data,
        cutoff_year=int(_data_cfg.get("train_cutoff_year", 2010)),
        val_min_year=_data_cfg.get("val_min_year", None),
        val_max_year=_data_cfg.get("val_max_year", None),
        random_val_frac=_data_cfg.get("random_val_frac", None),
        random_seed=int(_data_cfg.get("random_seed", cfg.get("seed", 42))),
    )
    print(f"  Cutoff year: {cutoff_year}")
    if _data_cfg.get("random_val_frac") is not None:
        print(f"  RANDOM train/val split: val_frac={_data_cfg.get('random_val_frac')} "
              f"(test stays temporal >=2016)")
    elif _data_cfg.get("val_min_year") is not None or _data_cfg.get("val_max_year") is not None:
        print(f"  Val window:  [{_data_cfg.get('val_min_year','-')}, {_data_cfg.get('val_max_year','-')}]")
    print(f"  Train edges: {train_mask.sum().item()}")
    print(f"  Val   edges: {val_mask.sum().item()}")
    print(f"  Test  edges: {test_mask.sum().item()}")

    edge_index = data[ADV_ETYPE].edge_index
    edge_attr  = data[ADV_ETYPE].edge_attr
    edge_time  = data[ADV_ETYPE].edge_time

    # Temporal masking for FUTURE-link prediction: the advancement edge_time is
    # the transition (outcome) year; a pair must be scored on evidence strictly
    # BEFORE that year. PyG's LinkNeighborLoader keeps context edges with
    # edge_time <= edge_label_time (inclusive of the transition year), so we
    # pass (edge_time - 1) as the label time to get strict `<` — the standard
    # temporal-LP convention (t < target_time). See memory: masking_strict_before.
    seed_time = edge_time - 1

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
        decoder_kind=str(cfg.model.get("decoder_kind", "mlp")),
        decoder_dropout=float(cfg.model.get("decoder_dropout", cfg.model.get("dropout", 0.1))),
        latest_edge_only=cfg.model.get("latest_edge_only", False),
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
    ta_by_disease_idx, group_tas, primary_tas = None, None, None
    _group_all_tas = bool(cfg.train.get("lambdarank", {}).get("group_all_tas", False))
    if Path(ta_parquet_path).exists() and Path(primary_tas_json_path).exists():
        mappings_for_ta = torch.load(cfg.data.mappings_file, weights_only=False)
        disease_mapping = mappings_for_ta["node_mapping"]["disease"]
        ta_by_disease_idx, group_tas, primary_tas = _load_disease_ta_map(
            ta_parquet_path, primary_tas_json_path, disease_mapping,
            group_all_tas=_group_all_tas,
        )
        print(
            f"TA-grouped NDCG enabled: {len(ta_by_disease_idx)} diseases mapped, "
            f"grouping over {len(group_tas)} TAs (group_all_tas={_group_all_tas}), "
            f"eval on {len(primary_tas)} primary TAs."
        )
    else:
        print(
            f"TA-grouped NDCG disabled (missing {ta_parquet_path} or "
            f"{primary_tas_json_path})."
        )

    loss_fn, loss_desc = _make_loss_fn(
        cfg,
        ta_by_disease_idx=ta_by_disease_idx,
        primary_tas=group_tas,  # grouped loss slates over the grouping whitelist
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

    # Type-aware neighbor budget (see note/neighbor_sampler_type_aware_vs_temporal_loader.md).
    # strategy="off" (default) reproduces the flat type-blind list exactly; other
    # strategies cap dominant edge types (literature) and boost rare predictive
    # ones (clinical_trial_*, genetic_association). PyG samples the dict natively.
    _nn_base = list(cfg.train.num_neighbors)
    _nn_strategy = str(cfg.train.get("num_neighbors_strategy", "off"))
    num_neighbors = build_num_neighbors(
        context,
        base=_nn_base,
        strategy=_nn_strategy,
        overrides=dict(cfg.train.get("num_neighbors_by_type", {}) or {}),
        cap_relations=list(cfg.train.get("num_neighbors_cap", []) or []),
        boost_relations=list(cfg.train.get("num_neighbors_boost", []) or []),
        cap_value=int(cfg.train.get("num_neighbors_cap_value", 2)),
        boost_value=int(cfg.train.get("num_neighbors_boost_value", 40)),
    )
    if _nn_strategy != "off":
        _n_types = len(num_neighbors) if isinstance(num_neighbors, dict) else 0
        print(f"  Type-aware sampling: strategy={_nn_strategy} over {_n_types} edge types")

    # PyG neighbor sampling is CPU-bound; without workers it runs single-process
    # on the main core and starves the GPU. Parallelise with num_workers (set to
    # the SLURM core allocation by default) so extra --cpus-per-gpu actually help.
    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_GPU", 0)) or 0
    # leave one core for the main/training process; cap to avoid oversubscription
    _n_workers = int(cfg.train.get("num_workers", max(_slurm_cpus - 1, 0)))
    _loader_kw = {}
    if _n_workers > 0:
        _loader_kw = {"num_workers": _n_workers, "persistent_workers": True}

    train_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, train_mask]),
        edge_label=edge_attr[train_mask, 0],
        edge_label_time=seed_time[train_mask],
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=cfg.train.batch_size,
        shuffle=True,
        **_loader_kw,
    )
    val_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, val_mask]),
        edge_label=edge_attr[val_mask, 0],
        edge_label_time=seed_time[val_mask],
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=cfg.train.batch_size,
        shuffle=False,
        **_loader_kw,
    )

    epoch_rows = []
    ckpt_path = output_dir / "best_model.pt"
    es_cfg = cfg.train.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 10)) if es_enabled else int(1e9)
    es_metric = str(es_cfg.get("metric", "ndcg@50"))
    es_mode = str(es_cfg.get("mode", "max")).lower()
    assert es_mode in ("max", "min"), f"early_stopping.mode must be 'max' or 'min', got {es_mode}"
    print(f"Early stopping on val/{es_metric} ({es_mode}), patience={patience}")

    test_edge_index = edge_index[:, test_mask]
    test_edge_labels = edge_attr[test_mask, 0]
    test_edge_times  = seed_time[test_mask]   # strict `<`: score on evidence before the transition year

    test_ta_per_item = None
    if ta_by_disease_idx is not None:
        test_disease_idx = test_edge_index[1].tolist()
        test_ta_per_item = [
            ta_by_disease_idx.get(int(d), []) for d in test_disease_idx
        ]

    best_val = float("-inf") if es_mode == "max" else float("inf")
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
            val_score = float("-inf") if es_mode == "max" else float("inf")

        # Per-epoch test rs@K tracking (diagnostic only; does not affect ES).
        # Gated by train.eval_every (default 1 = every epoch). Setting it higher
        # skips the expensive predict_test + TA-grouped metrics on most epochs
        # for a faster wall-clock (always evaluated on the final epoch).
        _eval_every = int(cfg.train.get("eval_every", 1))
        _do_test_eval = (epoch % _eval_every == 0) or (epoch == cfg.train.num_epochs)
        test_rs = {}
        if _do_test_eval:
            test_scores_ep, test_labels_ep = predict_test(
                model, context,
                edge_index=test_edge_index,
                edge_labels=test_edge_labels,
                edge_times=test_edge_times,
                num_neighbors=num_neighbors,
                batch_size=cfg.train.batch_size,
                device=device,
                edge_feat_cols=_edge_feat_cols,
                num_workers=_n_workers,
            )
            test_metrics_ep = compute_metrics(test_labels_ep, test_scores_ep)
            test_rs = {
                "rs@10":  float(test_metrics_ep["rs@10"]),
                "rs@50":  float(test_metrics_ep["rs@50"]),
                "rs@100": float(test_metrics_ep["rs@100"]),
            }

            # Optional diagnostic: dump per-epoch top-K test pairs so we can see
            # whether the model's top-ranked predictions are stable across epochs
            # or whether each epoch's rs@K is hitting a different 9/10 positives.
            # Enable with: SAVE_PER_EPOCH_TOPK=1 (or set to a number = K).
            _topk_env = os.environ.get("SAVE_PER_EPOCH_TOPK")
            if _topk_env:
                _k = int(_topk_env) if _topk_env.isdigit() else 100
                # argsort returns ascending; reverse with [::-1] gives a
                # negative-stride view that torch indexing rejects. Materialise
                # the slice with .copy() so it's a contiguous int64 array.
                _topk_order = np.argsort(test_scores_ep)[::-1][:_k].copy()
                _topk_t = torch.as_tensor(_topk_order, dtype=torch.long)
                _topk_df = pd.DataFrame({
                    "epoch": epoch,
                    "rank": np.arange(_k),
                    "src_idx": test_edge_index[0, _topk_t].cpu().numpy(),
                    "dst_idx": test_edge_index[1, _topk_t].cpu().numpy(),
                    "score":   test_scores_ep[_topk_order],
                    "label":   test_labels_ep[_topk_order].astype(int),
                })
                _topk_path = output_dir / "per_epoch_topk.parquet"
                if _topk_path.exists():
                    _existing = pd.read_parquet(_topk_path)
                    _topk_df = pd.concat([_existing, _topk_df], ignore_index=True)
                _topk_df.to_parquet(_topk_path, index=False)

            # Optional diagnostic: dump the FULL per-epoch test scores (one row
            # per test pair, in the fixed test-pair order) so the per-TA RS
            # leaderboard can be tracked across epochs post-hoc. Row order is
            # identical every epoch and matches the final test_predictions.parquet,
            # so target/disease/TA are joined by position afterwards — no inverse
            # node map needed here. Enable with: SAVE_PER_EPOCH_PREDS=1.
            if os.environ.get("SAVE_PER_EPOCH_PREDS"):
                _ep_dir = output_dir / "per_epoch_preds"
                _ep_dir.mkdir(exist_ok=True)
                pd.DataFrame({
                    "score": test_scores_ep,
                    "label": test_labels_ep.astype(int),
                }).to_parquet(_ep_dir / f"epoch_{epoch:03d}.parquet", index=False)
            if test_ta_per_item is not None:
                test_scores_t = torch.from_numpy(test_scores_ep)
                test_labels_t = torch.from_numpy(test_labels_ep).float()
                for kk in (10, 30, 50, 100):
                    test_rs[f"ndcg_ta_mean@{kk}"] = ndcg_ta_mean_at_k(
                        test_scores_t, test_labels_t, test_ta_per_item, kk,
                        primary_tas=primary_tas,
                    )
                    test_rs[f"rs_ta_mean@{kk}"] = rs_ta_mean_at_k(
                        test_scores_t, test_labels_t, test_ta_per_item, kk,
                        primary_tas=primary_tas,
                    )
                    test_rs[f"rs_ta_median@{kk}"] = rs_ta_median_at_k(
                        test_scores_t, test_labels_t, test_ta_per_item, kk,
                        primary_tas=primary_tas,
                    )
                    # Breadth: how many TAs qualify (have a defined RS) and how
                    # many have an "actual" (>0) value, per epoch.
                    _ta_vals = _rs_ta_values_at_k(
                        test_scores_t, test_labels_t, test_ta_per_item, kk,
                        primary_tas=primary_tas,
                    )
                    test_rs[f"n_ta_qualify@{kk}"] = len(_ta_vals)
                    test_rs[f"n_ta_nonzero@{kk}"] = int(sum(1 for v in _ta_vals if v > 0))

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_rs.items()})
        epoch_rows.append(row)

        wandb.log(
            {"train/loss": train_loss}
            | {f"val/{k}": v for k, v in val_metrics.items()}
            | {f"test/{k}": v for k, v in test_rs.items()},
            step=epoch,
        )
        ta_str = ""
        if "ndcg_ta_mean@50" in val_metrics:
            ta_str = (
                f"| val ndcg_ta_mean@10/50: "
                f"{val_metrics['ndcg_ta_mean@10']:.3f}/{val_metrics['ndcg_ta_mean@50']:.3f} "
                f"| val rs_ta_mean@10/50: "
                f"{val_metrics['rs_ta_mean@10']:.3f}/{val_metrics['rs_ta_mean@50']:.3f} "
                f"| val rs_ta_median@10/50: "
                f"{val_metrics['rs_ta_median@10']:.3f}/{val_metrics['rs_ta_median@50']:.3f} "
            )
            # test metrics only present on eval epochs (gated by eval_every)
            if "rs_ta_mean@10" in test_rs:
                ta_str += (
                    f"| test rs_ta_mean@10/50: "
                    f"{test_rs['rs_ta_mean@10']:.3f}/{test_rs['rs_ta_mean@50']:.3f} "
                    f"| test rs_ta_median@10/50: "
                    f"{test_rs['rs_ta_median@10']:.3f}/{test_rs['rs_ta_median@50']:.3f} "
                )
        test_str = ""
        if "rs@10" in test_rs:
            test_str = (
                f"| test rs@10/50/100: "
                f"{test_rs['rs@10']:.3f}/{test_rs['rs@50']:.3f}/{test_rs['rs@100']:.3f}"
            )
        print(
            f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} "
            f"| val ndcg@10/50/100: {val_metrics['ndcg@10']:.3f}/{val_metrics['ndcg@50']:.3f}/{val_metrics['ndcg@100']:.3f} "
            f"{ta_str}"
            f"| val rs@10/50/100: {val_metrics['rs@10']:.3f}/{val_metrics['rs@50']:.3f}/{val_metrics['rs@100']:.3f} "
            f"{test_str}"
        )

        improved = (val_score > best_val) if es_mode == "max" else (val_score < best_val)
        if improved:
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
        num_workers=_n_workers,
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
            test_metrics[f"rs_ta_mean@{kk}"] = rs_ta_mean_at_k(
                scores_t, labels_t, test_ta_per_item, kk,
                primary_tas=primary_tas,
            )
            test_metrics[f"rs_ta_median@{kk}"] = rs_ta_median_at_k(
                scores_t, labels_t, test_ta_per_item, kk,
                primary_tas=primary_tas,
            )

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    print(
        f"\nTest | roc_auc: {test_metrics['roc_auc']:.4f} "
        f"| ap: {test_metrics['average_precision']:.4f} "
        f"| ndcg@100: {test_metrics['ndcg@100']:.4f} "
        f"| rs@100: {test_metrics['rs@100']:.4f} "
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

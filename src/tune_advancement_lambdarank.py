#!/usr/bin/env python3
"""W&B sweep hyperparameter tuning for LambdaRank advancement prediction.

Mirrors src/tune_advancement_hgt.py but swaps BCE/focal for the LambdaRank
objective from src/train_advancement_lambdarank.py. Sweep metric is
val/ndcg@10 (the ranking metric selected in the LambdaRank config).

Usage
-----
# Create a sweep and launch agent:
python -m src.tune_advancement_lambdarank \
    --config config/experiments/advancement_lambdarank_tune.yaml \
    --create_sweep

# Launch an additional agent on an existing sweep:
python -m src.tune_advancement_lambdarank \
    --config config/experiments/advancement_lambdarank_tune.yaml \
    --sweep_id <id>
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.temporal_loader import load_event_graph
from src.models.utils import build_model
from torch_geometric.loader import LinkNeighborLoader

from src.train_advancement_hgt import (
    ADV_ETYPE,
    split_advancement_edges,
    build_context_graph,
    TeeLogger,
)
from src.train_advancement_lambdarank import (
    run_epoch_lambdarank,
    evaluate_lambdarank,
)


# ---------------------------------------------------------------------------
# Search-space translation: experiment config → W&B sweep config
# ---------------------------------------------------------------------------

def _spec_to_wandb(name: str, spec) -> dict:
    if isinstance(spec, (list, tuple)):
        return {"values": [None if v is None else v for v in spec]}

    if isinstance(spec, dict):
        low  = spec["low"]
        high = spec["high"]
        log  = spec.get("log", False)
        step = spec.get("step", None)

        if log:
            return {"distribution": "log_uniform_values", "min": low, "max": high}
        if step is not None:
            return {"distribution": "q_uniform", "min": low, "max": high, "q": step}
        if isinstance(low, int) and isinstance(high, int):
            return {"distribution": "int_uniform", "min": low, "max": high}
        return {"distribution": "uniform", "min": low, "max": high}

    raise ValueError(f"Unknown search-space spec for '{name}': {spec!r}")


def build_sweep_config(cfg: DictConfig) -> dict:
    ss = OmegaConf.to_container(cfg.tune.search_space, resolve=True)
    parameters = {name: _spec_to_wandb(name, spec) for name, spec in ss.items()}

    metric_name = cfg.tune.get("metric", "val/ndcg@10")
    return {
        "method": "bayes",
        "metric": {"name": metric_name, "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
            "eta": 3,
        },
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Single sweep agent run
# ---------------------------------------------------------------------------

def run_trial(cfg: DictConfig, device: torch.device,
              context, edge_index, edge_attr, edge_time,
              train_idx: torch.Tensor, val_idx: torch.Tensor,
              output_dir: Path):
    wc = wandb.config

    hidden_dim    = wc.hidden_dim
    num_heads     = wc.num_heads
    num_layers    = wc.num_layers
    dropout       = wc.dropout
    lr            = wc.lr
    weight_decay  = wc.weight_decay
    batch_size    = wc.batch_size
    sigma         = wc.sigma
    ndcg_k        = wc.get("ndcg_k", cfg.train.lambdarank.get("ndcg_k", 100))
    cosine_t_max  = wc.cosine_t_max
    num_neighbors = [wc.num_neighbors_0, wc.num_neighbors_1]

    _edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))
    patience = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else int(1e9)
    es_metric = str(cfg.train.early_stopping.get("metric", "ndcg@10"))
    _log_keys = {"average_precision", "roc_auc",
                 "rr@10", "rr@50", "rr@100",
                 "ndcg@10", "ndcg@30", "ndcg@50", "ndcg@100",
                 "val_loss"}

    run_dir = output_dir / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        model_name=cfg.model.name, data=context,
        hidden_dim=hidden_dim, out_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        use_rte=cfg.model.get("use_rte", False),
        use_edge_features=cfg.model.get("use_edge_features", False),
        edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cosine_t_max), eta_min=cfg.train.get("eta_min", 1e-6),
    )

    train_loader = LinkNeighborLoader(
        data=context, num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, train_idx]),
        edge_label=edge_attr[train_idx, 0],
        edge_label_time=edge_time[train_idx],
        time_attr="edge_time", temporal_strategy="last",
        batch_size=batch_size, shuffle=True,
    )
    val_loader = LinkNeighborLoader(
        data=context, num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, val_idx]),
        edge_label=edge_attr[val_idx, 0],
        edge_label_time=edge_time[val_idx],
        time_attr="edge_time", temporal_strategy="last",
        batch_size=batch_size, shuffle=False,
    )

    best_val  = -1.0
    best_epoch = 1
    patience_ctr = 0

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss  = run_epoch_lambdarank(
            model, train_loader, optimizer, device,
            edge_feat_cols=_edge_feat_cols, sigma=sigma, ndcg_k=int(ndcg_k), train=True,
        )
        val_metrics = evaluate_lambdarank(
            model, val_loader, device,
            edge_feat_cols=_edge_feat_cols, sigma=sigma, ndcg_k=int(ndcg_k),
        )

        val_score = val_metrics.get(es_metric, float("nan"))
        if np.isnan(val_score):
            val_score = -1.0

        scheduler.step()

        wandb.log(
            {"train/loss": train_loss, "epoch": epoch} |
            {f"val/{k}": v for k, v in val_metrics.items() if k in _log_keys},
            step=epoch,
        )

        if val_score > best_val:
            best_val   = val_score
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    wandb.summary[f"best_val_{es_metric}"] = best_val
    wandb.log({f"val/{es_metric}": best_val})
    print(f"  trial best {es_metric}={best_val:.4f}  best_epoch={best_epoch}  "
          f"(train={len(train_idx)}, val={len(val_idx)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W&B sweep tuning for LambdaRank advancement")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--sweep_id",     default=None)
    parser.add_argument("--create_sweep", action="store_true")
    parser.add_argument("--n_trials",     type=int, default=None)
    parser.add_argument("--output_dir",   default=None)
    parser.add_argument("--entity",       default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    assert OmegaConf.select(cfg, "tune") is not None, \
        "Config must contain a 'tune:' section."

    tune_cfg   = cfg.tune
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.train.output_dir) / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "sweep_config_snapshot.yaml")

    tee = TeeLogger(output_dir / "sweep.log")
    sys.stdout = tee

    project = "advancement_lambdarank_tune"
    entity  = args.entity

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_config = build_sweep_config(cfg)
        sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
        print(f"Created sweep: {sweep_id}")
        (output_dir / "sweep_id.txt").write_text(sweep_id)
    else:
        print(f"Joining sweep: {sweep_id}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading graph from {cfg.data.graph_file}")
    data = load_event_graph(cfg.data.graph_file)

    train_mask, val_mask, test_mask, cutoff_year = split_advancement_edges(data)

    edge_index = data[ADV_ETYPE].edge_index
    edge_attr  = data[ADV_ETYPE].edge_attr
    edge_time  = data[ADV_ETYPE].edge_time

    train_idx = train_mask.nonzero(as_tuple=True)[0]
    val_idx   = val_mask.nonzero(as_tuple=True)[0]
    print(f"Cutoff year: {cutoff_year} | train edges: {len(train_idx)} "
          f"| val edges: {len(val_idx)} | test edges: {int(test_mask.sum())}")

    context = build_context_graph(data)

    def _run_trial():
        wandb.init(
            project=project,
            entity=entity,
            group=tune_cfg.study_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        run_trial(
            cfg, device,
            context, edge_index, edge_attr, edge_time,
            train_idx, val_idx,
            output_dir,
        )
        wandb.finish()

    count = args.n_trials or tune_cfg.get("n_trials", 50)
    print(f"\nStarting agent for sweep '{sweep_id}' | max trials={count}")
    wandb.agent(sweep_id, function=_run_trial, project=project, entity=entity, count=count)

    sys.stdout = tee._terminal
    tee.close()
    print(f"Sweep log saved to {output_dir / 'sweep.log'}")


if __name__ == "__main__":
    main()

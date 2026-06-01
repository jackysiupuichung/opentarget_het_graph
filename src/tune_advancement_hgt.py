#!/usr/bin/env python3
"""
W&B sweep-based hyperparameter tuning for clinical trial advancement link prediction.

The sweep config is built automatically from the `tune.search_space` section of
the experiment config YAML and registered with W&B.  Each sweep agent run trains
one trial and logs metrics + best_epoch to W&B.

Search-space spec (in config YAML):
  - list                      → categorical
  - {low, high}               → int_uniform
  - {low, high, log: true}    → log_uniform_values
  - {low, high, step}         → q_uniform (quantised float)

Usage
-----
# 1. Create the sweep and print the sweep ID:
python -m src.tune_advancement_hgt \\
    --config config/experiments/p3_eahgt_both.yaml \\
    --create_sweep

# 2. Launch an agent (one per GPU job):
python -m src.tune_advancement_hgt \\
    --config config/experiments/p3_eahgt_both.yaml \\
    --sweep_id <id>

# Both steps can be combined (create + launch agent) by passing --create_sweep
# without --sweep_id, which is the default mode for the Slurm script.
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
    run_epoch,
    evaluate,
    TeeLogger,
)


# ---------------------------------------------------------------------------
# Search-space translation: experiment config → W&B sweep config
# ---------------------------------------------------------------------------

def _spec_to_wandb(name: str, spec) -> dict:
    """Translate one search-space spec to a W&B sweep parameter dict."""
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

        # integer range
        if isinstance(low, int) and isinstance(high, int):
            return {"distribution": "int_uniform", "min": low, "max": high}

        return {"distribution": "uniform", "min": low, "max": high}

    raise ValueError(f"Unknown search-space spec for '{name}': {spec!r}")


def build_sweep_config(cfg: DictConfig) -> dict:
    """Build a W&B sweep config dict from the experiment config."""
    ss = OmegaConf.to_container(cfg.tune.search_space, resolve=True)
    parameters = {name: _spec_to_wandb(name, spec) for name, spec in ss.items()}

    return {
        "method": "bayes",
        "metric": {"name": "val/rs@100", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 5,
            "eta": 3,
        },
        "parameters": parameters,
    }


# ---------------------------------------------------------------------------
# Single sweep agent run (chronological train/val split)
# ---------------------------------------------------------------------------

def run_trial(cfg: DictConfig, device: torch.device,
              context, edge_index, edge_attr, edge_time,
              train_idx: torch.Tensor, val_idx: torch.Tensor,
              output_dir: Path):
    """One W&B sweep agent call = single chronological train/val fit.

    Trains until val/rs@100 stops improving (early stopping with patience),
    reports the best val/rs@100 as the trial's score.
    """

    wc = wandb.config

    hidden_dim    = wc.hidden_dim
    num_heads     = wc.num_heads
    num_layers    = wc.num_layers
    dropout       = wc.dropout
    lr            = wc.lr
    weight_decay  = wc.weight_decay
    batch_size    = wc.batch_size
    focal_gamma   = wc.focal_gamma
    num_neighbors = [wc.num_neighbors_0, wc.num_neighbors_1]

    _edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))
    patience = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else int(1e9)
    _log_keys = {"average_precision", "average_precision@100", "roc_auc",
                 "rs@10", "rs@50", "rs@100", "val_loss"}

    run_dir = output_dir / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    # pos_weight from train labels
    train_labels = edge_attr[train_idx, 0]
    n_pos = train_labels.sum().item()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

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
        optimizer, T_max=cfg.train.num_epochs, eta_min=cfg.train.get("eta_min", 1e-6),
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

    best_val_rs  = -1.0
    best_epoch   = 1
    patience_ctr = 0

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss  = run_epoch(model, train_loader, optimizer, device, train=True,
                                edge_feat_cols=_edge_feat_cols,
                                pos_weight=pos_weight, focal_gamma=focal_gamma)
        val_metrics = evaluate(model, val_loader, device, edge_feat_cols=_edge_feat_cols)

        val_rs = val_metrics["rs@100"]
        if np.isnan(val_rs):
            val_rs = -1.0

        scheduler.step()

        wandb.log(
            {"train/loss": train_loss, "epoch": epoch} |
            {f"val/{k}": v for k, v in val_metrics.items() if k in _log_keys},
            step=epoch,
        )

        if val_rs > best_val_rs:
            best_val_rs  = val_rs
            best_epoch   = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    wandb.summary["best_val_rs@100"] = best_val_rs
    wandb.log({"val/rs@100": best_val_rs})
    print(f"  trial best rs@100={best_val_rs:.4f}  best_epoch={best_epoch}  "
          f"(train={len(train_idx)}, val={len(val_idx)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W&B sweep tuning for advancement HGT")
    parser.add_argument("--config",       required=True, help="Experiment config YAML (must contain a tune: section)")
    parser.add_argument("--sweep_id",     default=None,  help="Existing W&B sweep ID to join as an agent")
    parser.add_argument("--create_sweep", action="store_true", help="Create a new sweep and launch agent")
    parser.add_argument("--n_trials",     type=int, default=None, help="Max trials for this agent run")
    parser.add_argument("--output_dir",   default=None,  help="Override output directory")
    parser.add_argument("--entity",       default=None,  help="W&B entity (defaults to logged-in user)")
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

    project = "advancement_hgt_tune"
    entity  = args.entity

    # ── Create sweep or use existing ─────────────────────────────────────────
    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_config = build_sweep_config(cfg)
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=project,
            entity=entity,
        )
        print(f"Created sweep: {sweep_id}")
        # Save sweep ID so subsequent agents can join without --create_sweep
        (output_dir / "sweep_id.txt").write_text(sweep_id)
    else:
        print(f"Joining sweep: {sweep_id}")

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load graph (shared across all trials in this agent) ──────────────────
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

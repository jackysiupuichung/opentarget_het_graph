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
import multiprocessing as mp
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
    _make_loss_fn,
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

    metric_name = cfg.tune.get("metric", "val/ndcg@50")
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
    ndcg_k        = wc.get("ndcg_k", cfg.train.lambdarank.get("ndcg_k", 50))
    cosine_t_max  = wc.cosine_t_max
    num_neighbors = [wc.num_neighbors_0, wc.num_neighbors_1]
    # Optional model-specific knob (CompGCN composition op). Pulled from sweep
    # config when present, otherwise from the base config, otherwise None (model default).
    composition   = wc.get("composition", cfg.model.get("composition", None))

    _edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))
    patience = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else int(1e9)
    es_metric = str(cfg.train.early_stopping.get("metric", "ndcg@50"))
    _log_keys = {"average_precision", "roc_auc",
                 "rs@10", "rs@50", "rs@100",
                 "ndcg@10", "ndcg@30", "ndcg@50", "ndcg@100",
                 "val_loss"}

    run_dir = output_dir / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    build_kwargs = dict(
        model_name=cfg.model.name, data=context,
        hidden_dim=hidden_dim, out_dim=hidden_dim,
        num_heads=num_heads, num_layers=num_layers, dropout=dropout,
        use_rte=cfg.model.get("use_rte", False),
        use_edge_features=cfg.model.get("use_edge_features", False),
        edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
    )
    if composition is not None and cfg.model.name == "compgcn":
        build_kwargs["composition"] = composition
    model = build_model(**build_kwargs).to(device)
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

    loss_fn, _ = _make_loss_fn(cfg, sigma=sigma, ndcg_k=int(ndcg_k))
    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss  = run_epoch_lambdarank(
            model, train_loader, optimizer, device,
            edge_feat_cols=_edge_feat_cols, loss_fn=loss_fn, train=True,
        )
        val_metrics = evaluate_lambdarank(
            model, val_loader, device,
            edge_feat_cols=_edge_feat_cols, loss_fn=loss_fn,
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
# Subprocess worker — runs one trial in an isolated CUDA context.
# ---------------------------------------------------------------------------


def _trial_worker(config_path: str, sweep_overrides: dict, run_id: str,
                  output_dir_str: str, queue) -> None:
    """Child process: load graph, build model, train, stream metrics via `queue`.

    Communicates with parent (which owns the W&B run) by pushing tagged tuples
    onto `queue`:
      ("log", {metric_name: value, ..., "__step": int})  — log dict at step
      ("summary", {key: value, ...})                     — set wandb summary key
      ("print", str)                                     — append to parent stdout
      None                                               — sentinel: end of trial
    """
    import os
    import sys
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from pathlib import Path

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.temporal_loader import load_event_graph
    from src.models.utils import build_model
    from torch_geometric.loader import LinkNeighborLoader
    from src.train_advancement_hgt import (
        ADV_ETYPE, split_advancement_edges, build_context_graph,
    )
    from src.train_advancement_lambdarank import (
        run_epoch_lambdarank, evaluate_lambdarank, _make_loss_fn,
        _load_disease_ta_map,
    )

    try:
        cfg = OmegaConf.load(config_path)

        # Determinism: mirror train_advancement_lambdarank.main(). Without
        # this, every sweep trial runs with arbitrary RNG state, so "best
        # trial" rankings reflect random init noise rather than HP effects.
        seed = int(cfg.get("seed", 42))
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and split graph.
        _data_cfg = cfg.data
        data = load_event_graph(cfg.data.graph_file,
                                to_undirected=bool(_data_cfg.get("undirected", False)))
        train_mask, val_mask, _, _ = split_advancement_edges(
            data,
            cutoff_year=int(_data_cfg.get("train_cutoff_year", 2010)),
            val_min_year=_data_cfg.get("val_min_year", None),
            val_max_year=_data_cfg.get("val_max_year", None),
        )
        edge_index = data[ADV_ETYPE].edge_index
        edge_attr  = data[ADV_ETYPE].edge_attr
        edge_time  = data[ADV_ETYPE].edge_time
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        val_idx   = val_mask.nonzero(as_tuple=True)[0]
        context = build_context_graph(data)

        # TA mapping for ndcg_ta_mean@K / rs_ta_mean@K (when configured as the
        # selection / sweep metric, these must be computed; otherwise eval falls
        # back to pooled metrics only).
        repo_root = Path(__file__).resolve().parents[1]
        ta_parquet_path = str(_data_cfg.get(
            "ta_parquet",
            repo_root / "advancement_data/features/therapeutic_areas.parquet",
        ))
        primary_tas_json_path = str(_data_cfg.get(
            "primary_tas_json",
            repo_root / "advancement_data/results/primary_therapeutic_areas.json",
        ))
        ta_by_disease_idx, primary_tas = None, None
        if Path(ta_parquet_path).exists() and Path(primary_tas_json_path).exists():
            mappings_for_ta = torch.load(_data_cfg.mappings_file, weights_only=False)
            disease_mapping = mappings_for_ta["node_mapping"]["disease"]
            ta_by_disease_idx, primary_tas = _load_disease_ta_map(
                ta_parquet_path, primary_tas_json_path, disease_mapping,
            )
            queue.put(("print",
                       f"  TA-grouped NDCG enabled: {len(ta_by_disease_idx)} diseases, "
                       f"{len(primary_tas)} primary TAs"))
        else:
            queue.put(("print",
                       f"  TA-grouped NDCG disabled (missing {ta_parquet_path} or "
                       f"{primary_tas_json_path})"))

        # Resolve sweep-overridden hyperparameters.
        wc = sweep_overrides
        hidden_dim    = int(wc["hidden_dim"])
        num_heads     = int(wc["num_heads"])
        num_layers    = int(wc["num_layers"])
        dropout       = float(wc["dropout"])
        lr            = float(wc["lr"])
        weight_decay  = float(wc["weight_decay"])
        batch_size    = int(wc["batch_size"])
        sigma         = float(wc["sigma"])
        ndcg_k        = int(wc.get("ndcg_k", cfg.train.lambdarank.get("ndcg_k", 50)))
        cosine_t_max  = int(wc["cosine_t_max"])
        num_neighbors = [int(wc["num_neighbors_0"]), int(wc["num_neighbors_1"])]
        composition   = wc.get("composition", cfg.model.get("composition", None))
        decoder_kind    = str(wc.get("decoder_kind", cfg.model.get("decoder_kind", "mlp")))
        decoder_dropout = float(wc.get("decoder_dropout", cfg.model.get("decoder_dropout", dropout)))

        edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))
        patience  = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else int(1e9)
        es_metric = str(cfg.train.early_stopping.get("metric", "ndcg@50"))
        es_mode   = str(cfg.train.early_stopping.get("mode", "max")).lower()
        assert es_mode in ("max", "min"), f"early_stopping.mode must be 'max' or 'min', got {es_mode}"
        log_keys  = {"average_precision", "roc_auc",
                     "rs@10", "rs@50", "rs@100",
                     "ndcg@10", "ndcg@30", "ndcg@50", "ndcg@100",
                     "ndcg_ta_mean@10", "ndcg_ta_mean@30",
                     "ndcg_ta_mean@50", "ndcg_ta_mean@100",
                     "rs_ta_mean@10", "rs_ta_mean@30",
                     "rs_ta_mean@50", "rs_ta_mean@100",
                     "val_loss"}

        # Build model (with optional composition for CompGCN).
        build_kwargs = dict(
            model_name=cfg.model.name, data=context,
            hidden_dim=hidden_dim, out_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers, dropout=dropout,
            use_rte=cfg.model.get("use_rte", False),
            use_edge_features=cfg.model.get("use_edge_features", False),
            edge_feat_dim=cfg.model.get("edge_feat_dim", 2),
            decoder_kind=decoder_kind,
            decoder_dropout=decoder_dropout,
        )
        if composition is not None and cfg.model.name == "compgcn":
            build_kwargs["composition"] = composition

        model = build_model(**build_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cfg.train.get("eta_min", 1e-6),
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

        best_val = float("-inf") if es_mode == "max" else float("inf")
        best_epoch = 1
        patience_ctr = 0

        loss_fn, _ = _make_loss_fn(
            cfg, sigma=sigma, ndcg_k=ndcg_k,
            ta_by_disease_idx=ta_by_disease_idx, primary_tas=primary_tas,
        )
        for epoch in range(1, cfg.train.num_epochs + 1):
            train_loss = run_epoch_lambdarank(
                model, train_loader, optimizer, device,
                edge_feat_cols=edge_feat_cols, loss_fn=loss_fn, train=True,
            )
            val_metrics = evaluate_lambdarank(
                model, val_loader, device,
                edge_feat_cols=edge_feat_cols, loss_fn=loss_fn,
                ta_by_disease_idx=ta_by_disease_idx,
                primary_tas=primary_tas,
            )
            val_score = val_metrics.get(es_metric, float("nan"))
            if np.isnan(val_score):
                val_score = float("-inf") if es_mode == "max" else float("inf")
            scheduler.step()

            log_payload = {"train/loss": train_loss, "epoch": epoch, "__step": epoch}
            log_payload.update({f"val/{k}": float(v) for k, v in val_metrics.items() if k in log_keys})
            queue.put(("log", log_payload))

            improved = (val_score > best_val) if es_mode == "max" else (val_score < best_val)
            if improved:
                best_val   = val_score
                best_epoch = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    queue.put(("print", f"  early stop at epoch {epoch} (best epoch {best_epoch})"))
                    break

        # Guard against -inf / +inf if no epoch ever improved.
        if not np.isfinite(best_val):
            best_val = float("nan")
        queue.put(("summary", {f"best_val_{es_metric}": float(best_val)}))
        queue.put(("log", {f"val/{es_metric}": float(best_val)}))
        queue.put(("print", f"  trial best {es_metric}={best_val:.4f}  best_epoch={best_epoch}  "
                            f"(train={len(train_idx)}, val={len(val_idx)})"))
    except Exception as e:
        import traceback
        queue.put(("print", f"  ⚠️  trial worker exception: {e}\n{traceback.format_exc()}"))
    finally:
        queue.put(None)


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

    # Parent never touches the GPU — each trial does its own CUDA init in a
    # subprocess so memory is fully released between trials.
    print(f"GPU visible: {torch.cuda.is_available()}")

    # Note: the graph is loaded inside each trial subprocess (see _trial_worker).
    # Loading once in the parent and passing tensors to children is unsafe with
    # CUDA: any tensor accessed in the parent gets the parent's CUDA context
    # baked in, which leaks across forked children. Per-child load is the
    # cleanest way to guarantee a fresh CUDA context per trial.

    cfg_dict_static = OmegaConf.to_container(cfg, resolve=True)
    config_path = str(Path(args.config).resolve())

    def _run_trial():
        """Parent-side: init W&B, spawn child for actual training, stream metrics back."""
        wandb.init(
            project=project,
            entity=entity,
            group=tune_cfg.study_name,
            config=cfg_dict_static,
        )
        # Resolve sweep-suggested overrides from the W&B run config.
        wc = dict(wandb.config)
        run_id = wandb.run.id

        # Use spawn to get a fresh Python interpreter + fresh CUDA context.
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(
            target=_trial_worker,
            args=(config_path, dict(wc), run_id, str(output_dir), q),
        )
        proc.start()

        # Drain the queue: child sends per-epoch metrics dicts and a final
        # summary dict; we forward each to wandb in the parent.
        while True:
            msg = q.get()
            if msg is None:
                break
            kind, payload = msg
            if kind == "log":
                step = payload.pop("__step", None)
                if step is None:
                    wandb.log(payload)
                else:
                    wandb.log(payload, step=step)
            elif kind == "summary":
                for k, v in payload.items():
                    wandb.summary[k] = v
            elif kind == "print":
                print(payload)

        proc.join()
        if proc.exitcode != 0:
            print(f"  ⚠️  trial subprocess exited with code {proc.exitcode}")
        wandb.finish()

    count = args.n_trials or tune_cfg.get("n_trials", 50)
    print(f"\nStarting agent for sweep '{sweep_id}' | max trials={count}")
    wandb.agent(sweep_id, function=_run_trial, project=project, entity=entity, count=count)

    sys.stdout = tee._terminal
    tee.close()
    print(f"Sweep log saved to {output_dir / 'sweep.log'}")


if __name__ == "__main__":
    main()

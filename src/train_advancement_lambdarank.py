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
from src.losses.lambdarank import lambdarank_loss
from src.benchmark.metrics import ndcg_at_k
from src.models.utils import build_model


def run_epoch_lambdarank(model, loader, optimizer, device, edge_feat_cols, sigma, ndcg_k, train=True):
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
            )
            labels = batch[ADV_ETYPE].edge_label.float()
            logits = out["logits_exist"].squeeze(-1)
            loss = lambdarank_loss(logits, labels, sigma=sigma, k=ndcg_k)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_lambdarank(model, loader, device, edge_feat_cols, sigma, ndcg_k):
    model.eval()
    all_logits, all_labels = [], []

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
        )
        all_logits.append(out["logits_exist"].squeeze(-1).cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())

    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_labels).float()

    val_loss = lambdarank_loss(logits_t, labels_t, sigma=sigma, k=ndcg_k).item()

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
    print(f"Loading graph from {cfg.data.graph_file}")
    data = load_event_graph(cfg.data.graph_file)

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
    ).to(device)
    print(f"Model: {cfg.model.name}")

    _edge_feat_cols = list(cfg.model.get("edge_feat_cols", [0, 1]))

    sigma = float(cfg.train.lambdarank.get("sigma", 1.0))
    ndcg_k = cfg.train.lambdarank.get("ndcg_k", 100)
    if ndcg_k is not None:
        ndcg_k = int(ndcg_k)
    print(f"LambdaRank: sigma={sigma}, ndcg_k={ndcg_k}")

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
    es_metric = str(es_cfg.get("metric", "ndcg@100"))
    print(f"Early stopping on val/{es_metric}, patience={patience}")

    test_edge_index = edge_index[:, test_mask]
    test_edge_labels = edge_attr[test_mask, 0]
    test_edge_times  = edge_time[test_mask]

    best_val = -1.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss = run_epoch_lambdarank(
            model, train_loader, optimizer, device,
            edge_feat_cols=_edge_feat_cols, sigma=sigma, ndcg_k=ndcg_k, train=True,
        )
        scheduler.step()

        val_metrics = evaluate_lambdarank(
            model, val_loader, device,
            edge_feat_cols=_edge_feat_cols, sigma=sigma, ndcg_k=ndcg_k,
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
        print(
            f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} "
            f"| val ndcg@10/50/100: {val_metrics['ndcg@10']:.3f}/{val_metrics['ndcg@50']:.3f}/{val_metrics['ndcg@100']:.3f} "
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

    wandb.finish()

    mappings    = torch.load(cfg.data.mappings_file, weights_only=False)
    inv_target  = {v: k for k, v in mappings["node_mapping"]["target"].items()}
    inv_disease = {v: k for k, v in mappings["node_mapping"]["disease"].items()}

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

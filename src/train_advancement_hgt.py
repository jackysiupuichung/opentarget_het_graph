#!/usr/bin/env python3
"""
Train HGT for clinical trial advancement link prediction.

- Context graph: all non-advancement edges, full temporal structure (no collapse by default)
- Train/val split: chronological 85/15 on train_dataset rows (transition_year <= 2013)
- Test: original test_dataset rows (transition_year >= 2016)
- Task: binary link prediction (outcome 0/1), BCE loss
- Output: best_model.pt, results.yaml, test_predictions.parquet
"""

import os
import sys
import argparse
from pathlib import Path

# Allow MPS to fall back to CPU for unsupported ops (e.g. scatter_reduce)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, brier_score_loss, log_loss,
    precision_recall_curve,
)
from torch_geometric.loader import LinkNeighborLoader
import wandb


class TeeLogger:
    """Mirrors stdout to both the terminal and a log file."""
    def __init__(self, log_path):
        self._terminal = sys.stdout
        self._log = open(log_path, "a", buffering=1)  # line-buffered

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.temporal_loader import load_event_graph, to_time_agnostic
from src.models.utils import build_model


ADV_ETYPE = ("target", "advancement", "disease")
TRAIN_YEAR_MAX = 2013   # transition years from train_dataset.csv
TEST_YEAR_MIN  = 2016   # transition years from test_dataset.csv


def split_advancement_edges(data, train_quantile=0.85):
    """
    Chronological split of advancement edges.

    Returns
    -------
    train_mask, val_mask, test_mask : BoolTensor[num_adv_edges]
    cutoff_year : int  (85th percentile transition_year among train rows)
    """
    edge_time = data[ADV_ETYPE].edge_time  # [E]

    train_year_mask = edge_time <= TRAIN_YEAR_MAX
    test_year_mask  = edge_time >= TEST_YEAR_MIN

    train_years = edge_time[train_year_mask].float()
    cutoff_year = int(torch.quantile(train_years, train_quantile).item())

    train_mask = train_year_mask & (edge_time <= cutoff_year)
    val_mask   = train_year_mask & (edge_time >  cutoff_year)
    test_mask  = test_year_mask

    return train_mask, val_mask, test_mask, cutoff_year


def build_context_graph(data, collapse: bool = False):
    """Remove advancement edges from the graph.

    Parameters
    ----------
    collapse : bool, default False
        If True, collapse parallel temporal edges into a single static edge
        via to_time_agnostic().  Leave False to retain full temporal structure
        so that LinkNeighborLoader can apply per-query time filtering.
    """
    from torch_geometric.data import HeteroData

    # Manually copy only non-advancement edge types to avoid PyG tracking the
    # empty storage that results from deleting attributes on a clone.
    context = HeteroData()

    for node_type in data.node_types:
        for key, val in data[node_type].items():
            context[node_type][key] = val

    for edge_type in data.edge_types:
        if edge_type == ADV_ETYPE:
            continue
        for key, val in data[edge_type].items():
            context[edge_type][key] = val

    if collapse:
        context = to_time_agnostic(context)

    return context


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[ADV_ETYPE].edge_label_index,
                src_type="target",
                dst_type="disease",
            )
            labels = batch[ADV_ETYPE].edge_label.float()
            logits = out["logits_exist"].squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute the full evaluation metric suite.

    Parameters
    ----------
    labels : int array of shape [N]  (0 / 1)
    scores : float array of shape [N]  (predicted probabilities)
    """
    n_samples   = len(labels)
    n_positives = int(labels.sum())
    balance     = n_positives / n_samples if n_samples > 0 else float("nan")

    if n_positives == 0 or n_positives == n_samples:
        nan = float("nan")
        return {k: nan for k in [
            "n_samples", "n_positives", "balance",
            "precision", "recall", "f1", "mcc",
            "roc_auc", "average_precision", "brier", "balanced_mae", "log_loss",
            "precision@10", "precision@30", "precision@50",
            "average_precision@10", "average_precision@30", "average_precision@50",
            "recall@10", "recall@30", "recall@50",
        ]} | {"n_samples": n_samples, "n_positives": n_positives, "balance": balance}

    preds = (scores >= 0.5).astype(int)

    # Threshold-based
    precision   = precision_score(labels, preds, zero_division=0)
    recall      = recall_score(labels, preds, zero_division=0)
    f1          = f1_score(labels, preds, zero_division=0)
    mcc         = matthews_corrcoef(labels, preds)

    # Ranking / probabilistic
    roc_auc            = roc_auc_score(labels, scores)
    avg_precision      = average_precision_score(labels, scores)
    brier              = brier_score_loss(labels, scores)
    ll                 = log_loss(labels, scores)
    # Balanced MAE: mean |score - label| weighted so pos/neg classes contribute equally
    pos_mae = np.abs(scores[labels == 1] - 1).mean() if n_positives > 0 else float("nan")
    neg_mae = np.abs(scores[labels == 0] - 0).mean() if (n_samples - n_positives) > 0 else float("nan")
    balanced_mae = (pos_mae + neg_mae) / 2

    # Rank-based @K metrics
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]

    def _precision_at_k(k):
        top = labels_sorted[:k]
        return top.sum() / k if k > 0 else float("nan")

    def _recall_at_k(k):
        top = labels_sorted[:k]
        return top.sum() / n_positives if n_positives > 0 else float("nan")

    def _ap_at_k(k):
        top_labels  = labels_sorted[:k]
        top_scores  = scores[order][:k]
        if top_labels.sum() == 0:
            return 0.0
        return average_precision_score(top_labels, top_scores)

    metrics = {
        "n_samples":           n_samples,
        "n_positives":         n_positives,
        "balance":             balance,
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "mcc":                 mcc,
        "roc_auc":             roc_auc,
        "average_precision":   avg_precision,
        "brier":               brier,
        "balanced_mae":        balanced_mae,
        "log_loss":            ll,
        "precision@10":        _precision_at_k(10),
        "precision@30":        _precision_at_k(30),
        "precision@50":        _precision_at_k(50),
        "average_precision@10": _ap_at_k(10),
        "average_precision@30": _ap_at_k(30),
        "average_precision@50": _ap_at_k(50),
        "recall@10":           _recall_at_k(10),
        "recall@30":           _recall_at_k(30),
        "recall@50":           _recall_at_k(50),
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target",
            dst_type="disease",
        )
        all_logits.append(out["logits_exist"].squeeze(-1).cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = (torch.cat(all_labels) > 0).numpy().astype(int)
    scores = expit(logits)

    return compute_metrics(labels, scores)


@torch.no_grad()
def predict_test(model, context, edge_index, edge_labels, edge_times, num_neighbors, batch_size, device):
    """Score test edges using temporally-constrained subgraphs."""
    model.eval()
    loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index),
        edge_label=edge_labels,
        edge_label_time=edge_times,
        time_attr="edge_time",
        temporal_strategy="last",
        batch_size=batch_size,
        shuffle=False,
    )
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(
            batch.x_dict, batch.edge_index_dict,
            batch[ADV_ETYPE].edge_label_index,
            src_type="target", dst_type="disease",
        )
        all_logits.append(out["logits_exist"].squeeze(-1).cpu())
        all_labels.append(batch[ADV_ETYPE].edge_label.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = (torch.cat(all_labels) > 0).numpy().astype(int)
    scores = expit(logits)
    return scores, labels


def main(cfg):
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    tee = TeeLogger(output_dir / "train.log")
    sys.stdout = tee

    wandb.init(
        project="advancement_hgt",
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="offline",
        dir=str(output_dir),
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load graph ──────────────────────────────────────────────────────────
    print(f"Loading graph from {cfg.data.graph_file}")
    data = load_event_graph(cfg.data.graph_file)

    # ── Split advancement edges ──────────────────────────────────────────────
    train_mask, val_mask, test_mask, cutoff_year = split_advancement_edges(data)
    print(f"Cutoff year (85th pct of train rows): {cutoff_year}")
    print(f"  Train edges: {train_mask.sum().item()}")
    print(f"  Val   edges: {val_mask.sum().item()}")
    print(f"  Test  edges: {test_mask.sum().item()}")

    edge_index = data[ADV_ETYPE].edge_index  # [2, E]
    edge_attr  = data[ADV_ETYPE].edge_attr   # [E, 1]
    edge_time  = data[ADV_ETYPE].edge_time   # [E]

    # ── Build static context graph (no advancement edges) ───────────────────
    print("Building context graph...")
    context = build_context_graph(data)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        model_name=cfg.model.name,
        data=context,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)
    print(f"Model: {cfg.model.name} | params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    num_neighbors = list(cfg.train.num_neighbors)

    train_loader = LinkNeighborLoader(
        data=context,
        num_neighbors=num_neighbors,
        edge_label_index=(ADV_ETYPE, edge_index[:, train_mask]),
        edge_label=edge_attr[train_mask, 0],   # 0.0/1.0 from CSV — no synthetic negatives
        edge_label_time=edge_time[train_mask],  # anchor time per query edge
        time_attr="edge_time",                  # filter neighbours to <= anchor time
        temporal_strategy="last",               # sample most recent qualifying neighbours
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

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_auroc  = -1.0
    best_val_metrics = {}
    patience_counter = 0
    patience = cfg.train.early_stopping.patience if cfg.train.early_stopping.enabled else int(1e9)

    metric_cols = [
        "epoch", "split", "train_loss",
        "n_samples", "n_positives", "balance",
        "precision", "recall", "f1", "mcc",
        "roc_auc", "average_precision", "brier", "balanced_mae", "log_loss",
        "precision@10", "precision@30", "precision@50",
        "average_precision@10", "average_precision@30", "average_precision@50",
        "recall@10", "recall@30", "recall@50",
    ]
    epoch_rows = []

    for epoch in range(1, cfg.train.num_epochs + 1):
        train_loss  = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = evaluate(model, val_loader, device)

        row = {"epoch": epoch, "split": "val", "train_loss": train_loss}
        row.update(val_metrics)
        epoch_rows.append(row)

        wandb.log({"train/loss": train_loss} |
                  {f"val/{k}": v for k, v in val_metrics.items()},
                  step=epoch)

        print(
            f"Epoch {epoch:3d} | loss: {train_loss:.4f} "
            f"| roc_auc: {val_metrics['roc_auc']:.4f} "
            f"| ap: {val_metrics['average_precision']:.4f} "
            f"| f1: {val_metrics['f1']:.4f} "
            f"| p@10: {val_metrics['precision@10']:.4f} "
            f"| r@50: {val_metrics['recall@50']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auroc:
            best_val_auroc   = val_metrics["roc_auc"]
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ Best model saved (roc_auc={best_val_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Save per-epoch metrics
    epoch_df = pd.DataFrame(epoch_rows, columns=[c for c in metric_cols if c in epoch_rows[0]])
    epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
    print(f"Epoch metrics saved to {output_dir / 'epoch_metrics.csv'}")

    # ── Test prediction ───────────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    test_edge_index = edge_index[:, test_mask]
    test_scores, test_labels = predict_test(
        model, context,
        edge_index=test_edge_index,
        edge_labels=edge_attr[test_mask, 0],
        edge_times=edge_time[test_mask],
        num_neighbors=num_neighbors,
        batch_size=cfg.train.batch_size,
        device=device,
    )

    test_metrics = compute_metrics(test_labels, test_scores)
    print(
        f"\nTest | roc_auc: {test_metrics['roc_auc']:.4f} "
        f"| ap: {test_metrics['average_precision']:.4f} "
        f"| f1: {test_metrics['f1']:.4f} "
        f"| p@10: {test_metrics['precision@10']:.4f} "
        f"| r@50: {test_metrics['recall@50']:.4f}"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "cutoff_year": int(cutoff_year),
        "train_edges": int(train_mask.sum().item()),
        "val_edges":   int(val_mask.sum().item()),
        "test_edges":  int(test_mask.sum().item()),
        "val":  {f"val_{k}":  float(v) for k, v in best_val_metrics.items()},
        "test": {f"test_{k}": float(v) for k, v in test_metrics.items()},
    }
    OmegaConf.save(OmegaConf.create(results), output_dir / "results.yaml")
    print(f"Results saved to {output_dir / 'results.yaml'}")

    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.finish()

    # ── Save test predictions parquet ─────────────────────────────────────────
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
        default="config/experiments/advancement_hgt.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)

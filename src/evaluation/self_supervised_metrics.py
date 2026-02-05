#!/usr/bin/env python3
"""
Evaluation metrics for self-supervised link prediction.

Provides utilities for:
- Creating negative samples for evaluation
- Computing link prediction metrics (ROC-AUC, AP, MSE, MAE, Huber)
- Evaluating models on validation and test sets
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def create_negative_samples_pyg(
    num_src_nodes: int,
    num_dst_nodes: int,
    pos_edge_index: torch.Tensor,
    num_neg_per_pos: int = 1,
    mode: str = 'both',
    seed: int = None
) -> torch.Tensor:
    """
    Create negative samples by mutating source or destination nodes.
    
    Compatible with PyG's negative sampling approach - samples random nodes
    within the same node type to replace either source or destination.
    
    Args:
        num_src_nodes: Number of source nodes
        num_dst_nodes: Number of destination nodes
        pos_edge_index: Positive edge indices [2, num_pos_edges]
        num_neg_per_pos: Number of negative samples per positive edge
        mode: 'src' (mutate source), 'dst' (mutate destination), or 'both' (random choice)
        seed: Random seed for reproducibility
        
    Returns:
        Negative edge indices [2, num_pos_edges * num_neg_per_pos]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_pos = pos_edge_index.size(1)
    total_neg = num_pos * num_neg_per_pos
    
    # Repeat positive edges
    neg_edge_index = pos_edge_index.repeat(1, num_neg_per_pos)
    
    if mode == 'src':
        # Replace source nodes with random nodes
        neg_edge_index[0] = torch.randint(0, num_src_nodes, (total_neg,))
    elif mode == 'dst':
        # Replace destination nodes with random nodes
        neg_edge_index[1] = torch.randint(0, num_dst_nodes, (total_neg,))
    elif mode == 'both':
        # Randomly choose to replace source or destination for each edge
        replace_src = torch.rand(total_neg) < 0.5
        neg_edge_index[0, replace_src] = torch.randint(0, num_src_nodes, (replace_src.sum(),))
        neg_edge_index[1, ~replace_src] = torch.randint(0, num_dst_nodes, ((~replace_src).sum(),))
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'src', 'dst', or 'both'")
    
    return neg_edge_index


def evaluate_link_prediction(
    model: torch.nn.Module,
    graph: HeteroData,
    pos_edges: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    edge_loss_config: Dict[Tuple[str, str, str], str],
    device: torch.device,
    num_neg_per_pos: int = 1
) -> Dict[str, float]:
    """
    Evaluate link prediction on positive edges with negative sampling.
    
    Args:
        model: Link prediction model
        graph: HeteroData object
        pos_edges: Dict mapping edge_type -> {'edge_index': ..., 'edge_attr': ...}
        edge_loss_config: Dict mapping edge_type -> 'bce' or 'huber'
        device: Device to run evaluation on
        num_neg_per_pos: Number of negative samples per positive edge
        
    Returns:
        Dictionary of metrics:
        - For BCE edges: 'roc_auc', 'avg_precision'
        - For Huber edges: 'mse', 'mae', 'huber'
        - Aggregated metrics with '_mean' suffix
    """
    model.eval()
    
    metrics = {}
    bce_metrics = {'roc_auc': [], 'avg_precision': []}
    huber_metrics = {'mse': [], 'mae': [], 'huber': []}
    
    with torch.no_grad():
        for etype, pos_data in pos_edges.items():
            if pos_data['edge_index'].size(1) == 0:
                continue
            
            src_type, rel, dst_type = etype
            loss_type = edge_loss_config[etype]
            
            # Get positive edge predictions
            pos_edge_index = pos_data['edge_index'].to(device)
            pos_scores = model(
                {k: v.to(device) for k, v in graph.x_dict.items()},
                {k: v.to(device) for k, v in graph.edge_index_dict.items()},
                pos_edge_index,
                src_type,
                dst_type
            )
            
            if loss_type == 'bce':
                # Binary classification metrics - create negative samples
                neg_edge_index = create_negative_samples_pyg(
                    num_src_nodes=graph[src_type].num_nodes,
                    num_dst_nodes=graph[dst_type].num_nodes,
                    pos_edge_index=pos_edge_index,
                    num_neg_per_pos=num_neg_per_pos,
                    mode='both'
                )
                
                # Get negative edge predictions
                neg_scores = model(
                    {k: v.to(device) for k, v in graph.x_dict.items()},
                    {k: v.to(device) for k, v in graph.edge_index_dict.items()},
                    neg_edge_index.to(device),
                    src_type,
                    dst_type
                )
                
                # Combine positive and negative
                all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
                all_labels = np.concatenate([
                    np.ones(pos_scores.size(0)),
                    np.zeros(neg_scores.size(0))
                ])
                
                # Compute metrics
                try:
                    roc_auc = roc_auc_score(all_labels, all_scores)
                    avg_prec = average_precision_score(all_labels, all_scores)
                    
                    metrics[f'{etype}_roc_auc'] = roc_auc
                    metrics[f'{etype}_avg_precision'] = avg_prec
                    
                    bce_metrics['roc_auc'].append(roc_auc)
                    bce_metrics['avg_precision'].append(avg_prec)
                except ValueError as e:
                    print(f"Warning: Could not compute metrics for {etype}: {e}")
            
            else:  # huber
                # Regression metrics
                if pos_data['edge_attr'] is not None:
                    targets = pos_data['edge_attr'].flatten().to(device)
                    
                    # MSE
                    mse = F.mse_loss(pos_scores, targets).item()
                    
                    # MAE
                    mae = F.l1_loss(pos_scores, targets).item()
                    
                    # Huber loss
                    huber = F.huber_loss(pos_scores, targets).item()
                    
                    metrics[f'{etype}_mse'] = mse
                    metrics[f'{etype}_mae'] = mae
                    metrics[f'{etype}_huber'] = huber
                    
                    huber_metrics['mse'].append(mse)
                    huber_metrics['mae'].append(mae)
                    huber_metrics['huber'].append(huber)
    
    # Compute aggregated metrics
    if bce_metrics['roc_auc']:
        metrics['roc_auc_mean'] = np.mean(bce_metrics['roc_auc'])
        metrics['avg_precision_mean'] = np.mean(bce_metrics['avg_precision'])
    
    if huber_metrics['mse']:
        metrics['mse_mean'] = np.mean(huber_metrics['mse'])
        metrics['mae_mean'] = np.mean(huber_metrics['mae'])
        metrics['huber_mean'] = np.mean(huber_metrics['huber'])
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metric name -> value
        prefix: Prefix for print statements (e.g., "Validation", "Test")
    """
    if not metrics:
        print(f"{prefix} No metrics to display")
        return
    
    print(f"\n{prefix} Metrics:")
    print("=" * 60)
    
    # Print aggregated metrics first
    agg_metrics = {k: v for k, v in metrics.items() if '_mean' in k}
    if agg_metrics:
        print("Aggregated:")
        for name, value in sorted(agg_metrics.items()):
            print(f"  {name}: {value:.4f}")
    
    # Print per-edge-type metrics
    per_type_metrics = {k: v for k, v in metrics.items() if '_mean' not in k}
    if per_type_metrics:
        print("\nPer Edge Type:")
        for name, value in sorted(per_type_metrics.items()):
            print(f"  {name}: {value:.4f}")
    
    print("=" * 60)

#!/usr/bin/env python3
"""
Evaluation metrics for link prediction / recommendation.

Implements ranking metrics: Recall@k, Precision@k, NDCG@k, MRR.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def recall_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Recall@k.
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        Recall@k score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # Count relevant items in top-k
    relevant_in_top_k = labels[top_k_indices].sum().item()
    total_relevant = labels.sum().item()
    
    return relevant_in_top_k / total_relevant


def precision_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Precision@k.
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        Precision@k score
    """
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # Count relevant items in top-k
    relevant_in_top_k = labels[top_k_indices].sum().item()
    
    return relevant_in_top_k / k


def ndcg_at_k(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> float:
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        k: Top-k
        
    Returns:
        NDCG@k score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Get top-k indices
    _, top_k_indices = torch.topk(scores, k=min(k, len(scores)))
    
    # DCG: sum of (relevance / log2(rank + 1))
    relevance = labels[top_k_indices].float()
    ranks = torch.arange(1, len(top_k_indices) + 1, dtype=torch.float32)
    dcg = (relevance / torch.log2(ranks + 1)).sum().item()
    
    # IDCG: ideal DCG (all relevant items at top)
    ideal_relevance, _ = torch.sort(labels.float(), descending=True)
    ideal_relevance = ideal_relevance[:k]
    ideal_ranks = torch.arange(1, len(ideal_relevance) + 1, dtype=torch.float32)
    idcg = (ideal_relevance / torch.log2(ideal_ranks + 1)).sum().item()
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mean_reciprocal_rank(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        scores: Predicted scores [num_items]
        labels: Binary labels [num_items]
        
    Returns:
        MRR score
    """
    if labels.sum() == 0:
        return 0.0
    
    # Sort by scores (descending)
    _, sorted_indices = torch.sort(scores, descending=True)
    sorted_labels = labels[sorted_indices]
    
    # Find rank of first relevant item (1-indexed)
    relevant_ranks = torch.where(sorted_labels == 1)[0]
    if len(relevant_ranks) == 0:
        return 0.0
    
    first_relevant_rank = relevant_ranks[0].item() + 1  # 1-indexed
    
    return 1.0 / first_relevant_rank


def compute_ranking_metrics(
    scores_dict: Dict[int, torch.Tensor],
    labels_dict: Dict[int, torch.Tensor],
    k_values: List[int] = [10, 20, 50, 100],
) -> Dict[str, float]:
    """
    Compute ranking metrics for all users.
    
    Args:
        scores_dict: Dict mapping user index to scores
        labels_dict: Dict mapping user index to labels
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of aggregated metrics
    """
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics["mrr"] = []
    
    for user_idx in scores_dict.keys():
        scores = scores_dict[user_idx]
        labels = labels_dict[user_idx]
        
        # Skip users with no relevant items
        if labels.sum() == 0:
            continue
        
        # Compute metrics for each k
        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(scores, labels, k))
            metrics[f"precision@{k}"].append(precision_at_k(scores, labels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(scores, labels, k))
        
        # MRR
        metrics["mrr"].append(mean_reciprocal_rank(scores, labels))
    
    # Average across users
    aggregated = {}
    for metric_name, values in metrics.items():
        if values:
            aggregated[metric_name] = np.mean(values)
        else:
            aggregated[metric_name] = 0.0
    
    return aggregated

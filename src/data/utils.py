#!/usr/bin/env python3
"""
Data utilities for temporal splitting, cold-start evaluation, and node features.
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Set


def temporal_split(
    edges: pd.DataFrame,
    cutoff_year: int,
    horizon: int = 2,
    val_years: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split edges temporally into train/val/test sets.
    
    Args:
        edges: DataFrame with edges containing 'year' column
        cutoff_year: Training data up to this year (inclusive)
        horizon: Total prediction horizon in years
        val_years: Number of years for validation
        
    Returns:
        train_edges: Edges with year <= cutoff_year
        val_edges: Edges in (cutoff_year, cutoff_year + val_years]
        test_edges: Edges in (cutoff_year + val_years, cutoff_year + horizon]
    """
    if "year" not in edges.columns:
        raise ValueError("Edges DataFrame must have 'year' column for temporal split")
    
    # Training: up to cutoff year
    train_edges = edges[edges["year"] <= cutoff_year].copy()
    
    # Validation: next val_years
    val_start = cutoff_year + 1
    val_end = cutoff_year + val_years
    val_edges = edges[(edges["year"] >= val_start) & (edges["year"] <= val_end)].copy()
    
    # Test: remaining horizon
    test_start = val_end + 1
    test_end = cutoff_year + horizon
    test_edges = edges[(edges["year"] >= test_start) & (edges["year"] <= test_end)].copy()
    
    print(f"\n📊 Temporal Split:")
    print(f"  Train: year <= {cutoff_year} ({len(train_edges)} edges)")
    print(f"  Val:   {val_start} <= year <= {val_end} ({len(val_edges)} edges)")
    print(f"  Test:  {test_start} <= year <= {test_end} ({len(test_edges)} edges)")
    
    return train_edges, val_edges, test_edges


def cold_start_split(
    edges: pd.DataFrame,
    train_edges: pd.DataFrame,
    cold_start_ids: Optional[List[str]] = None,
    min_interactions: int = 5,
    user_col: str = "sourceId",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate cold-start users from warm-start users.
    
    Args:
        edges: All edges (val or test)
        train_edges: Training edges
        cold_start_ids: Optional list of cold-start user IDs
        min_interactions: Minimum interactions in training to be warm-start
        user_col: Column name for user IDs
        
    Returns:
        warm_edges: Edges for warm-start users
        cold_edges: Edges for cold-start users
    """
    # Count training interactions per user
    train_counts = train_edges[user_col].value_counts()
    
    if cold_start_ids is not None:
        # Use provided cold-start IDs
        cold_set = set(cold_start_ids)
    else:
        # Define cold-start as users with < min_interactions in training
        cold_set = set(train_counts[train_counts < min_interactions].index)
        # Also include users not in training at all
        all_users = set(edges[user_col].unique())
        train_users = set(train_edges[user_col].unique())
        cold_set = cold_set | (all_users - train_users)
    
    # Split edges
    cold_edges = edges[edges[user_col].isin(cold_set)].copy()
    warm_edges = edges[~edges[user_col].isin(cold_set)].copy()
    
    print(f"\n🧊 Cold-Start Split:")
    print(f"  Warm-start: {len(warm_edges)} edges ({len(warm_edges[user_col].unique())} users)")
    print(f"  Cold-start: {len(cold_edges)} edges ({len(cold_edges[user_col].unique())} users)")
    
    return warm_edges, cold_edges


def attach_node_features(
    data: HeteroData,
    id_maps: Dict[str, Dict[str, int]],
    init_method: str = "random",
    embedding_dim: int = 128,
    pretrained_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    seed: int = 42,
) -> HeteroData:
    """
    Initialize node features for heterogeneous graph.
    
    Args:
        data: HeteroData object
        id_maps: Node ID to index mappings
        init_method: "random" or "pretrained"
        embedding_dim: Dimension for random embeddings
        pretrained_embeddings: Optional dict mapping node_type to embedding tensor
        seed: Random seed
        
    Returns:
        data: HeteroData with node features attached
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        
        if init_method == "random":
            # Random normal initialization
            data[node_type].x = torch.randn(num_nodes, embedding_dim)
            print(f"✅ Initialized {node_type} with random features: {data[node_type].x.shape}")
            
        elif init_method == "pretrained":
            if pretrained_embeddings is None or node_type not in pretrained_embeddings:
                # Fall back to random if pretrained not available
                data[node_type].x = torch.randn(num_nodes, embedding_dim)
                print(f"⚠️ No pretrained embeddings for {node_type}, using random")
            else:
                data[node_type].x = pretrained_embeddings[node_type]
                print(f"✅ Loaded pretrained features for {node_type}: {data[node_type].x.shape}")
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    return data

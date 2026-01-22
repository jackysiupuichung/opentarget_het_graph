#!/usr/bin/env python3
"""
Temporal graph loader utilities for event-based graphs.

Handles loading of event-based HeteroData and creating temporal masks.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional
from pathlib import Path
from torch_geometric.utils import coalesce


def load_event_graph(
    filepath: str,
    attach_features: bool = False,
    embedding_dim: int = 128,
    seed: int = 42
) -> HeteroData:
    """
    Load event-based temporal graph (HeteroData).
    
    Args:
        filepath: Path to temporal graph file (.pt)
        attach_features: Whether to attach random node features
        embedding_dim: Embedding dimension
        seed: Random seed
        
    Returns:
        HeteroData object with edge_time and edge_weight attributes
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Temporal graph file not found: {filepath}")
    
    # Load HeteroData object
    data = torch.load(filepath, weights_only=False)
    
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")
    
    # Optionally attach features
    if attach_features:
        from .utils import attach_node_features
        # Create dummy id_maps (nodes already indexed in graph)
        id_maps = {}
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            id_maps[node_type] = {str(i): i for i in range(num_nodes)}
        
        data = attach_node_features(
            data,
            id_maps,
            init_method="random",
            embedding_dim=embedding_dim,
            seed=seed
        )
    
    return data


def filter_graph_by_time(data: HeteroData, year: int) -> HeteroData:
    """
    Filter graph to include only edges up to a specific year.
    Creates a snapshot view of the graph.
    
    Args:
        data: HeteroData object with edge_time
        year: Max year to include (inclusive)
        
    Returns:
        New HeteroData object with filtered edges
    """
    new_data = data.clone()
    
    for et in new_data.edge_types:
        if 'edge_time' in new_data[et]:
            edge_time = new_data[et].edge_time
            mask = edge_time <= year
            
            # Filter edge_index
            new_data[et].edge_index = new_data[et].edge_index[:, mask]
            
            # Filter attributes
            for key in ['edge_time', 'edge_weight', 'edge_attr']:
                if key in new_data[et]:
                    new_data[et][key] = new_data[et][key][mask]
        else:
            # Keep static edges as is
            pass
            
    return new_data


def get_temporal_masks(
    data: HeteroData,
    train_year: int,
    val_year: int
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create train/val/test masks based on edge_time.
    
    Args:
        data: HeteroData object with edge_time
        train_year: Max year for training (inclusive)
        val_year: Max year for validation (inclusive)
        
    Returns:
        Dictionary mapping edge_type -> (train_mask, val_mask, test_mask)
    """
    masks = {}
    
    for edge_type in data.edge_types:
        if 'edge_time' not in data[edge_type]:
            # If no time, assume context (all train)
            num_edges = data[edge_type].edge_index.size(1)
            train_mask = torch.ones(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask = torch.zeros(num_edges, dtype=torch.bool)
        else:
            edge_time = data[edge_type].edge_time
            
            # Allow scalar comparison even if edge_time is float
            train_mask = edge_time <= train_year
            val_mask = (edge_time > train_year) & (edge_time <= val_year)
            test_mask = edge_time > val_year
            
        masks[edge_type] = (train_mask, val_mask, test_mask)
        
    return masks


def print_temporal_summary(data: HeteroData):
    """
    Print summary of temporal graph events.
    
    Args:
        data: HeteroData object
    """
    print(f"\n📊 Temporal Graph Summary")
    print(f"{'='*80}")
    
    print(f"Nodes:")
    for nt in data.node_types:
        print(f"   {nt}: {data[nt].num_nodes:,}")
        
    print(f"\nEdges:")
    for et in data.edge_types:
        num_edges = data[et].edge_index.size(1)
        has_time = 'edge_time' in data[et]
        has_weight = 'edge_weight' in data[et] or 'edge_attr' in data[et]
        
        info = []
        if has_time: 
            min_t = int(data[et].edge_time.min())
            max_t = int(data[et].edge_time.max())
            info.append(f"Time: {min_t}-{max_t}")
        if has_weight: 
            info.append("Weighted")
            
        print(f"   {et}: {num_edges:,} {' | '.join(info)}")


def to_time_agnostic(data: HeteroData) -> HeteroData:
    """
    Collapse temporal graph into a static time-agnostic graph.
    
    Aggregates multiple edges between the same (src, dst) pair into a single edge.
    Aggregation method: 'max' for edge weights/attributes.
    Removes 'edge_time' attribute.
    
    Args:
        data: HeteroData object (temporal)
        
    Returns:
        New HeteroData object (static)
    """
    new_data = data.clone()
    
    print(f"\nTime-Agnostic Collapsing:")
    
    for et in new_data.edge_types:
        edge_index = new_data[et].edge_index
        num_edges_before = edge_index.size(1)
        
        # Gather attributes to aggregate
        # We assume 'edge_weight' or 'edge_attr' are the scores to max.
        # If both exist, we need to handle them. Coalesce handles one 'edge_attr'.
        # If we have multiple, we might need multiple passes or stack them?
        # Typically HGT uses 'edge_attr' or 'edge_weight'.
        
        edge_attr = None
        if 'edge_weight' in new_data[et]:
            edge_attr = new_data[et].edge_weight
            if edge_attr.dim() == 1: edge_attr = edge_attr.view(-1, 1) # Make sure it's [N, 1]
        elif 'edge_attr' in new_data[et]:
            edge_attr = new_data[et].edge_attr
        
        if edge_attr is not None:
            # Coalesce with max reduction
            new_index, new_attr = coalesce(
                edge_index, 
                edge_attr, 
                reduce='max'
            )
            
            new_data[et].edge_index = new_index
            
            # Restore attribute name
            if 'edge_weight' in new_data[et]:
                new_data[et].edge_weight = new_attr.squeeze() # [N] or [N, 1]
            elif 'edge_attr' in new_data[et]:
                new_data[et].edge_attr = new_attr
                
        else:
            # Just unique edges (no weights)
            # coalesce without attr works? No, it expects attr or you use unique.
            # torch_geometric.utils.coalesce requires edge_attr? 
            # Actually, if edge_attr is None, it returns (edge_index, None) if provided?
            # Documentation says: "If edge_attr is None, it will be ignored."?
            # Actually, `coalesce(edge_index, edge_attr=None, ...)` returns `edge_index`.
            
            new_index = coalesce(
                edge_index, 
                None
            )
            new_data[et].edge_index = new_index

        # Remove temporal attributes
        if 'edge_time' in new_data[et]:
            del new_data[et].edge_time
            
        num_edges_after = new_data[et].edge_index.size(1)
        print(f"   {et}: {num_edges_before:,} -> {num_edges_after:,} edges (Max Aggregation)")
        
    return new_data

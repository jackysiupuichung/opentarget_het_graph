#!/usr/bin/env python3
"""
Temporal graph loader utilities for event-based graphs.

Handles loading of event-based HeteroData and creating temporal masks.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T


def load_event_graph(
    filepath: str,
    attach_features: bool = False,
    to_undirected: bool = False,
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
    
    # 1. Convert to undirected for GNN message passing
    if to_undirected:
        print("🔄 Converting to undirected graph (adding reverse edges)...")
        data = T.ToUndirected()(data)
    
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
    Creates a temporal cut off view of the graph.
    
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
    split_config=None,
    train_year: int = None,
    val_year: int = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create train/val/test masks based on edge_time.
    Supports either explicit ranges (split_config) or legacy cumulative years (train_year, val_year).
    
    Args:
        data: HeteroData object with edge_time
        split_config: Dict with 'train', 'val', 'test' keys containing [start, end] lists.
        train_year: (Legacy) Max year for training.
        val_year: (Legacy) Max year for validation.
        
    Returns:
        Dictionary mapping edge_type -> (train_mask, val_mask, test_mask)
    """
    masks = {}
    
    # helper to check range
    def is_in_range(times, rng):
        start, end = rng
        return (times >= start) & (times <= end)

    for edge_type in data.edge_types:
        if 'edge_time' not in data[edge_type]:
            # If no time, assume context (all train)
            num_edges = data[edge_type].edge_index.size(1)
            train_mask = torch.ones(num_edges, dtype=torch.bool)
            val_mask = torch.zeros(num_edges, dtype=torch.bool)
            test_mask = torch.zeros(num_edges, dtype=torch.bool)
        else:
            edge_time = data[edge_type].edge_time
            
            if split_config is not None:
                # New Range-based Logic
                # Check for explicit 'train', 'val', 'test' ranges
                # split_config might be OmegaConf object or dict
                
                # Default ranges if missing
                tr_range = split_config.get('train', [0, 0])
                val_range = split_config.get('val', [0, 0])
                test_range = split_config.get('test', [0, 0])
                
                train_mask = is_in_range(edge_time, tr_range)
                val_mask = is_in_range(edge_time, val_range)
                test_mask = is_in_range(edge_time, test_range)
                
            else:
                # Legacy Logic (Cumulative)
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


def to_temporal_snapshots(
    data: HeteroData,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    verbose: bool = True
) -> Dict[int, HeteroData]:
    """
    Materialize yearly snapshots of the graph.
    
    For each year y, creates a static graph containing the max score of edges
    observed up to year y (cumulative).
    
    Args:
        data: HeteroData object (temporal)
        start_year: Start year (inclusive). Defaults to min edge time.
        end_year: End year (inclusive). Defaults to max edge time.
        verbose: Print progress
        
    Returns:
        Dictionary {year: static_hetero_data}
    """
    if verbose:
        print("\n📸 Materializing Temporal Snapshots...")
        
    # Determine range
    all_times = []
    for et in data.edge_types:
        if 'edge_time' in data[et]:
            all_times.append(data[et].edge_time)
            
    if not all_times:
        print("⚠️ No temporal information found. Returning single snapshot.")
        return {0: to_time_agnostic(data)}
        
    all_times = torch.cat(all_times)
    min_t = int(all_times.min().item())
    max_t = int(all_times.max().item())
    
    if start_year is None: start_year = min_t
    if end_year is None: end_year = max_t
    
    snapshots = {}
    
    for year in range(start_year, end_year + 1):
        if verbose: print(f"\n🗓️  Year: {year}")
        
        # 1. Filter (Cumulative <= year)
        # Note: filter_graph_by_time follows cumulative logic
        snapshot_temporal = filter_graph_by_time(data, year)
        
        # 2. Collapse to static
        # This keeps the MAX score for duplicate edges
        snapshot_static = to_time_agnostic(snapshot_temporal)
        
        snapshots[year] = snapshot_static
        
    if verbose: print(f"\n✅ Created {len(snapshots)} snapshots ({start_year}-{end_year})")
    return snapshots

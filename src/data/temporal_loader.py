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
    to_undirected: bool = False,
) -> HeteroData:
    """
    Load event-based temporal graph (HeteroData).
    
    Note: Features should be attached separately using src/pipeline/attach_features.py
    
    Args:
        filepath: Path to temporal graph file (.pt)
        to_undirected: Whether to add reverse edges for message passing
        
    Returns:
        HeteroData object with edge_time and edge_weight attributes
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Temporal graph file not found: {filepath}")
    
    # Load HeteroData object
    data = torch.load(filepath, weights_only=False)
    
    if not isinstance(data, HeteroData):
        raise TypeError(f"Expected HeteroData, got {type(data)}")
    
    # Convert to undirected for GNN message passing
    if to_undirected:
        print("🔄 Converting to undirected graph (adding reverse edges)...")
        data = T.ToUndirected()(data)
    
    # Remove node_id attribute if present to avoid PyG loader errors
    # (PyG loader tries to slice all attributes, and list[str] fails)
    for node_type in data.node_types:
        if hasattr(data[node_type], 'node_id'):
            del data[node_type].node_id
            
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
    split_config: Dict[str, List[int]]
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create train/val/test masks based on edge_time using split configuration.
    
    Args:
        data: HeteroData object with edge_time
        split_config: Dict with 'train', 'val', 'test' keys containing [start, end] lists.
                     Example: {'train': [1990, 2015], 'val': [2016, 2017], 'test': [2018, 2024]}
        
    Returns:
        Dictionary mapping edge_type -> (train_mask, val_mask, test_mask)
        
    Raises:
        ValueError: If split_config is None or missing required keys
    """
    if split_config is None:
        raise ValueError(
            "split_config is required. Must provide a dict with 'train', 'val', 'test' keys.\n"
            "Example: {'train': [1990, 2015], 'val': [2016, 2017], 'test': [2018, 2024]}"
        )
    
    # Validate split_config has required keys
    required_keys = {'train', 'val', 'test'}
    missing_keys = required_keys - set(split_config.keys())
    if missing_keys:
        raise ValueError(
            f"split_config missing required keys: {missing_keys}. "
            f"Must provide 'train', 'val', and 'test' ranges."
        )
    
    masks = {}
    
    # Helper to check if times fall within range
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
            
            # Get ranges from split_config (handles both dict and OmegaConf)
            tr_range = split_config.get('train') or split_config['train']
            val_range = split_config.get('val') or split_config['val']
            test_range = split_config.get('test') or split_config['test']
            
            train_mask = is_in_range(edge_time, tr_range)
            val_mask = is_in_range(edge_time, val_range)
            test_mask = is_in_range(edge_time, test_range)
            
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

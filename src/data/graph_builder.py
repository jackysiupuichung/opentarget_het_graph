#!/usr/bin/env python3
"""
Graph builder for heterogeneous graphs - RELATION::SOURCE LEVEL ONLY.

This module constructs PyTorch Geometric HeteroData objects from edge files,
always using (source, relation::datasource, target) format with edge scores.
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from glob import glob
import os


def load_edges(edge_dir: str, cutoff_year: int = None) -> pd.DataFrame:
    """
    Load all edge parquet files from directory.
    
    Args:
        edge_dir: Directory containing edge parquet files
        cutoff_year: Optional year cutoff (only include edges <= cutoff_year)
        
    Returns:
        DataFrame with all edges
    """
    dfs = []
    
    for parquet_file in glob(os.path.join(edge_dir, "*.parquet")):
        df = pd.read_parquet(parquet_file)
        
        if df.empty:
            continue
        
        # Filter by cutoff year if specified
        if cutoff_year is not None and "year" in df.columns:
            df = df[df["year"] <= cutoff_year]
        
        dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def extract_nodes_from_edges(edges: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Extract unique nodes from edges.
    
    Args:
        edges: DataFrame with edges containing sourceId, targetId, source_type, target_type
        
    Returns:
        nodes: Dictionary mapping node type to list of node IDs
        id_to_type: Dictionary mapping node ID to node type
    """
    nodes = {}
    id_to_type = {}
    
    # Extract source nodes
    for _, row in edges[['sourceId', 'source_type']].drop_duplicates().iterrows():
        node_id = str(row['sourceId'])
        node_type = row['source_type']
        
        if node_type not in nodes:
            nodes[node_type] = []
        
        if node_id not in id_to_type:
            nodes[node_type].append(node_id)
            id_to_type[node_id] = node_type
    
    # Extract target nodes
    for _, row in edges[['targetId', 'target_type']].drop_duplicates().iterrows():
        node_id = str(row['targetId'])
        node_type = row['target_type']
        
        if node_type not in nodes:
            nodes[node_type] = []
        
        if node_id not in id_to_type:
            nodes[node_type].append(node_id)
            id_to_type[node_id] = node_type
    
    # Remove duplicates and sort
    for node_type in nodes:
        nodes[node_type] = sorted(list(set(nodes[node_type])))
    
    return nodes, id_to_type


def build_hetero_graph(edges: pd.DataFrame) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
    """
    Build heterogeneous graph from edges - RELATION::SOURCE LEVEL ONLY.
    
    Always uses (source_type, "relation::datasource", target_type) format with scores.
    Supports temporal attributes: edge_time and edge_weight.
    
    Args:
        edges: DataFrame with edges (sourceId, targetId, source_type, target_type, 
               relation, datasourceId, score)
               Optional: edge_time (year/timestamp), edge_weight (for events)
        
    Returns:
        hetero_data: HeteroData object
        id_maps: Dictionary mapping node_type -> {node_id_str -> internal_idx}
    """
    print("\n🔨 Building HeteroData (relation::source level)...")
    
    # Extract nodes
    print("📊 Extracting nodes from edges...")
    nodes, id_to_type = extract_nodes_from_edges(edges)
    
    # Create ID mappings
    id_maps = {}
    for node_type, node_list in nodes.items():
        id_maps[node_type] = {node_id: idx for idx, node_id in enumerate(node_list)}
        print(f"   {node_type}: {len(node_list)} nodes")
    
    # Build HeteroData
    hetero_data = HeteroData()
    
    # Add nodes
    print("\n🔗 Adding nodes...")
    for node_type, node_list in nodes.items():
        hetero_data[node_type].num_nodes = len(node_list)
        print(f"✅ Added {len(node_list)} {node_type} nodes")
    
    # Build edges
    print("\n🔗 Building edges (relation::datasource level)...")
    
    # Check for temporal attributes
    has_edge_time = 'edge_time' in edges.columns
    has_edge_weight = 'edge_weight' in edges.columns
    has_score = 'score' in edges.columns and not has_edge_weight
    
    # Group by edge type
    edge_groups = edges.groupby(['source_type', 'relation', 'target_type', 'datasourceId'])
    
    for (src_type, relation, dst_type, datasource), group in edge_groups:
        # Create edge type key
        edge_type_key = (src_type, f"{relation}::{datasource}", dst_type)
        
        # Map node IDs to indices
        src_indices = [id_maps[src_type][str(sid)] for sid in group['sourceId']]
        dst_indices = [id_maps[dst_type][str(tid)] for tid in group['targetId']]
        
        # Create edge_index
        edge_index = torch.tensor(
            [src_indices, dst_indices],
            dtype=torch.long
        )
        
        hetero_data[edge_type_key].edge_index = edge_index
        
        # Add edge attributes
        if has_edge_weight:
            # Event-based: use edge_weight
            hetero_data[edge_type_key].edge_attr = torch.tensor(
                group['edge_weight'].values,
                dtype=torch.long
            ).unsqueeze(-1)
        elif has_score:
            # Snapshot-based: use score
            hetero_data[edge_type_key].edge_attr = torch.tensor(
                group['score'].values,
                dtype=torch.long
            ).unsqueeze(-1)
        
        # Add temporal attribute
        if has_edge_time:
            hetero_data[edge_type_key].edge_time = torch.tensor(
                group['edge_time'].values,
                dtype=torch.long  # Use long for temporal sampling compatibility
            )
        
        num_edges = edge_index.size(1)
        attrs_str = []
        if has_edge_weight or has_score:
            attrs_str.append("edge_attr")
        if has_edge_time:
            attrs_str.append("edge_time")
        
        attr_info = f" ({', '.join(attrs_str)})" if attrs_str else ""
        print(f"✅ Added {num_edges} edges for {edge_type_key}{attr_info}")
    
    return hetero_data, id_maps

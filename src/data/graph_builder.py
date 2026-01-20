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
    
    Args:
        edges: DataFrame with edges (sourceId, targetId, source_type, target_type, 
               relation, datasourceId, score)
        
    Returns:
        data: HeteroData object
        id_maps: Dictionary mapping node type to {node_id: index}
    """
    data = HeteroData()
    id_maps = {}
    
    # Required columns
    required_cols = ["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "score"]
    for col in required_cols:
        if col not in edges.columns:
            raise ValueError(f"Edges DataFrame missing required column: {col}")
    
    # ============================================================
    # 1. Extract and add nodes
    # ============================================================
    print("📊 Extracting nodes from edges...")
    nodes, id_to_type = extract_nodes_from_edges(edges)
    
    for node_type, node_ids in nodes.items():
        # Create ID mapping
        id_map = {nid: i for i, nid in enumerate(node_ids)}
        id_maps[node_type] = id_map
        
        # Store number of nodes
        data[node_type].num_nodes = len(node_ids)
        
        print(f"✅ Added {len(node_ids)} {node_type} nodes")
    
    # ============================================================
    # 2. Add edges - ALWAYS relation::datasource level
    # ============================================================
    if not edges.empty:
        print("\n🔗 Building edges (relation::datasource level)...")
        
        # Group by: source_type, relation, datasourceId, target_type
        edge_groups = edges.groupby(["source_type", "relation", "datasourceId", "target_type"])
        
        for edge_key, edge_df in edge_groups:
            src_type, relation, datasource, dst_type = edge_key
            
            # Edge type: (src_type, "relation::datasource", dst_type)
            edge_type = (src_type, f"{relation}::{datasource}", dst_type)
            
            # Get ID mappings
            if src_type not in id_maps or dst_type not in id_maps:
                print(f"⚠️ Skipping edge type {edge_type}: missing node type")
                continue
            
            src_map = id_maps[src_type]
            dst_map = id_maps[dst_type]
            
            # Map source and target IDs to indices
            src_ids = edge_df["sourceId"].astype(str).tolist()
            dst_ids = edge_df["targetId"].astype(str).tolist()
            scores = edge_df["score"].values
            
            # Filter out edges with unknown nodes
            valid_edges = []
            valid_scores = []
            for src_id, dst_id, score in zip(src_ids, dst_ids, scores):
                if src_id in src_map and dst_id in dst_map:
                    valid_edges.append((src_map[src_id], dst_map[dst_id]))
                    valid_scores.append(score)
            
            if not valid_edges:
                continue
            
            # Create edge_index tensor
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
            data[edge_type].edge_index = edge_index
            
            # ALWAYS add edge scores as edge_attr
            edge_attr = torch.tensor(valid_scores, dtype=torch.float).unsqueeze(1)
            data[edge_type].edge_attr = edge_attr
            
            print(f"✅ Added {edge_index.size(1)} edges for {edge_type}")
    
    return data, id_maps


def add_supervision_labels(
    data: HeteroData,
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    id_maps: Dict[str, Dict[str, int]],
    supervision_src_type: str = "disease",
    supervision_relation: str = "clinical_trial",
    supervision_dst_type: str = "target",
) -> HeteroData:
    """
    Add train/val/test edge labels to HeteroData for link prediction.
    
    Note: This aggregates across all datasources for the supervision relation.
    
    Args:
        data: HeteroData object
        train_edges: Training edges DataFrame
        val_edges: Validation edges DataFrame
        test_edges: Test edges DataFrame
        id_maps: Node ID to index mappings
        supervision_src_type: Source node type
        supervision_relation: Relation type
        supervision_dst_type: Destination node type
        
    Returns:
        data: HeteroData with edge labels added
    """
    src_map = id_maps[supervision_src_type]
    dst_map = id_maps[supervision_dst_type]
    
    def map_edges(edge_df: pd.DataFrame, split: str):
        """Map edges to indices and add to data."""
        if edge_df.empty:
            return
        
        # Filter to supervision relation
        supervision_df = edge_df[
            (edge_df["source_type"] == supervision_src_type) &
            (edge_df["target_type"] == supervision_dst_type) &
            (edge_df["relation"] == supervision_relation)
        ]
        
        if supervision_df.empty:
            return
        
        src_ids = supervision_df["sourceId"].astype(str).tolist()
        dst_ids = supervision_df["targetId"].astype(str).tolist()
        
        valid_edges = []
        for src_id, dst_id in zip(src_ids, dst_ids):
            if src_id in src_map and dst_id in dst_map:
                valid_edges.append((src_map[src_id], dst_map[dst_id]))
        
        if valid_edges:
            # Remove duplicates (same edge from multiple datasources)
            valid_edges = list(set([(s, d) for s, d in valid_edges]))
            
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
            
            # Store as edge_label_index for each datasource edge type
            # We'll use the first datasource edge type we find
            for edge_type in data.edge_types:
                if (edge_type[0] == supervision_src_type and 
                    edge_type[2] == supervision_dst_type and
                    supervision_relation in edge_type[1]):
                    data[edge_type][f"{split}_edge_label_index"] = edge_index
                    data[edge_type][f"{split}_edge_label"] = torch.ones(edge_index.size(1), dtype=torch.float)
                    print(f"✅ Added {edge_index.size(1)} {split} supervision edges to {edge_type}")
                    break
    
    map_edges(train_edges, "train")
    map_edges(val_edges, "val")
    map_edges(test_edges, "test")
    
    return data

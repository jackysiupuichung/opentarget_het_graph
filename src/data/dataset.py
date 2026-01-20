#!/usr/bin/env python3
"""
Dataset for heterogeneous link prediction.

This module provides a PyTorch Dataset for link prediction on heterogeneous graphs,
compatible with PyTorch Geometric's LinkNeighborLoader for mini-batch training.
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Set, Optional


class HeteroLinkDataset(Dataset):
    """
    Dataset for heterogeneous link prediction.
    
    Provides positive and negative edges for training/evaluation on heterogeneous graphs.
    """
    
    def __init__(
        self,
        edges: pd.DataFrame,
        src_id_map: Dict[str, int],
        dst_id_map: Dict[str, int],
        num_negatives: int = 1,
        all_interactions: Optional[Dict[int, Set[int]]] = None,
        exhaustive_eval: bool = False,
        seed: int = 42,
    ):
        """
        Initialize dataset.
        
        Args:
            edges: DataFrame with positive edges (sourceId, targetId)
            src_id_map: Mapping from source node ID to index
            dst_id_map: Mapping from destination node ID to index
            num_negatives: Number of negative samples per positive edge
            all_interactions: Dict mapping user index to set of item indices (for filtering)
            exhaustive_eval: If True, evaluate against all items (for test set)
            seed: Random seed
        """
        self.edges = edges.copy()
        self.src_id_map = src_id_map
        self.dst_id_map = dst_id_map
        self.num_negatives = num_negatives
        self.exhaustive_eval = exhaustive_eval
        self.seed = seed
        
        # Map edges to indices
        self.edge_pairs = []
        for _, row in edges.iterrows():
            src_id = str(row["sourceId"])
            dst_id = str(row["targetId"])
            
            if src_id in src_id_map and dst_id in dst_id_map:
                src_idx = src_id_map[src_id]
                dst_idx = dst_id_map[dst_id]
                self.edge_pairs.append((src_idx, dst_idx))
        
        print(f"✅ Dataset initialized: {len(self.edge_pairs)} positive edges")
    
    def __len__(self) -> int:
        """Return number of positive edges."""
        return len(self.edge_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with edge information
        """
        src_idx, dst_idx = self.edge_pairs[idx]
        
        return {
            "user_id": torch.tensor(src_idx, dtype=torch.long),
            "item_id": torch.tensor(dst_idx, dtype=torch.long),
            "label": torch.tensor(1.0, dtype=torch.float),
        }
    
    def get_edge_index(self) -> torch.Tensor:
        """
        Get all positive edges as edge_index tensor.
        
        Returns:
            Edge index tensor [2, num_edges]
        """
        if not self.edge_pairs:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(self.edge_pairs, dtype=torch.long).t().contiguous()


def build_all_interactions(
    edges: pd.DataFrame,
    src_id_map: Dict[str, int],
    dst_id_map: Dict[str, int],
) -> Dict[int, Set[int]]:
    """
    Build dictionary of all user-item interactions.
    
    Args:
        edges: DataFrame with edges
        src_id_map: Source ID to index mapping
        dst_id_map: Destination ID to index mapping
        
    Returns:
        Dictionary mapping source index to set of destination indices
    """
    interactions = {}
    
    for _, row in edges.iterrows():
        src_id = str(row["sourceId"])
        dst_id = str(row["targetId"])
        
        if src_id in src_id_map and dst_id in dst_id_map:
            src_idx = src_id_map[src_id]
            dst_idx = dst_id_map[dst_id]
            interactions.setdefault(src_idx, set()).add(dst_idx)
    
    return interactions

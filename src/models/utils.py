#!/usr/bin/env python3
"""
Model utilities for building and managing HGT models.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from .hgt import HGTLinkPredictor
from .gat_v2 import GATv2
from .gat_v3 import GATv3



def get_metadata(data: HeteroData) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract metadata from HeteroData.
    
    Args:
        data: HeteroData object
        
    Returns:
        (node_types, edge_types) tuple
    """
    node_types = data.node_types
    edge_types = data.edge_types
    return (node_types, edge_types)


def build_model(
    model_name: str,
    data: HeteroData,
    hidden_dim: int = 128,
    out_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    Build model from HeteroData.
    
    Args:
        model_name: Name of model to build (hgt, gatv2, gatv3, etc)
        data: HeteroData object
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of layers
        dropout: Dropout rate
        
    Returns:
        differentiable PyTorch model
    """
    node_types, edge_types = get_metadata(data)
    metadata = (node_types, edge_types)
    
    if model_name == 'hgt':
        model = HGTLinkPredictor(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            node_types=node_types,
            metadata=metadata,
            dropout=dropout,
        )
    elif model_name == 'gatv2':
        model = GATv2(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    elif model_name == 'gatv3':
        model = GATv3(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model




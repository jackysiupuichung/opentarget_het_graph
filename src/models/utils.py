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
    use_rte: bool = False,
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
        use_rte: Enable Relative Temporal Encoding (HGT only)
        
    Returns:
        differentiable PyTorch model
    """
    node_types, edge_types = get_metadata(data)
    metadata = (node_types, edge_types)
    
    # Extract input dimensions for each node type
    in_channels = {}
    for nt in node_types:
        if hasattr(data[nt], 'x') and data[nt].x is not None:
            in_channels[nt] = data[nt].x.size(1)
        else:
            # Fallback or specific handling for nodes without features
            # Usually strict error or defaulting is better. 
            # Assuming all have features for now based on 'hetero_graph_with_features'
            # If missing, we might use Embedding. But let's assume features.
            # Default to hidden_dim if not found (risky but maintains old behavior if no features)
            in_channels[nt] = hidden_dim

    if model_name == 'hgt':
        model = HGTLinkPredictor(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            node_types=node_types,
            metadata=metadata,
            dropout=dropout,
            use_rte=use_rte,
        )
    elif model_name == 'gatv2':
        model = GATv2(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            metadata=metadata,
            dropout=dropout
        )
    elif model_name == 'gatv3':
        model = GATv3(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            metadata=metadata,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model




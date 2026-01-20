#!/usr/bin/env python3
"""
Model utilities for building and managing HGT models.
"""

import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple
from .hgt import HGTLinkPredictor


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


def build_hgt_model(
    data: HeteroData,
    hidden_dim: int = 128,
    out_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> HGTLinkPredictor:
    """
    Build HGT model from HeteroData.
    
    Args:
        data: HeteroData object
        hidden_dim: Hidden dimension
        out_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of HGT layers
        dropout: Dropout rate
        
    Returns:
        HGTLinkPredictor model
    """
    node_types, edge_types = get_metadata(data)
    metadata = (node_types, edge_types)
    
    model = HGTLinkPredictor(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        node_types=node_types,
        metadata=metadata,
        dropout=dropout,
    )
    
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

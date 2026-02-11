#!/usr/bin/env python3
"""
Heterogeneous Graph Transformer (HGT) for link prediction.

This module implements HGT encoder and link predictor for heterogeneous graphs
with relation::datasource level edges and scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hgt_conv_rte import HGTConv
from torch_geometric.nn import Linear
from typing import Dict, List, Tuple, Optional
from .decoder import DualHeadDecoder


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer encoder.
    
    Stacks multiple HGTConv layers to learn node embeddings.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        node_types: List[str],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        dropout: float = 0.1,
        use_rte: bool = False,
    ):
        """
        Initialize HGT encoder.
        
        Args:
            in_channels: Dictionary of node type -> input dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            num_heads: Number of attention heads
            num_layers: Number of HGT layers
            node_types: List of node type names
            metadata: (node_types, edge_types) tuple
            dropout: Dropout rate
            use_rte: Enable Relative Temporal Encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.node_types = node_types
        self.use_rte = use_rte
        
        # Input projection layer
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_dim)
        
        # HGT convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
                use_RTE=use_rte,
            )
            self.convs.append(conv)
        
        # Layer normalization for each node type
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_dim)
                for node_type in node_types
            })
            self.norms.append(norm_dict)
        
        self.dropout = dropout
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_time_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_dict: Node features {node_type: features}
            edge_index_dict: Edge indices {edge_type: edge_index}
            
        Returns:
            Node embeddings {node_type: embeddings}
        """
        # Project inputs to hidden_dim
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                x_dict_proj[node_type] = self.lin_dict[node_type](x)
            else:
                x_dict_proj[node_type] = x # Should not happen if in_channels is correct
        
        x_dict = x_dict_proj

        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            # HGT convolution with optional temporal encoding
            x_dict = conv(x_dict, edge_index_dict, edge_time_diff_dict=edge_time_dict)
            
            # Layer norm + dropout
            x_dict = {
                node_type: F.dropout(
                    self.norms[i][node_type](x),
                    p=self.dropout,
                    training=self.training
                )
                for node_type, x in x_dict.items()
            }
        
        return x_dict


class HGTLinkPredictor(nn.Module):
    """
    HGT-based link predictor.
    
    Combines HGT encoder with dot product decoder for link prediction.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_heads: int,
        num_layers: int,
        node_types: List[str],
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        dropout: float = 0.1,
        use_rte: bool = False,
    ):
        """
        Initialize link predictor.
        
        Args:
            in_channels: Input feature dimensions
            hidden_dim: Hidden dimension
            out_dim: Output dimension
            use_rte: Enable Relative Temporal Encoding
            ...
        """
        super().__init__()
        
        self.encoder = HGT(
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
        
        # Decoder input dim must match encoder output (hidden_dim)
        self.decoder = DualHeadDecoder(hidden_dim)
    
    def encode(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_time_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode nodes to embeddings.
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            
        Returns:
            Node embeddings
        """
        return self.encoder(x_dict, edge_index_dict, edge_time_dict)
    
    def decode(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode link scores using DualHeadDecoder.
        
        Args:
            z_src: Source node embeddings [num_edges, hidden_dim]
            z_dst: Destination node embeddings [num_edges, hidden_dim]
            
        Returns:
            Dict containing 'logits_exist' and 'logits_prob'
        """
        return self.decoder(z_src, z_dst)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: torch.Tensor,
        src_type: str,
        dst_type: str,
        edge_time_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for link prediction.
        
        Args:
            x_dict: Node features
            edge_index_dict: Edge indices
            edge_label_index: Edges to predict [2, num_edges]
            src_type: Source node type
            dst_type: Destination node type
            
        Returns:
            Dict with 'logits_exist' and 'logits_prob'
        """
        # Encode all nodes
        z_dict = self.encode(x_dict, edge_index_dict, edge_time_dict)
        
        # Get embeddings for edges to predict
        z_src = z_dict[src_type][edge_label_index[0]]
        z_dst = z_dict[dst_type][edge_label_index[1]]
        
        # Decode link scores
        return self.decode(z_src, z_dst)

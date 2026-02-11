import torch
import torch.nn as nn
from torch.nn import Linear

class DualHeadDecoder(nn.Module):
    """
    Dual-head decoder for Multi-Task Link Prediction.
    
    Head A: Existence (Binary)
    Head B: Probability (Regression)
    
    Uses concatenation of node embeddings followed by reverse pyramid MLP.
    Architecture: [2*in_channels] -> [in_channels] -> [in_channels//2] -> [1]
    """
    def __init__(self, in_channels=-1, dropout=0.1):
        super().__init__()
        
        # Shared MLP backbone (reverse pyramid)
        self.mlp = nn.Sequential(
            Linear(2 * in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for existence and probability
        self.lin_exist = Linear(in_channels // 2, 1)
        self.lin_prob = Linear(in_channels // 2, 1)

    def forward(self, z_src, z_dst):
        # Concatenate source and destination embeddings
        edge_feat = torch.cat([z_src, z_dst], dim=-1)
        
        # Pass through shared MLP
        hidden = self.mlp(edge_feat)
        
        # Separate heads
        logits_exist = self.lin_exist(hidden).squeeze(-1)
        logits_prob = self.lin_prob(hidden).squeeze(-1)
        
        return {
            'logits_exist': logits_exist,
            'logits_prob': logits_prob
        }

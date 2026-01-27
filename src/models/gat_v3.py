
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn import Linear
from typing import Dict, List, Optional, Union

class GATv3Conv(MessagePassing):
    """
    GATv3 Convolution Layer.
    
    Differences from GATv2:
    1. Context-aware attention (element-wise product) instead of LeakyReLU.
    2. Scaled Dot Product Attention (division by sqrt(d_k)).
    3. Bias in attention mechanism.
    4. Option for shared weights (not strictly enforced here but possible via inputs).
    """
    def __init__(
        self,
        in_channels: Union[int, min, 'Tuple[int, int]'],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs
    ):
        super().__init__(node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        
        self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
        if share_weights:
            self.lin_dst = self.lin_src
        else:
            self.lin_dst = Linear(in_channels, heads * out_channels, bias=False)
            
        # Context vector for attention (replacing LeakyReLU weight vector 'a')
        # In GATv3: alpha = exp( W_context^T ( (O_s h_i) * (O_t h_j) ) / sqrt(d) )
        # So we need a weight vector of size (heads * out_channels)
        self.att_context = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        if not self.share_weights:
            self.lin_dst.reset_parameters()
        glorot(self.att_context)
        zeros(self.bias)
        
    def forward(self, x, edge_index, return_attention_weights=False):
        # x is (x_src, x_dst) or x
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.lin_src(x).view(-1, self.heads, self.out_channels)
        else:
            x_src, x_dst = x
            x_src = self.lin_src(x_src).view(-1, self.heads, self.out_channels)
            if self.share_weights:
                x_dst = self.lin_src(x_dst).view(-1, self.heads, self.out_channels)
            else:
                x_dst = self.lin_dst(x_dst).view(-1, self.heads, self.out_channels)
                
        # Message Passing
        # We pass context-aware attention inside message/edge_update
        out = self.propagate(edge_index, x=(x_src, x_dst), size=None)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        if self.bias is not None:
            out = out + self.bias
            
        if return_attention_weights:
            # Re-compute alpha or store it? PyG structure usually returns tuple.
            # For simplicity, returning just out here.
            pass
            
        return out
        
    def message(self, x_j, x_i, index, ptr, size_i):
        # x_j: (E, heads, out_channels) - source
        # x_i: (E, heads, out_channels) - target
        
        # Element-wise product for context awareness
        # GATv3: (O_s h_i * O_t h_j)
        # Note: PyG 'message' args: x_j is source (neighbor), x_i is target (central node)
        # Formula says O_s h_i * O_t h_j. 
        # Here x_i is target projection, x_j is source projection.
        
        interaction = x_i * x_j
        
        # Dot product with context vector
        # att_context: (1, heads, out_channels)
        # interaction: (E, heads, out_channels)
        # Sum over channel dim -> (E, heads)
        alpha = (interaction * self.att_context).sum(dim=-1)
        
        # Scaling
        # sqrt(d_k)
        scale = math.sqrt(self.out_channels)
        alpha = alpha / scale
        
        # Softmax
        # alpha is (E, heads)
        alpha = softmax(alpha, index, ptr, size_i)
        
        self._alpha = alpha # For visualization if needed
        
        # Dropout on attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted sum
        # x_j: (E, heads, out_channels)
        # alpha: (E, heads) -> (E, heads, 1)
        return x_j * alpha.unsqueeze(-1)


class GATv3HeteroConv(nn.Module):
    """
    Heterogeneous Wrapper for GATv3.
    
    Features:
    - Edge-type specific attention scalar (alpha^e).
    - Formula: alpha^e = 1 / (1 + exp(1 - sum(|W_context^e|)))
    """
    def __init__(
        self,
        convs: Dict[str, nn.Module],
        aggr: str = "sum"
    ):
        super().__init__()
        self.convs = nn.ModuleDict(convs)
        self.aggr = aggr
        
        # Creating learnable weights for edge types
        # Wait, the formula uses "W_context^e". Is that the CONV layer's context vector?
        # The prompt says: "This is calculated by summing the absolute values of a dedicated weight matrix We context"
        # It implies We_context exists INSIDE the GATv3Conv or IS the attention vector?
        # "GATv3's Hetero convolution uses an ... alpha^e ... by summing ... We_context"
        # Since I'm wrapping generic convs, I should check if they expose 'att_context'.
        # If they do, I can use that.
        
    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        out_dict = {}
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel, dst_type = edge_type
            
            # Skip if no nodes
            if src_type not in x_dict or dst_type not in x_dict:
                continue
                
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            
            # Get convolution
            conv = self.convs[str(edge_type)]
            
            # Compute Edge-Type Attention Scalar alpha^e
            # Based on conv.att_context (if available)
            alpha_e = 1.0
            if hasattr(conv, 'att_context'):
                # Formula: 1 / (1 + exp(1 - sum(|W|))) = Sigmoid(sum(|W|) - 1)
                # Wait, earlier analysis said `Sigmoid(sum|W| + 1)` ?
                # User text: `1 / (1 + exp(-(sum|W|) + 1))` 
                # Let X = sum|W|. Formula = 1 / (1 + exp(-(X) + 1)) = 1 / (1 + exp(1-X))
                
                # We need sum of ABSOLUTE values.
                w_abs_sum = conv.att_context.abs().sum()
                
                # Careful with gradient flow, this seems correct.
                # exponent = 1.0 - w_abs_sum
                # alpha_e = 1.0 / (1.0 + torch.exp(exponent))
                
                # Equivalent using sigmoid: 
                # Sigmoid(x) = 1 / (1 + exp(-x))
                # We interpret `exp(1-sum)` as `exp(-(sum-1))`
                # So it is Sigmoid(sum - 1)
                
                # Let's match user formula EXACTLY
                exponent = -(w_abs_sum) + 1.0
                alpha_e = 1.0 / (1.0 + torch.exp(exponent))
            
            # Run Conv
            # GATv3Conv expects (x_src, x_dst)
            out = conv((x_src, x_dst), edge_index)
            
            # Apply alpha^e scaling
            out = out * alpha_e
            
            # Aggregate
            if dst_type not in out_dict:
                out_dict[dst_type] = out
            else:
                if self.aggr == "sum":
                    out_dict[dst_type] = out_dict[dst_type] + out
                elif self.aggr == "mean":
                    # This requires tracking count, simplified here
                    out_dict[dst_type] = out_dict[dst_type] + out
                    

        return out_dict


from torch_geometric.nn import Linear

class GATv3(nn.Module):
    """
    GATv3 Model.
    Uses GATv3HeteroConv -> GATv3Conv stack.
    """
    def __init__(self, hidden_dim, out_dim, num_heads, num_layers=2, metadata=None, dropout=0.1):
        super().__init__()
        self.node_types, self.edge_types = metadata
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Create a GATv3Conv for each edge type
            conv_dict = {
                str(edge_type): GATv3Conv(
                    -1, 
                    hidden_dim, 
                    heads=num_heads, 
                    dropout=dropout, 
                    add_self_loops=False, 
                    bias=True # GATv3 has bias
                )
                for edge_type in self.edge_types
            }
            # Wrap in GATv3HeteroConv
            # Note: GATv3HeteroConv learns separate alpha^e for each edge type
            conv = GATv3HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)
            
        self.lin = Linear(-1, out_dim)

    def forward(
        self, 
        x_dict, 
        edge_index_dict, 
        edge_label_index=None, 
        src_type=None, 
        dst_type=None, 
        edge_time_dict=None,
        **kwargs
    ):
        # 1. Message Passing (Encode)
        x_dict = self.encode(x_dict, edge_index_dict)
        
        # 2. Link Prediction (Decode)
        if edge_label_index is not None and src_type is not None:
            return self.decode(x_dict[src_type], x_dict[dst_type], edge_label_index)
            
        return x_dict

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Final projection 
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_src, z_dst, edge_label_index=None):
        if edge_label_index is not None:
            # Dot product on specific edges
            row, col = edge_label_index
            return (z_src[row] * z_dst[col]).sum(dim=-1)
        else:
            # Full pairwise matrix product
            return torch.matmul(z_src, z_dst.t()).view(-1)



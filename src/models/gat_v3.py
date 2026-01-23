import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear



class GATv3(nn.Module):
    """
    Static GATv3 model. 
    Placeholder distinction: maybe more heads or different aggregation?
    For now, implementing same as GATv2 but separate class.
    """
    def __init__(self, hidden_dim, out_dim, num_heads, num_layers=2, metadata=None, dropout=0.1):
        super().__init__()
        self.node_types, self.edge_types = metadata
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv(-1, hidden_dim, heads=num_heads, dropout=dropout, add_self_loops=False)
                for edge_type in self.edge_types
            }, aggr='mean') # Different aggregation for variety?
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
        # 1. Message Passing
        x_dict = self.encode(x_dict, edge_index_dict)
        
        # 2. Link Prediction
        if edge_label_index is not None and src_type is not None:
            return self.decode(x_dict[src_type], x_dict[dst_type], edge_label_index)
            
        return x_dict

    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        # Final projection (Added strictly for consistency with GATv2 logic)
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_src, z_dst, edge_label_index=None):
        if edge_label_index is not None:
            row, col = edge_label_index
            return (z_src[row] * z_dst[col]).sum(dim=-1)
        else:
            return torch.matmul(z_src, z_dst.t()).view(-1)

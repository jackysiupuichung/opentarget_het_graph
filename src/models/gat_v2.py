import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

class GATv2(nn.Module):
    """
    Static GATv2 model wrapped in HeteroConv.
    """
    def __init__(self, hidden_dim, out_dim, num_heads, num_layers=2, metadata=None, dropout=0.1):
        super().__init__()
        self.node_types, self.edge_types = metadata
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv(-1, hidden_dim, heads=num_heads, dropout=dropout, add_self_loops=False)
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)
            
        self.lin = Linear(-1, out_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        for conv in self.convs:
            if edge_attr_dict:
                x_dict = conv(x_dict, edge_index_dict, edge_attr=edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        return x_dict



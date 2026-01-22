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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        for conv in self.convs:
            if edge_attr_dict:
                x_dict = conv(x_dict, edge_index_dict, edge_attr=edge_attr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        return x_dict

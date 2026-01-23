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

    def forward(
        self, 
        x_dict, 
        edge_index_dict, 
        edge_label_index=None, 
        src_type=None, 
        dst_type=None, 
        edge_time_dict=None, # Ignored in static GAT
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
            # GATv2Conv typically supports edge_attr but tricky with HeteroConv dictionary
            # For simplicity in this static benchmark, we ignore edge_attr for now
            # or would need to pass edge_attr_dict matching edge_index_dict keys
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Final projection 
        x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        return x_dict

    def decode(self, z_src, z_dst, edge_label_index=None):
        if edge_label_index is not None:
            # Dot product on specific edges
            row, col = edge_label_index
            return (z_src[row] * z_dst[col]).sum(dim=-1)
        else:
            # Full pairwise dot product (for evaluation rankings)
            # z_src: [M, D], z_dst: [N, D] -> [M, N] -> flattened?
            # Evaluator typically handles the looping. 
            # If we return a matrix, the evaluator needs to handle it.
            # But standard 'decode' usually does dot product on provided indices.
            # If no indices, we might return matrix product.
            return torch.matmul(z_src, z_dst.t()).view(-1)



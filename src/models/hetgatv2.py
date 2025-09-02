import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv


class HetGATv2(nn.Module):
    """
    Heterogeneous GATv2-based recommender for target-disease link prediction.
    Uses scalar edge scores (edge_dim=1) as edge attributes in attention.
    """

    def __init__(
        self,
        metadata,
        hidden_dim,
        num_layers,
        heads,
        num_nodes,
        embedding_dim,
        pair_src_type,
        pair_dst_type,
        pair_mlp_hidden,
        dropout=0.2,
        pretrained_embeddings=None,
    ):
        super().__init__()
        self.metadata = metadata
        self.pair_src_type = pair_src_type
        self.pair_dst_type = pair_dst_type

        # === Node embeddings ===
        self.embeddings = nn.ModuleDict()
        for ntype in metadata[0]:
            if pretrained_embeddings and ntype in pretrained_embeddings:
                emb_tensor = torch.tensor(pretrained_embeddings[ntype], dtype=torch.float)
                self.embeddings[ntype] = nn.Embedding.from_pretrained(emb_tensor, freeze=False)
            else:
                self.embeddings[ntype] = nn.Embedding(num_nodes[ntype], embedding_dim)

        # === Graph encoder (stack of HetGATv2Conv layers) ===
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    (src, rel, dst): GATv2Conv(
                        in_channels=(embedding_dim, embedding_dim),
                        out_channels=hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        edge_dim=1,  # scalar edge score
                        # TODO: graph must have self loops in some relations
                        add_self_loops=False,
                    )
                    for src, rel, dst in metadata[1]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # === Pairwise head ===
        layers = []
        input_dim = 2 * hidden_dim
        for h in pair_mlp_hidden:
            layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.pair_mlp = nn.Sequential(*layers)

    def forward(self, x_dict, edge_index_dict, pairs, edge_attr_dict=None):
        # === Initialize node features ===
        h_dict = {}
        for ntype in self.metadata[0]:
            if ntype in x_dict and x_dict[ntype] is not None:
                h_dict[ntype] = x_dict[ntype]
            else:
                ids = torch.arange(
                    self.embeddings[ntype].num_embeddings,
                    device=next(self.parameters()).device,
                )
                h_dict[ntype] = self.embeddings[ntype](ids)

        # === Pass through stacked GATv2 layers ===
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict, edge_attr_dict)

        # === Select embeddings for prediction pairs ===
        src_ids, dst_ids = pairs
        src_emb = h_dict[self.pair_src_type][src_ids]
        dst_emb = h_dict[self.pair_dst_type][dst_ids]

        # === Pairwise scoring (logits) ===
        z = torch.cat([src_emb, dst_emb], dim=-1)
        out = self.pair_mlp(z)
        return out.squeeze(-1)  # logits

import torch
from torch_geometric.data import HeteroData

print("Loading graph...")
data = torch.load('/Users/pui.chungsiu/Documents/opentarget_het_graph/output/graph/hetero_graph_with_features.pt')

print("Node Types:", data.node_types)
for nt in data.node_types:
    print(f"\nNode Type: {nt}")
    if hasattr(data[nt], 'node_id'):
        print(f"  Has node_id: Yes ({len(data[nt].node_id)} IDs)")
        print(f"  Example: {data[nt].node_id[:5]}")
    else:
        print("  Has node_id: NO")

    if hasattr(data[nt], 'num_nodes'):
        print(f"  Num nodes: {data[nt].num_nodes}")

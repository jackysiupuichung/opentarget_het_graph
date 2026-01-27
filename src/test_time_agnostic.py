import torch
from torch_geometric.data import HeteroData
from data.temporal_loader import to_time_agnostic

def test_to_time_agnostic():
    print("🧪 Testing to_time_agnostic...")
    
    # 1. Create dummy temporal graph
    data = HeteroData()
    
    # Nodes
    num_src = 10
    num_dst = 10
    data['src'].num_nodes = num_src
    data['dst'].num_nodes = num_dst
    
    # Edges: Create duplicate edges with different times/weights
    # Edge 0: (0, 0) at t=2020, w=0.5
    # Edge 1: (0, 0) at t=2021, w=0.8 -> Max should be 0.8
    # Edge 2: (1, 1) at t=2020, w=0.2
    
    src = [0, 0, 1]
    dst = [0, 0, 1]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    edge_time = torch.tensor([2020, 2021, 2020], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.8, 0.2], dtype=torch.float)
    
    data['src', 'rel', 'dst'].edge_index = edge_index
    data['src', 'rel', 'dst'].edge_time = edge_time
    data['src', 'rel', 'dst'].edge_weight = edge_weight
    
    print("\nOriginal Graph:")
    print(f"  Edges: {data['src', 'rel', 'dst'].edge_index.size(1)}")
    print(f"  Weights: {data['src', 'rel', 'dst'].edge_weight}")
    
    # 2. Run to_time_agnostic
    new_data = to_time_agnostic(data)
    
    # 3. Validation
    # Expect 2 edges: (0,0) and (1,1)
    new_edges = new_data['src', 'rel', 'dst'].edge_index
    new_weights = new_data['src', 'rel', 'dst'].edge_weight
    
    print("\n collapsed Graph:")
    print(f"  Edges: {new_edges.size(1)}")
    print(f"  Weights: {new_weights}")
    
    assert new_edges.size(1) == 2, f"Expected 2 edges, got {new_edges.size(1)}"
    
    # Check weights
    # (0,0) should have weight 0.8
    # (1,1) should have weight 0.2
    
    # Find index of (0,0)
    # Since coalesce sorts by row then col usually
    # (0,0) is likely index 0
    
    mask00 = (new_edges[0] == 0) & (new_edges[1] == 0)
    w00 = new_weights[mask00].item()
    assert abs(w00 - 0.8) < 1e-6, f"Expected weight 0.8 for (0,0), got {w00}"
    
    mask11 = (new_edges[0] == 1) & (new_edges[1] == 1)
    w11 = new_weights[mask11].item()
    assert abs(w11 - 0.2) < 1e-6, f"Expected weight 0.2 for (1,1), got {w11}"
    
    # Check time removal
    assert 'edge_time' not in new_data['src', 'rel', 'dst'], "edge_time should be removed"
    
    print("\n✅ test_to_time_agnostic PASSED")

if __name__ == "__main__":
    test_to_time_agnostic()

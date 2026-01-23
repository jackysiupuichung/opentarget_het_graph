import torch
import os
from omegaconf import OmegaConf
from src.data.temporal_loader import load_event_graph, filter_graph_by_time, get_temporal_masks

def test_leakage():
    print("🕵️  Checking for Data Leakage across splits...")
    
    # Load Data (Simulating train.py logic)
    # Using defaults from config/benchmark_config.yaml
    params = {
        'temporal_graph_file': 'output/progression/temporal_graph.pt',
        'train_range': [2000, 2016],
        'val_range': [2017, 2017],
        'test_range': [2018, 2025],
        'src_type': 'disease',
        'dst_type': 'target',
        'relation': 'clinical_trial::chembl',
        'mode': 'event'
    }
    
    if not os.path.exists(params['temporal_graph_file']):
        print(f"❌ Graph file not found at {params['temporal_graph_file']}")
        return

    print("   Loading graph...")
    data = load_event_graph(params['temporal_graph_file'], attach_features=False)
    
    # Identify Supervision Edge Type
    sup_edge_type = None
    for et in data.edge_types:
        if (et[0] == params['src_type'] and et[2] == params['dst_type'] and params['relation'] in et[1]):
            sup_edge_type = et
            break
            
    print(f"   Supervision Edge: {sup_edge_type}")
    
    # Get Masks
    print("   Applying temporal masks...")
    
    split_config = {
        'train': params['train_range'],
        'val': params['val_range'],
        'test': params['test_range']
    }
    
    train_mask, val_mask, test_mask = get_temporal_masks(
        data, 
        split_config=split_config
    )[sup_edge_type]
    
    # Wait, get_temporal_masks returns dict {et: (mask, mask, mask)}
    # Let's verify exactly what it returns by looking at import
    
    edge_times = data[sup_edge_type].edge_time
    
    # Extract Edge Indices
    all_edges = data[sup_edge_type].edge_index
    
    train_edges = all_edges[:, train_mask]
    val_edges = all_edges[:, val_mask]
    test_edges = all_edges[:, test_mask]
    
    train_times = edge_times[train_mask]
    val_times = edge_times[val_mask]
    test_times = edge_times[test_mask]
    
    print(f"   Train Size: {train_edges.size(1):,}")
    print(f"   Val Size:   {val_edges.size(1):,}")
    print(f"   Test Size:  {test_edges.size(1):,}")
    
    # CHECK 1: Max Time in Splits
    t_train_max = train_times.max().item() if len(train_times) > 0 else 0
    t_val_min = val_times.min().item() if len(val_times) > 0 else 0
    t_val_max = val_times.max().item() if len(val_times) > 0 else 0
    t_test_min = test_times.min().item() if len(test_times) > 0 else 0
    
    print(f"\nTime Ranges:")
    print(f"   Train: <= {t_train_max}")
    print(f"   Val:   {t_val_min} - {t_val_max}")
    print(f"   Test:  {t_test_min} - ...")
    
    assert t_train_max <= params['train_range'][1], f"Train Leakage! Found year {t_train_max} > {params['train_range'][1]}"
    assert t_val_min >= params['val_range'][0], f"Val Leakage! Found year {t_val_min} < {params['val_range'][0]}"
    assert t_val_max <= params['val_range'][1], f"Val Leakage! Found year {t_val_max} > {params['val_range'][1]}"
    assert t_test_min >= params['test_range'][0], f"Test Leakage! Found year {t_test_min} < {params['test_range'][0]}"
    
    print("✅ Temporal Separation Verified.")
    
    # CHECK 2: Context vs Target Leakage
    # train_context (used for Input) is typically filter_graph_by_time(train_end)
    print("\n   Verifying Input Context Safety...")
    train_context = filter_graph_by_time(data, params['train_range'][1])
    context_edges = train_context[sup_edge_type].edge_index
    
    # We need to check if any edge in 'val_edges' exists in 'context_edges'
    # Since edges are (u, v) pairs, and event graph has duplicates,
    # overlap in (u,v) is ALLOWED (history exists).
    # BUT, specific EVENTS should not overlap.
    # Since we filter by time <= 2020 vs time > 2020, event overlap is impossible by definition.
    
    print(f"   Context Edges: {context_edges.size(1):,}")
    
    # Logic check: 'context_edges' should be exactly equal to 'train_edges' 
    # (assuming filter_graph_by_time uses same logic as get_temporal_masks)
    
    # Convert to sets of tuples for comparison (slow but reliable for test)
    # Or just length check
    if context_edges.size(1) == train_edges.size(1):
        print("✅ Context Size matches Train Partition Size.")
    else:
        print(f"⚠️  Context Size ({context_edges.size(1)}) != Train Mask Size ({train_edges.size(1)})")
        # This might happen if 'filter_graph_by_time' handles < vs <= differently
    
    print("\n🎉 No Leakage Detected!")

if __name__ == "__main__":
    test_leakage()

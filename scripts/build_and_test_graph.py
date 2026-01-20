#!/usr/bin/env python3
"""
Build static relation-level graph and test the pipeline.

This script:
1. Builds relation-level progression graph (no datasource, no scores)
2. Creates HeteroData object for static graph
3. Tests the complete training pipeline
"""

import os
import sys
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.graph_builder import load_edges, build_hetero_graph
from src.data.utils import temporal_split, attach_node_features
from src.models.utils import build_hgt_model, count_parameters


def main():
    print("\n" + "="*80)
    print("BUILDING STATIC RELATION-LEVEL GRAPH")
    print("="*80 + "\n")
    
    # ============================================================
    # 1. Load configuration
    # ============================================================
    config_path = "config/experiments/hgt_static_relation.yaml"
    print(f"📄 Loading config from {config_path}")
    
    cfg = OmegaConf.load(config_path)
    # Load base config and merge
    base_cfg = OmegaConf.load("config/benchmark_config.yaml")
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    print(f"✅ Configuration loaded")
    print(f"   Granularity: {cfg.data.graph.edge_granularity}")
    print(f"   Mode: {cfg.data.graph.mode}")
    
    # ============================================================
    # 2. Load edges
    # ============================================================
    print(f"\n📂 Loading edges from {cfg.data.edge_dir}...")
    all_edges = load_edges(cfg.data.edge_dir)
    
    print(f"✅ Loaded {len(all_edges)} total edges")
    print(f"   Unique relations: {all_edges['relation'].nunique()}")
    if 'datasourceId' in all_edges.columns:
        print(f"   Unique datasources: {all_edges['datasourceId'].nunique()}")
    
    # ============================================================
    # 3. Filter edges by cutoff year (static graph)
    # ============================================================
    cutoff_year = cfg.data.temporal_split.cutoff_year
    print(f"\n⏰ Filtering edges for static graph (year <= {cutoff_year})...")
    
    graph_edges = all_edges[all_edges["year"] <= cutoff_year].copy()
    print(f"✅ Filtered to {len(graph_edges)} edges")
    
    # ============================================================
    # 5. Get supervision edges for temporal split
    # ============================================================
    print(f"\n🎯 Extracting supervision edges...")
    supervision_edges = all_edges[
        (all_edges["source_type"] == cfg.data.graph.supervision.src_type) &
        (all_edges["target_type"] == cfg.data.graph.supervision.dst_type)
    ].copy()
    
    print(f"✅ Found {len(supervision_edges)} supervision edges")
    print(f"   ({cfg.data.graph.supervision.src_type} → {cfg.data.graph.supervision.dst_type})")
    
    # Temporal split
    train_edges, val_edges, test_edges = temporal_split(
        supervision_edges,
        cutoff_year=cfg.data.temporal_split.cutoff_year,
        horizon=cfg.data.temporal_split.horizon,
        val_years=cfg.data.temporal_split.val_years,
    )
    
    # ============================================================
    # 6. Build HeteroData
    # ============================================================
    print(f"\n🔨 Building HeteroData (granularity: {cfg.data.graph.edge_granularity})...")
    
    hetero_data, id_maps = build_hetero_graph(
        edges=graph_edges,
        edge_granularity=cfg.data.graph.edge_granularity,
    )
    
    print(f"\n✅ Built HeteroData:")
    print(f"   Node types ({len(hetero_data.node_types)}): {hetero_data.node_types}")
    print(f"   Edge types ({len(hetero_data.edge_types)}): {hetero_data.edge_types[:5]}...")
    
    # Print node counts
    print(f"\n📊 Node counts:")
    for node_type in hetero_data.node_types:
        print(f"   {node_type}: {hetero_data[node_type].num_nodes}")
    
    # Print edge counts
    print(f"\n🔗 Edge counts (top 10):")
    edge_counts = []
    for edge_type in hetero_data.edge_types:
        if hasattr(hetero_data[edge_type], 'edge_index'):
            count = hetero_data[edge_type].edge_index.size(1)
            edge_counts.append((edge_type, count))
    
    edge_counts.sort(key=lambda x: x[1], reverse=True)
    for edge_type, count in edge_counts[:10]:
        print(f"   {edge_type}: {count}")
    
    # ============================================================
    # 7. Attach node features
    # ============================================================
    print(f"\n🎨 Initializing node features...")
    
    hetero_data = attach_node_features(
        hetero_data,
        id_maps,
        init_method=cfg.model.node_features.init_method,
        embedding_dim=cfg.model.node_features.embedding_dim,
        seed=cfg.train.seed,
    )
    
    print(f"\n✅ Node features initialized:")
    for node_type in hetero_data.node_types:
        if hasattr(hetero_data[node_type], 'x'):
            print(f"   {node_type}: {hetero_data[node_type].x.shape}")
    
    # ============================================================
    # 8. Build HGT model
    # ============================================================
    print(f"\n🏗️ Building HGT model...")
    
    model = build_hgt_model(
        hetero_data,
        hidden_dim=cfg.model.hgt.hidden_dim,
        out_dim=cfg.model.hgt.hidden_dim,
        num_heads=cfg.model.hgt.num_heads,
        num_layers=cfg.model.hgt.num_layers,
        decoder_type=cfg.model.decoder.type,
        dropout=cfg.model.hgt.dropout,
    )
    
    # Initialize lazy parameters with a dummy forward pass
    print(f"🔧 Initializing model parameters...")
    with torch.no_grad():
        _ = model.encode(hetero_data.x_dict, hetero_data.edge_index_dict)
    
    num_params = count_parameters(model)
    print(f"✅ Built HGT model:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Hidden dim: {cfg.model.hgt.hidden_dim}")
    print(f"   Num heads: {cfg.model.hgt.num_heads}")
    print(f"   Num layers: {cfg.model.hgt.num_layers}")
    print(f"   Decoder: {cfg.model.decoder.type}")
    
    # ============================================================
    # 9. Test forward pass
    # ============================================================
    print(f"\n🧪 Testing forward pass...")
    
    # Get supervision edge type
    supervision_edge_type = (
        cfg.data.graph.supervision.src_type,
        cfg.data.graph.supervision.relation,
        cfg.data.graph.supervision.dst_type,
    )
    
    # Create dummy edge_label_index for testing
    num_test_edges = 10
    src_nodes = torch.randint(0, hetero_data[cfg.data.graph.supervision.src_type].num_nodes, (num_test_edges,))
    dst_nodes = torch.randint(0, hetero_data[cfg.data.graph.supervision.dst_type].num_nodes, (num_test_edges,))
    edge_label_index = torch.stack([src_nodes, dst_nodes], dim=0)
    
    # Forward pass
    with torch.no_grad():
        scores = model(
            x_dict=hetero_data.x_dict,
            edge_index_dict=hetero_data.edge_index_dict,
            edge_label_index=edge_label_index,
            src_type=cfg.data.graph.supervision.src_type,
            dst_type=cfg.data.graph.supervision.dst_type,
        )
    
    print(f"✅ Forward pass successful!")
    print(f"   Input edges: {edge_label_index.shape}")
    print(f"   Output scores: {scores.shape}")
    print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # ============================================================
    # 10. Save graph for reuse
    # ============================================================
    graph_save_path = "output/hetero_graph_static_relation.pt"
    print(f"\n💾 Saving graph to {graph_save_path}...")
    
    os.makedirs(os.path.dirname(graph_save_path), exist_ok=True)
    torch.save((hetero_data, id_maps), graph_save_path)
    
    print(f"✅ Graph saved!")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Graph built successfully!")
    print(f"   Nodes: {sum(data.num_nodes for data in hetero_data.node_stores)}")
    print(f"   Edges: {sum(data.edge_index.size(1) for data in hetero_data.edge_stores if hasattr(data, 'edge_index'))}")
    print(f"   Node types: {len(hetero_data.node_types)}")
    print(f"   Edge types: {len(hetero_data.edge_types)}")
    print(f"\n   Model parameters: {num_params:,}")
    print(f"   Saved to: {graph_save_path}")
    print(f"\n✅ Ready for training!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

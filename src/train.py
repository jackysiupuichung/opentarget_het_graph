#!/usr/bin/env python3
"""
Main training script for HGT link prediction on heterogeneous graphs.

This script implements the complete pipeline:
1. Load edges from parquet files
2. Temporal split (train/val/test)
3. Build HeteroData (relation::source level)
4. Initialize HGT model
5. Train with PyTorch Lightning
6. Evaluate with ranking metrics
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.graph_builder import load_edges, build_hetero_graph
from data.utils import temporal_split, attach_node_features
from models.utils import build_hgt_model, count_parameters
from models.base_lightning import HGTRecLightning
from benchmark.evaluator import Evaluator


def main(config_path: str):
    """
    Main training pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    print("\n" + "="*80)
    print("HGT LINK PREDICTION TRAINING")
    print("="*80 + "\n")
    
    # ============================================================
    # 1. Load Configuration
    # ============================================================
    print(f"📄 Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Load base config if using experiments
    if "defaults" in cfg:
        # Get project root (parent of src/)
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_config_path = os.path.join(project_root, "config/benchmark_config.yaml")
        base_cfg = OmegaConf.load(base_config_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
    
    print(f"✅ Configuration loaded")
    print(f"   Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"   Edge dir: {cfg.data.edge_dir}")
    
    # ============================================================
    # 2. Load Edges
    # ============================================================
    # Resolve edge_dir relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    edge_dir = os.path.join(project_root, cfg.data.edge_dir)
    
    print(f"\n📂 Loading edges from {edge_dir}...")
    all_edges = load_edges(edge_dir)
    
    if all_edges.empty:
        print(f"\n❌ ERROR: No edges found in {edge_dir}")
        print(f"   Please check that edge parquet files exist in this directory.")
        return
    
    print(f"✅ Loaded {len(all_edges)} total edges")
    print(f"   Unique relations: {all_edges['relation'].nunique()}")
    print(f"   Unique datasources: {all_edges['datasourceId'].nunique()}")
    
    # ============================================================
    # 3. Temporal Split
    # ============================================================
    print(f"\n⏰ Temporal splitting...")
    
    # Filter to cutoff year for graph
    cutoff_year = cfg.data.temporal_split.cutoff_year
    graph_edges = all_edges[all_edges["year"] <= cutoff_year].copy()
    print(f"✅ Graph edges (year <= {cutoff_year}): {len(graph_edges)}")
    
    # Get supervision edges
    supervision_edges = all_edges[
        (all_edges["source_type"] == cfg.data.graph.supervision.src_type) &
        (all_edges["target_type"] == cfg.data.graph.supervision.dst_type) &
        (all_edges["relation"] == cfg.data.graph.supervision.relation)
    ].copy()
    
    print(f"✅ Supervision edges: {len(supervision_edges)}")
    
    # Temporal split
    train_edges, val_edges, test_edges = temporal_split(
        supervision_edges,
        cutoff_year=cfg.data.temporal_split.cutoff_year,
        horizon=cfg.data.temporal_split.horizon,
        val_years=cfg.data.temporal_split.val_years,
    )
    
    # ============================================================
    # 4. Build HeteroData
    # ============================================================
    print(f"\n🔨 Building HeteroData (relation::source level)...")
    
    hetero_data, id_maps = build_hetero_graph(edges=graph_edges)
    
    print(f"\n✅ Built HeteroData:")
    print(f"   Node types ({len(hetero_data.node_types)}): {hetero_data.node_types}")
    print(f"   Edge types: {len(hetero_data.edge_types)}")
    
    # Print node counts
    print(f"\n📊 Node counts:")
    for node_type in hetero_data.node_types:
        print(f"   {node_type}: {hetero_data[node_type].num_nodes}")
    
    # ============================================================
    # 5. Attach Node Features
    # ============================================================
    print(f"\n🎨 Initializing node features...")
    
    hetero_data = attach_node_features(
        hetero_data,
        id_maps,
        init_method=cfg.model.node_features.init_method,
        embedding_dim=cfg.model.node_features.embedding_dim,
        seed=cfg.train.seed,
    )
    
    # ============================================================
    # 6. Build HGT Model
    # ============================================================
    print(f"\n🏗️ Building HGT model...")
    
    model = build_hgt_model(
        hetero_data,
        hidden_dim=cfg.model.hgt.hidden_dim,
        out_dim=cfg.model.hgt.hidden_dim,
        num_heads=cfg.model.hgt.num_heads,
        num_layers=cfg.model.hgt.num_layers,
        dropout=cfg.model.hgt.dropout,
    )
    
    # Initialize lazy parameters
    print(f"🔧 Initializing model parameters...")
    with torch.no_grad():
        _ = model.encode(hetero_data.x_dict, hetero_data.edge_index_dict)
    
    num_params = count_parameters(model)
    print(f"✅ Built HGT model:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Hidden dim: {cfg.model.hgt.hidden_dim}")
    print(f"   Num heads: {cfg.model.hgt.num_heads}")
    print(f"   Num layers: {cfg.model.hgt.num_layers}")
    
    # ============================================================
    # 7. Create Lightning Module
    # ============================================================
    print(f"\n⚡ Creating Lightning module...")
    
    lightning_model = HGTRecLightning(
        model=model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 0.0),
        supervision_src_type=cfg.data.graph.supervision.src_type,
        supervision_dst_type=cfg.data.graph.supervision.dst_type,
    )
    
    # ============================================================
    # 8. Setup Output Directory
    # ============================================================
    exp_name = cfg.get("experiment_name", "default")
    output_dir = f"runs/{exp_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # ============================================================
    # 9. Training (Placeholder - Full implementation needed)
    # ============================================================
    print(f"\n🎯 Training configuration:")
    print(f"   Epochs: {cfg.train.num_epochs}")
    print(f"   Batch size: {cfg.train.batch_size}")
    print(f"   Learning rate: {cfg.train.lr}")
    
    print(f"\n⚠️ Note: Full training loop requires LinkNeighborLoader implementation")
    print(f"   This is a simplified version for testing graph building and model initialization")
    
    # ============================================================
    # 10. Evaluation Setup
    # ============================================================
    print(f"\n📊 Evaluation configuration:")
    print(f"   K values: {cfg.eval.k_values}")
    
    evaluator = Evaluator(
        k_values=cfg.eval.k_values,
        output_dir=output_dir,
    )
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*80)
    print("PIPELINE SETUP COMPLETE")
    print("="*80)
    print(f"✅ Graph built successfully!")
    print(f"   Nodes: {sum(data.num_nodes for data in hetero_data.node_stores)}")
    print(f"   Edges: {sum(data.edge_index.size(1) for data in hetero_data.edge_stores if hasattr(data, 'edge_index'))}")
    print(f"   Model parameters: {num_params:,}")
    print(f"\n✅ Ready for training!")
    print(f"   Output: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HGT for link prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    main(args.config)

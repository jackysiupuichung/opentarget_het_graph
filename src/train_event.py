#!/usr/bin/env python3
"""
Training script for Event-Based Temporal Graph Learning (HGT/GAT with Temporal Edges).
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_geometric.loader import LinkNeighborLoader
from src.data.temporal_loader import load_event_graph, get_temporal_masks, filter_graph_by_time, to_time_agnostic
from src.data import init_wandb
from src.models.utils import build_model
from src.benchmark.evaluator import Evaluator
from src.data.evaluation_prep import build_evaluation_sets


def train_one_epoch(model, loader, optimizer, device, supervision_edge_type, src_type, dst_type):
    model.train()
    total_loss = 0
    total_examples = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Prepare edge_time_dict for the model
        edge_time_dict = None
        if any('edge_time' in batch[et] for et in batch.edge_types):
             edge_time_dict = {et: batch[et].edge_time for et in batch.edge_types if 'edge_time' in batch[et]}

        # Forward pass
        pred_scores = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[supervision_edge_type].edge_label_index,
            src_type,
            dst_type,
            edge_time_dict=edge_time_dict
        )
        
        # Prepare targets (MSE Regression)
        num_pos = batch[supervision_edge_type].edge_label.size(0)
        full_batch_size = batch[supervision_edge_type].edge_label_index.size(1)
        num_neg = full_batch_size - num_pos
        
        pos_targets = batch[supervision_edge_type].edge_label.float()
        neg_targets = torch.zeros(num_neg, device=device)
        targets = torch.cat([pos_targets, neg_targets])
        
        # Loss
        curr_pred = pred_scores[:targets.size(0)]
        loss = F.mse_loss(curr_pred, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * full_batch_size
        total_examples += full_batch_size
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / total_examples


def main(config_path: str):
    print("\n" + "="*80)
    print("EVENT-BASED TEMPORAL TRAINING")
    print("="*80 + "\n")
    
    # 1. Config
    project_root = os.path.dirname(os.path.dirname(__file__))
    base_cfg = OmegaConf.load(os.path.join(project_root, "config/benchmark_config.yaml"))
    
    try:
        exp_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base_cfg, exp_cfg)
    except:
        cfg = base_cfg
        
    # Initialize WandB
    init_wandb(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Data
    temporal_graph_path = os.path.join(project_root, cfg.data.temporal_graph_file)
    hetero_data = load_event_graph(
        temporal_graph_path, 
        attach_features=True, 
        to_undirected=True, 
        embedding_dim=cfg.model.node_features.embedding_dim
    )
    
    # 3. Splits
    # Support new explicit range config: train: [start, end], val: [start, end], test: [start, end]
    train_end = cfg.data.temporal_split.train[1]
    val_end = cfg.data.temporal_split.val[1]
    split_config = cfg.data.temporal_split
    
    # Contexts
    # Train context: Events up to train_end
    # Test/Val context: Events up to val_end (technically val_loader uses train_context, but test uses test_context)
    train_snapshot = filter_graph_by_time(hetero_data, train_end)
    val_snapshot = filter_graph_by_time(hetero_data, val_end)
    
    train_context = train_snapshot
    test_context = val_snapshot

    print(f"   Using event-based temporal graph.")

    # 4. Supervision Edge Info
    src_type = cfg.data.graph.supervision.src_type
    dst_type = cfg.data.graph.supervision.dst_type
    relation = cfg.data.graph.supervision.relation
    
    supervision_edge_type = None
    for et in hetero_data.edge_types:
        if (et[0] == src_type and et[2] == dst_type and relation in et[1]):
            supervision_edge_type = et
            break
    if not supervision_edge_type: raise ValueError("Supervision edge type not found")
    
    # 5. Extract Edges for Splits
    # Train Edges: From Train Context
    train_edge_index = train_context[supervision_edge_type].edge_index
    if 'edge_attr' in train_context[supervision_edge_type]:
        train_labels = train_context[supervision_edge_type].edge_attr.squeeze()
    else: train_labels = torch.ones(train_edge_index.size(1))
    
    # Val Edges & Test Edges: From Full Data, filtered by time
    masks = get_temporal_masks(
        hetero_data, 
        split_config=split_config,
        train_year=train_end, 
        val_year=val_end
    )
    
    train_mask, val_mask, test_mask = masks[supervision_edge_type]
    
    val_edge_index = hetero_data[supervision_edge_type].edge_index[:, val_mask]
    val_labels = hetero_data[supervision_edge_type].edge_attr.squeeze()[val_mask]
    
    print(f"Stats:")
    print(f"  Train edges (<= {train_end}): {train_edge_index.size(1):,}")
    print(f"  Val edges   (Target):      {val_edge_index.size(1):,}")
    print(f"  Test edges  (Target):      {test_mask.sum():,}")
    
    # 6. Loaders
    print("\n🚚 Creating Loaders (Temporal)...")
    
    # Extract edge times for temporal sampling
    train_edge_times = None
    if 'edge_time' in train_context[supervision_edge_type]:
        train_edge_times = train_context[supervision_edge_type].edge_time
    else:
        train_edge_times = torch.zeros(train_edge_index.size(1), dtype=torch.long)
    
    train_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, train_edge_index),
        edge_label=train_labels,
        edge_label_time=train_edge_times - 1 if train_edge_times is not None else None, 
        time_attr='edge_time',
        temporal_strategy='last',
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    # Validation Loader
    val_edge_times = None
    if 'edge_time' in hetero_data[supervision_edge_type]:
        val_edge_times = hetero_data[supervision_edge_type].edge_time[val_mask]
    else:
        val_edge_times = torch.zeros(val_edge_index.size(1), dtype=torch.long)
    
    val_loader = LinkNeighborLoader(
        data=train_context, # Training Context used to predict Validation Events
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, val_edge_index),
        edge_label=val_labels,
        edge_label_time=val_edge_times - 1 if val_edge_times is not None else None,
        time_attr='edge_time',
        temporal_strategy='last',
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # 7. Model
    model = build_model(
        model_name=cfg.model.get('name', 'hgt'), # Default to hgt for event
        data=train_context,
        hidden_dim=cfg.model.hgt.hidden_dim,
        num_heads=cfg.model.hgt.num_heads,
        num_layers=cfg.model.hgt.num_layers,
        dropout=cfg.model.hgt.dropout,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    evaluator = Evaluator(k_values=cfg.eval.k_values, output_dir=f"runs/{cfg.get('experiment_name', 'default')}")
    
    # 8. Pre-compute Evaluation Sets
    print("\n🛠️  Pre-computing Evaluation Sets...")
    val_targets, val_history, val_srcs = build_evaluation_sets(hetero_data, supervision_edge_type, val_mask, train_mask)
    
    exclusion_mask = (train_mask | val_mask)
    test_targets, test_history, test_srcs = build_evaluation_sets(hetero_data, supervision_edge_type, test_mask, exclusion_mask)
    
    # 9. Training Loop
    print("\n🔄 Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, supervision_edge_type, src_type, dst_type)
        val_loss = evaluator.validate_regression(model, val_loader, device, supervision_edge_type, src_type, dst_type)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Regression Loss: {val_loss:.4f}")
        
        # Log to WandB
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_regression_loss": val_loss
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{evaluator.output_dir}/best_model.pt")
            
            # Monitoring Ranking
            val_metrics = evaluator.evaluate_ranking(
                model, train_context, val_targets, val_history, val_srcs,
                supervision_edge_type, hetero_data[dst_type].num_nodes, device
            )
            
            # Log Validation Metrics to WandB
            if cfg.wandb.enabled:
                wandb.log({f"val_{k}": v for k, v in val_metrics.items()})

    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")
    
    # 10. Test Eval
    print(f"\n🧪 Starting TEST Evaluation...")
    model.load_state_dict(torch.load(f"{evaluator.output_dir}/best_model.pt"))
    model.eval()
    
    metrics = evaluator.evaluate_ranking(
        model, test_context, test_targets, test_history, test_srcs,
        supervision_edge_type, hetero_data[dst_type].num_nodes, device,
        num_negatives=None # Exhaustive
    )
    
    # Log Test Metrics to WandB
    if cfg.wandb.enabled:
        wandb.log({f"test_{k}": v for k, v in metrics.items()})
        
    # Finish WandB run
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

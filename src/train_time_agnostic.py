#!/usr/bin/env python3
"""
Training script for Time-Agnostic Graph Learning.
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
import pandas as pd


def train_one_epoch(model, loader, optimizer, device, supervision_edge_type, src_type, dst_type):
    model.train()
    total_loss = 0
    total_examples = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass (Standard Static)
        pred_scores = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[supervision_edge_type].edge_label_index,
            src_type,
            dst_type
        )
        
        # Prepare targets (MSE Regression)
        num_pos = batch[supervision_edge_type].edge_label.size(0)
        full_batch_size = batch[supervision_edge_type].edge_label_index.size(1)
        num_neg = full_batch_size - num_pos
        
        pos_targets = batch[supervision_edge_type].edge_label.float()
        neg_targets = torch.zeros(num_neg, device=device)
        targets = torch.cat([pos_targets, neg_targets])
        
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
    print("TIME-AGNOSTIC TRAINING")
    print("="*80 + "\n")
    
    # 1. Config
    project_root = os.path.dirname(os.path.dirname(__file__))
    base_cfg = OmegaConf.load(os.path.join(project_root, "config/benchmark_config.yaml"))
    try:
        exp_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base_cfg, exp_cfg)
    except:
        cfg = base_cfg # fallback if no exp config
        
    # Initialize WandB
    init_wandb(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Data
    temporal_graph_path = os.path.join(project_root, cfg.data.temporal_graph_file)
    hetero_data = load_event_graph(
        temporal_graph_path,
        to_undirected=True
    )
    
    # 3. Splits & Collapsing
    train_end = cfg.data.temporal_split.train[1]
    val_end = cfg.data.temporal_split.val[1]
    split_config = cfg.data.temporal_split
    
    # Create Snapshots (Context)
    train_snapshot = filter_graph_by_time(hetero_data, train_end)
    val_snapshot = filter_graph_by_time(hetero_data, val_end)
    
    print("   Collapsing temporal graph into static view...")
    train_context = to_time_agnostic(train_snapshot) # Input for Train/Val
    test_context = to_time_agnostic(val_snapshot)   # Input for Test
    
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
    # Train Edges (Collapsed Context) - NOTE: This uses ALL edges in train_context.
    # New logic: If temporal_split.train has a start year > 2000, we technically should filter out old edges?
    # But 'train_context' = filter_graph_by_time(train_end). It includes 2000-2016.
    # If config says Train: [2000, 2016], this is consistent.
    
    train_edge_index = train_context[supervision_edge_type].edge_index
    if 'edge_attr' in train_context[supervision_edge_type]:
        train_labels = train_context[supervision_edge_type].edge_attr.squeeze()
    else: train_labels = torch.ones(train_edge_index.size(1))
    
    # Val Edges (From Raw, masked to Val Range)
    # Pass split_config if available, or legacy years
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
    print(f"  Train edges (Context): {train_edge_index.size(1):,}")
    print(f"  Val edges   (Target):  {val_edge_index.size(1):,}")
    print(f"  Test edges  (Target):  {test_mask.sum():,}")
    
    # 6. Loaders
    print("\n🚚 Creating Loaders (Static)...")
    
    train_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, train_edge_index),
        edge_label=train_labels,
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    val_loader = LinkNeighborLoader(
        data=train_context, # Static Context
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, val_edge_index),
        edge_label=val_labels,
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # 7. Model
    model = build_model(
        model_name=cfg.model.get('name', 'gatv2'), # Default to gatv2 for agnostic
        data=train_context,
        hidden_dim=cfg.model.hgt.hidden_dim, # Reuse hgt config block
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
    
    # Early Stopping
    patience = cfg.train.get('early_stopping', {}).get('patience', 10)
    enabled = cfg.train.get('early_stopping', {}).get('enabled', False)
    
    if enabled:
        from src.utils.early_stopping import EarlyStopper
        early_stopper = EarlyStopper(patience=patience, verbose=True)
        print(f"🛑 Early stopping enabled with patience {patience}")
    else:
        early_stopper = None

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
            
            # Run ranking only on improvement or periodicity to save time
            val_metrics = evaluator.evaluate_ranking(
                model, train_context, val_targets, val_history, val_srcs,
                supervision_edge_type, hetero_data[dst_type].num_nodes, device
            )
            
            # Log Validation Metrics to WandB
            if cfg.wandb.enabled:
                wandb.log({f"val_{k}": v for k, v in val_metrics.items()})

        # Early Stopping Check
        if early_stopper:
            if early_stopper(val_loss):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
            
    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")
    
    # 10. Test Eval
    print(f"\n🧪 Starting TEST Evaluation...")
    model.load_state_dict(torch.load(f"{evaluator.output_dir}/best_model.pt"))
    model.eval()
    
    # Evaluate on ALL test diseases
    print("\n📊 Evaluating on ALL test diseases...")
    metrics = evaluator.evaluate_ranking(
        model, test_context, test_targets, test_history, test_srcs,
        supervision_edge_type, hetero_data[dst_type].num_nodes, device,
        num_negatives=None # Exhaustive
    )
    
    # Log Test Metrics to WandB
    if cfg.wandb.enabled:
        wandb.log({f"test_all_{k}": v for k, v in metrics.items()})
    
    # Evaluate on VALIDATION diseases only
    print("\n📊 Evaluating on VALIDATION diseases...")
    try:
        val_csv_path = os.path.join(project_root, "data/validation_diseases.csv")
        val_df = pd.read_csv(val_csv_path)
        
        # Filter for rows where graph_node_idx != -1 and get the list
        if 'graph_node_idx' in val_df.columns:
            validation_indices = val_df[val_df['graph_node_idx'] != -1]['graph_node_idx'].tolist()
            print(f"   Loaded {len(validation_indices)} validation disease indices from CSV.")
            
            metrics_validation = evaluator.evaluate_ranking(
                model, test_context, test_targets, test_history, test_srcs,
                supervision_edge_type, hetero_data[dst_type].num_nodes, device,
                num_negatives=None,
                validation_src_filter=validation_indices
            )
            
            # Log Validation Disease Metrics to WandB
            if cfg.wandb.enabled:
                wandb.log({f"test_validation_{k}": v for k, v in metrics_validation.items()})
        else:
            print("⚠️  'graph_node_idx' column not found in validation_diseases.csv. Skipping validation disease evaluation.")
        
        metrics_validation = evaluator.evaluate_ranking(
            model, test_context, test_targets, test_history, test_srcs,
            supervision_edge_type, hetero_data[dst_type].num_nodes, device,
            num_negatives=None,
            validation_src_filter=validation_indices
        )
        
        # Log Validation Disease Metrics to WandB
        if cfg.wandb.enabled:
            wandb.log({f"test_validation_{k}": v for k, v in metrics_validation.items()})
    except Exception as e:
        print(f"⚠️  Could not evaluate validation diseases: {e}")
        
    # Finish WandB run
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)

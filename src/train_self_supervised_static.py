#!/usr/bin/env python3
"""
Self-supervised pretraining on static (time-agnostic) graphs.

Uses temporal snapshots to create train/val/test splits based on accumulative
monotonic increasing graphs. Supports multiple model architectures (HGT, GATv2, GATv3).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_geometric.loader import LinkNeighborLoader
from src.data.temporal_loader import (
    load_event_graph,
    filter_graph_by_time,
    to_time_agnostic,
    get_temporal_masks
)
from src.data import init_wandb
from src.models.utils import build_model
from src.evaluation.self_supervised_metrics import (
    evaluate_link_prediction,
    print_metrics
)


# Edge type configuration
CLINICAL_TRIAL_KEYWORDS = ['clinical_trial']


def is_clinical_trial_edge(edge_type):
    """Check if edge type is clinical trial (exclude from pretraining)."""
    return any(kw in edge_type[1] for kw in CLINICAL_TRIAL_KEYWORDS)


def get_edge_loss_type(graph, edge_type):
    """Determine if edge uses BCE (binary) or Huber (continuous) loss."""
    edge_store = graph[edge_type]
    if 'edge_attr' not in edge_store:
        return 'bce'  # Default to binary
    
    scores = edge_store['edge_attr'].flatten()
    unique_vals = torch.unique(scores)
    is_binary = set(unique_vals.tolist()).issubset({0.0, 1.0})
    return 'bce' if is_binary else 'huber'


def prepare_edge_splits(
    temporal_graph,
    train_end_year,
    val_end_year,
    test_end_year,
    split_config
):
    """
    Prepare train/val/test edge splits using TGN-style temporal snapshots.
    
    Following TGN's approach: accumulative contexts with forecasting objective.
    
    Creates accumulative static graphs for each split:
    - Train context: All edges up to train_end_year (≤ 2015)
    - Val context: All edges up to val_end_year (≤ 2017) - includes train+val
    - Test context: All edges up to test_end_year (≤ 2024) - full graph
    
    Supervision edges (what we predict):
    - Train edges: From temporal masks (typically 2015 or 2014-2015)
    - Val edges: 2016-2017 (using context ≤ 2017)
    - Test edges: 2018-2024 (using context ≤ 2024)
    
    This matches TGN's strategy: at each evaluation point, the model has access
    to all historical edges up to that time, making it a realistic forecasting task.
    
    Returns:
        train_context: Static graph for training (≤ train_end_year)
        val_context: Static graph for validation (≤ val_end_year)
        test_context: Static graph for testing (≤ test_end_year)
        train_edges: Dict of train edges per type
        val_edges: Dict of val edges per type
        test_edges: Dict of test edges per type
        edge_loss_config: Dict mapping edge_type -> loss type
    """
    print("\n📸 Creating Temporal Snapshots (TGN-style)...")
    print(f"   Train context: edges ≤ {train_end_year}")
    print(f"   Val context: edges ≤ {val_end_year} (accumulative)")
    print(f"   Test context: edges ≤ {test_end_year} (full graph)")
    
    # Create accumulative snapshots - each includes all previous periods
    train_snapshot = filter_graph_by_time(temporal_graph, train_end_year)
    val_snapshot = filter_graph_by_time(temporal_graph, val_end_year)
    test_snapshot = filter_graph_by_time(temporal_graph, test_end_year)
    
    print("   Collapsing temporal graphs to static views...")
    train_context = to_time_agnostic(train_snapshot)
    val_context = to_time_agnostic(val_snapshot)
    test_context = to_time_agnostic(test_snapshot)
    
    # Get temporal masks for supervision edges
    masks = get_temporal_masks(
        temporal_graph,
        split_config=split_config
    )
    
    # Extract edges for each split
    train_edges = {}
    val_edges = {}
    test_edges = {}
    edge_loss_config = {}
    
    print("\n🎯 Extracting Supervision Edges:")
    
    for etype in temporal_graph.edge_types:
        if is_clinical_trial_edge(etype):
            print(f"   ❌ Excluding: {etype}")
            continue
        
        # Determine loss type
        loss_type = get_edge_loss_type(temporal_graph, etype)
        edge_loss_config[etype] = loss_type
        
        # Get masks
        train_mask, val_mask, test_mask = masks[etype]
        
        # Extract edges from temporal graph
        full_edge_index = temporal_graph[etype].edge_index
        full_edge_attr = temporal_graph[etype].edge_attr if 'edge_attr' in temporal_graph[etype] else None
        
        # Train edges (from collapsed context)
        if etype in train_context.edge_types:
            train_edges[etype] = {
                'edge_index': train_context[etype].edge_index,
                'edge_attr': train_context[etype].edge_attr if 'edge_attr' in train_context[etype] else None
            }
        
        # Val edges (from temporal graph, filtered by mask)
        if val_mask.sum() > 0:
            val_edges[etype] = {
                'edge_index': full_edge_index[:, val_mask],
                'edge_attr': full_edge_attr[val_mask] if full_edge_attr is not None else None
            }
        
        # Test edges (from temporal graph, filtered by mask)
        if test_mask.sum() > 0:
            test_edges[etype] = {
                'edge_index': full_edge_index[:, test_mask],
                'edge_attr': full_edge_attr[test_mask] if full_edge_attr is not None else None
            }
        
        # Print stats
        num_train = train_edges[etype]['edge_index'].size(1) if etype in train_edges else 0
        num_val = val_edges[etype]['edge_index'].size(1) if etype in val_edges else 0
        num_test = test_edges[etype]['edge_index'].size(1) if etype in test_edges else 0
        
        print(f"   ✓ {etype}: {num_train:,} train, {num_val:,} val, {num_test:,} test ({loss_type})")
    
    return train_context, val_context, test_context, train_edges, val_edges, test_edges, edge_loss_config


def train_one_epoch(
    model,
    loaders,
    optimizer,
    device,
    edge_loss_config
):
    """Train one epoch using mini-batch training with LinkNeighborLoader."""
    model.train()
    total_loss = 0
    total_examples = 0
    loss_breakdown = {'bce': 0, 'huber': 0}
    num_batches = {'bce': 0, 'huber': 0}
    
    # Train on each edge type
    for etype, loader in loaders.items():
        loss_type = edge_loss_config[etype]
        src_type, rel, dst_type = etype
        
        pbar = tqdm(loader, desc=f"Training {etype[1][:30]}", leave=False)
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred_scores = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[etype].edge_label_index,
                src_type,
                dst_type
            )
            
            # Compute loss based on edge type
            if loss_type == 'bce':
                # Binary: LinkNeighborLoader already provides edge_label with 1s (pos) and 0s (neg)
                targets = batch[etype].edge_label.float()
                loss = F.binary_cross_entropy_with_logits(pred_scores[:targets.size(0)], targets)
                loss_breakdown['bce'] += loss.item()
                num_batches['bce'] += 1
            else:  # huber
                # Continuous: use actual edge scores (no negative sampling for Huber)
                targets = batch[etype].edge_label.flatten()
                loss = F.huber_loss(pred_scores[:targets.size(0)], targets)
                loss_breakdown['huber'] += loss.item()
                num_batches['huber'] += 1
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch[etype].edge_label_index.size(1)
            total_examples += batch[etype].edge_label_index.size(1)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Average loss breakdown
    if num_batches['bce'] > 0:
        loss_breakdown['bce'] /= num_batches['bce']
    if num_batches['huber'] > 0:
        loss_breakdown['huber'] /= num_batches['huber']
    
    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0
    return avg_loss, loss_breakdown


def main(config_path):
    print("\n" + "="*80)
    print("STATIC SELF-SUPERVISED PRETRAINING")
    print("="*80 + "\n")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Initialize WandB
    if cfg.wandb.enabled:
        init_wandb(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load temporal graph
    print(f"\n📊 Loading graph from {cfg.data.graph_file}")
    temporal_graph = load_event_graph(cfg.data.graph_file, to_undirected=True)
    
    # Prepare edge splits using temporal snapshots
    train_end = cfg.pretrain.temporal_split.train[1]
    val_end = cfg.pretrain.temporal_split.val[1]
    test_end = cfg.pretrain.temporal_split.test[1]
    split_config = cfg.pretrain.temporal_split
    
    (train_context, val_context, test_context,
     train_edges, val_edges, test_edges,
     edge_loss_config) = prepare_edge_splits(
        temporal_graph, train_end, val_end, test_end, split_config
    )
    
    # Move contexts to device
    train_context = train_context.to(device)
    val_context = val_context.to(device)
    test_context = test_context.to(device)
    
    # Create LinkNeighborLoaders for mini-batch training
    print("\n🚚 Creating Data Loaders...")
    batch_size = cfg.pretrain.get('batch_size', 512)
    # Use list for multi-hop sampling (e.g., [20, 10] means 20 neighbors for 1st hop, 10 for 2nd)
    # PyG LinkNeighborLoader supports list of ints for all edge types
    num_neighbors = cfg.pretrain.get('num_neighbors', [20, 10])
    
    # Ensure num_neighbors is a list of integers
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors]
    
    print(f"   Batch size: {batch_size}")
    print(f"   Num neighbors: {num_neighbors}")
    print(f"   Negative sampling ratio: {cfg.pretrain.get('neg_sampling_ratio', 1.0)}")

    
    train_loaders = {}
    val_loaders = {}
    
    # Construct num_neighbors dict for all edge types to ensure safety
    # This prevents PyG errors if some types are empty
    # Use list for multi-hop: [20, 10]
    default_neighbors = cfg.pretrain.get('num_neighbors', [20, 10])
    if isinstance(default_neighbors, int):
        default_neighbors = [default_neighbors]
        
    num_neighbors_dict = {et: default_neighbors for et in temporal_graph.edge_types}
    
    for etype in train_edges.keys():
        if train_edges[etype]['edge_index'].size(1) == 0:
            continue
        
        loss_type = edge_loss_config[etype]
        
        # Train loader
        edge_label = train_edges[etype]['edge_attr'].squeeze() if train_edges[etype]['edge_attr'] is not None else torch.ones(train_edges[etype]['edge_index'].size(1))
        
        # CRITICAL: Only use negative sampling for binary edges (BCE)
        if loss_type == 'bce':
            neg_sampling_config = dict(mode='binary', amount=cfg.pretrain.get('neg_sampling_ratio', 1.0))
        else:  # huber
            neg_sampling_config = None
        
        train_loaders[etype] = LinkNeighborLoader(
            data=train_context,
            num_neighbors=num_neighbors_dict,
            edge_label_index=(etype, train_edges[etype]['edge_index']),
            edge_label=edge_label,
            neg_sampling=neg_sampling_config,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            persistent_workers=False
        )
        
        # Val loader
        if etype in val_edges and val_edges[etype]['edge_index'].size(1) > 0:
            val_edge_label = val_edges[etype]['edge_attr'].squeeze() if val_edges[etype]['edge_attr'] is not None else torch.ones(val_edges[etype]['edge_index'].size(1))
            
            val_loaders[etype] = LinkNeighborLoader(
                data=val_context,
                num_neighbors=num_neighbors_dict,
                edge_label_index=(etype, val_edges[etype]['edge_index']),
                edge_label=val_edge_label,
                neg_sampling=neg_sampling_config,  # Same as train
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                persistent_workers=False
            )
    
    print(f"   Batch size: {batch_size}")
    print(f"   Num neighbors: {num_neighbors}")
    print(f"   Negative sampling ratio: {cfg.pretrain.get('neg_sampling_ratio', 1.0)}")
    
    # Build model
    print(f"\n🏗️  Building {cfg.model.name} model...")
    model = build_model(
        model_name=cfg.model.name,
        data=train_context,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.get('weight_decay', 0.0)
    )
    
    # Create output directory
    output_dir = Path(cfg.pretrain.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🔄 Starting Training...")
    print(f"   Epochs: {cfg.pretrain.num_epochs}")
    print(f"   Learning rate: {cfg.pretrain.lr}")
    print(f"   Early stopping patience: {cfg.pretrain.get('early_stopping_patience', 10)}")
    
    best_val_metric = float('inf')  # Lower is better for loss
    patience_counter = 0
    patience = cfg.pretrain.get('early_stopping_patience', 10)
    
    for epoch in range(cfg.pretrain.num_epochs):
        # Train
        train_loss, loss_breakdown = train_one_epoch(
            model, train_loaders,
            optimizer, device, edge_loss_config
        )
        
        # Evaluate on validation set (using loaders for consistency)
        model.eval()
        val_loss = 0
        val_examples = 0
        val_breakdown = {'bce': 0, 'huber': 0}
        val_batches = {'bce': 0, 'huber': 0}
        
        with torch.no_grad():
            for etype, loader in val_loaders.items():
                loss_type = edge_loss_config[etype]
                src_type, rel, dst_type = etype
                
                pbar = tqdm(loader, desc=f"Validating {etype[1][:30]}", leave=False)
                
                for batch in pbar:
                    batch = batch.to(device)
                    
                    pred_scores = model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch[etype].edge_label_index,
                        src_type,
                        dst_type
                    )
                    
                    if loss_type == 'bce':
                        # Binary: edge_label already contains 1s (pos) and 0s (neg)
                        targets = batch[etype].edge_label.float()
                        loss = F.binary_cross_entropy_with_logits(pred_scores[:targets.size(0)], targets)
                        val_breakdown['bce'] += loss.item()
                        val_batches['bce'] += 1
                    else:  # huber
                        # Continuous: use actual edge scores
                        targets = batch[etype].edge_label.flatten()
                        loss = F.huber_loss(pred_scores[:targets.size(0)], targets)
                        val_breakdown['huber'] += loss.item()
                        val_batches['huber'] += 1
                    
                    val_loss += loss.item() * batch[etype].edge_label_index.size(1)
                    val_examples += batch[etype].edge_label_index.size(1)
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average validation metrics
        if val_batches['bce'] > 0:
            val_breakdown['bce'] /= val_batches['bce']
        if val_batches['huber'] > 0:
            val_breakdown['huber'] /= val_batches['huber']
        
        avg_val_loss = val_loss / val_examples if val_examples > 0 else float('inf')
        
        # Use Huber loss as primary metric if available
        if val_batches['huber'] > 0:
            val_metric = val_breakdown['huber']
            metric_name = 'Huber'
        elif val_batches['bce'] > 0:
            val_metric = val_breakdown['bce']
            metric_name = 'BCE'
        else:
            val_metric = avg_val_loss
            metric_name = 'Loss'
        
        # Logging
        print(f"\nEpoch {epoch+1:03d}/{cfg.pretrain.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} (BCE: {loss_breakdown['bce']:.4f}, Huber: {loss_breakdown['huber']:.4f})")
        print(f"  Val {metric_name}: {val_metric:.4f}")
        
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_bce_loss": loss_breakdown['bce'],
                "train_huber_loss": loss_breakdown['huber'],
                "val_loss": avg_val_loss,
                "val_bce_loss": val_breakdown['bce'],
                "val_huber_loss": val_breakdown['huber']
            })
        
        # Save best model
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "pretrained_best.pt")
            print(f"  ✓ New best model saved (Val {metric_name}: {val_metric:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Test evaluation
    print(f"\n🧪 Starting TEST Evaluation...")
    model.load_state_dict(torch.load(output_dir / "pretrained_best.pt"))
    
    test_metrics = evaluate_link_prediction(
        model, test_context, test_edges,
        edge_loss_config, device,
        num_neg_per_pos=cfg.pretrain.get('num_neg_samples', 1)
    )
    
    print_metrics(test_metrics, prefix="Test")
    
    if cfg.wandb.enabled:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()
    
    print(f"\n✅ Pretraining Complete!")
    print(f"   Best Val {metric_name}: {best_val_metric:.4f}")
    print(f"   Model saved to: {output_dir / 'pretrained_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static self-supervised pretraining")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)

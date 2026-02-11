#!/usr/bin/env python3
"""
Self-supervised pretraining on event-based (temporal) graphs using Causal Sampling.

Features:
- Uses LinkNeighborLoader with `time_attr='edge_time'` for strict causal sampling.
- Use `temporal_strategy='last'` to sample most recent neighbors.
- Incorporates Time Encoding (Time features) into the GNN.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_geometric.loader import LinkNeighborLoader
from src.data.temporal_loader import (
    load_event_graph,
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



def prepare_event_splits(
    temporal_graph,
    split_config
):
    """
    Prepare train/val/test splits for EVENT-BASED training.
    
    Extracts indices AND timestamps for seeds.
    """
    print("\n📅 Preparing Causal Event Splits...")
    
    # Get temporal masks
    masks = get_temporal_masks(temporal_graph, split_config)
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    edge_loss_config = {}
    
    for etype in temporal_graph.edge_types:
        if is_clinical_trial_edge(etype):
            continue
            
        # Determine loss
        edge_loss_config[etype] = get_edge_loss_type(temporal_graph, etype)
        
        train_mask, val_mask, test_mask = masks[etype]
        
        # Edge Index & Time
        edge_index = temporal_graph[etype].edge_index
        edge_time = temporal_graph[etype].edge_time if 'edge_time' in temporal_graph[etype] else None
        edge_attr = temporal_graph[etype].edge_attr if 'edge_attr' in temporal_graph[etype] else None
        
        if edge_time is None:
            raise AttributeError(f"Edge type {etype} is missing edge_time attribute.")
            
        # Extract
        for mode, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
            if mask.sum() > 0:
                splits[mode][etype] = {
                    'edge_index': edge_index[:, mask],
                    'edge_label_time': edge_time[mask],
                    'edge_attr': edge_attr[mask] if edge_attr is not None else None
                }
                
    return splits['train'], splits['val'], splits['test'], edge_loss_config


def train_one_epoch(
    model,
    loaders,
    optimizer,
    device,
    edge_loss_config
):
    """Train one epoch using causal temporal sampling."""
    model.train()
    total_loss = 0
    total_examples = 0
    loss_breakdown = {'exist': 0, 'prob': 0}
    
    # Calculate total batches for unified progress bar
    total_batches = sum(len(loader) for loader in loaders.values())
    pbar = tqdm(total=total_batches, desc="Training Epoch", leave=False)
    num_batches = 0
    
    for etype, loader in loaders.items():
        loss_type = edge_loss_config[etype]
        src_type, _, dst_type = etype
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Standard forward pass (temporal info used in sampling)
            out = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch[etype].edge_label_index,
                src_type,
                dst_type
            )
            
            targets = batch[etype].edge_label.float()
            
            # Handle dual-head output
            if isinstance(out, dict):
                logits_exist = out['logits_exist']
                logits_prob = out['logits_prob']
                
                num_preds = logits_exist.size(0)
                targets = targets[:num_preds]
                
                # Head A: Existence (Binary Discovery)
                exist_targets = (targets > 0).float()
                loss_exist = F.binary_cross_entropy_with_logits(logits_exist, exist_targets)
                
                # Head B: Probability (Calibrated Strength)
                pos_mask = (targets > 0)
                if pos_mask.sum() > 0 and loss_type != 'bce':
                    prob_logits_pos = logits_prob[pos_mask]
                    prob_targets_pos = targets[pos_mask]
                    loss_prob = F.binary_cross_entropy_with_logits(prob_logits_pos, prob_targets_pos)
                else:
                    loss_prob = torch.tensor(0.0, device=device)
                    
                loss = loss_exist + loss_prob
                
                loss_breakdown['exist'] += loss_exist.item()
                loss_breakdown['prob'] += loss_prob.item()
                
            else:
                # Fallback for single-head models
                if loss_type == 'bce':
                    loss = F.binary_cross_entropy_with_logits(out[:targets.size(0)], targets)
                else:
                    loss = F.huber_loss(out[:targets.size(0)], targets.flatten())
                loss_breakdown['exist'] += loss.item()
            
            loss.backward()
            optimizer.step()
            
            batch_size = batch[etype].edge_label_index.size(1)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            num_batches += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    pbar.close()
    
    # Average loss breakdown
    if num_batches > 0:
        loss_breakdown['exist'] /= num_batches
        loss_breakdown['prob'] /= num_batches
    
    return total_loss / total_examples if total_examples > 0 else 0.0, loss_breakdown


def main(config_path):
    print("\n" + "="*80)
    print("EVENT-BASED SELF-SUPERVISED PRETRAINING (Causal Temporal Sampling)")
    print("="*80)
    
    cfg = OmegaConf.load(config_path)
    if cfg.wandb.enabled: 
        init_wandb(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Graph
    print(f"\n📊 Loading graph from {cfg.data.graph_file}")
    temporal_graph = load_event_graph(
        cfg.data.graph_file, 
        to_undirected=True,
        normalize_features=True
    )
    
    # 2. Splits
    train_splits, val_splits, test_splits, edge_loss_config = prepare_event_splits(
        temporal_graph, cfg.data.temporal_split
    )
    
    # 3. Create Data Loaders with Causal Sampling
    print("\n🚚 Creating Data Loaders (Causal Temporal Sampling)...")
    batch_size = cfg.pretrain.batch_size
    num_neighbors = cfg.pretrain.num_neighbors
    
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors]
    
    print(f"   Batch size: {batch_size}")
    print(f"   Num neighbors: {num_neighbors}")
    print(f"   Negative sampling ratio: {cfg.pretrain.get('neg_sampling_ratio', 1.0)}")
    print(f"   ⏰ Temporal strategy: 'last' (causal neighbor sampling)")
    
    # Robustness: Map num_neighbors for all edge types
    if not isinstance(num_neighbors, dict):
        num_neighbors_dict = {et: num_neighbors for et in temporal_graph.edge_types}
    else:
        num_neighbors_dict = num_neighbors
    
    train_loaders = {}
    val_loaders = {}
    
    # Helper for Loader creation
    def create_loader(splits, shuffle=True):
        loaders = {}
        for etype, data_dict in splits.items():
            if data_dict['edge_index'].size(1) == 0: continue
            
            loss_type = edge_loss_config[etype]
            neg_sampling = dict(mode='binary', amount=1.0) if loss_type == 'bce' else None
            
            loaders[etype] = LinkNeighborLoader(
                data=temporal_graph, # Full graph!
                num_neighbors=num_neighbors_dict,
                edge_label_index=(etype, data_dict['edge_index']),
                edge_label_time=data_dict['edge_label_time'], # Crucial for causal sampling
                time_attr='edge_time', # Crucial
                temporal_strategy='last', # Causal strategy
                neg_sampling=neg_sampling,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0
            )
        return loaders
        
    train_loaders = create_loader(train_splits, shuffle=True)
    val_loaders = create_loader(val_splits, shuffle=False)
    
    # 4. Build Model
    model_name = cfg.model.get('name') or cfg.model.encoder.name
    hidden_dim = cfg.model.get('hidden_dim') or cfg.model.encoder.hidden_dim
    num_heads = cfg.model.get('num_heads') or cfg.model.encoder.num_heads
    num_layers = cfg.model.get('num_layers') or cfg.model.encoder.num_layers
    dropout = cfg.model.get('dropout') or cfg.model.encoder.dropout
    use_rte = cfg.model.get('use_rte', False)  # Enable RTE for event-based training

    print(f"\n🏗️  Building {model_name} model...")
    if use_rte:
        print(f"   ⏰ Relative Temporal Encoding (RTE): ENABLED")
    
    model = build_model(
        model_name=model_name,
        data=temporal_graph,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_rte=use_rte,
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.get('weight_decay', 0.0)
    )
    
    # Create output directory
    output_dir = Path(cfg.pretrain.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Training Loop with Early Stopping
    print(f"\n🔄 Starting Training...")
    print(f"   Epochs: {cfg.pretrain.num_epochs}")
    print(f"   Learning rate: {cfg.pretrain.lr}")
    print(f"   Early stopping patience: {cfg.pretrain.early_stopping.patience}")
    
    best_val_metric = float('inf')
    patience_counter = 0
    patience = cfg.pretrain.early_stopping.patience
    
    for epoch in range(cfg.pretrain.num_epochs):
        # Train
        train_loss, loss_breakdown = train_one_epoch(
            model, train_loaders,
            optimizer, device, edge_loss_config
        )
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_examples = 0
        val_breakdown = {'exist': 0, 'prob': 0}
        val_batches = 0
        
        total_val_batches = sum(len(loader) for loader in val_loaders.values())
        val_pbar = tqdm(total=total_val_batches, desc="Validating", leave=False)
        
        with torch.no_grad():
            for etype, loader in val_loaders.items():
                src_type, rel, dst_type = etype
                loss_type = edge_loss_config[etype]
                
                for batch in loader:
                    batch = batch.to(device)
                    
                    out = model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch[etype].edge_label_index,
                        src_type,
                        dst_type
                    )
                    
                    targets = batch[etype].edge_label.float()
                    
                    # Handle dual-head output
                    if isinstance(out, dict):
                        logits_exist = out['logits_exist']
                        logits_prob = out['logits_prob']
                        
                        num_preds = logits_exist.size(0)
                        targets = targets[:num_preds]
                        
                        exist_targets = (targets > 0).float()
                        loss_exist = F.binary_cross_entropy_with_logits(logits_exist, exist_targets)
                        
                        pos_mask = (targets > 0)
                        if pos_mask.sum() > 0 and loss_type != 'bce':
                            prob_logits_pos = logits_prob[pos_mask]
                            prob_targets_pos = targets[pos_mask]
                            loss_prob = F.binary_cross_entropy_with_logits(prob_logits_pos, prob_targets_pos)
                        else:
                            loss_prob = torch.tensor(0.0, device=device)
                            
                        loss = loss_exist + loss_prob
                        val_breakdown['exist'] += loss_exist.item()
                        val_breakdown['prob'] += loss_prob.item()
                    else:
                        loss = F.binary_cross_entropy_with_logits(out[:targets.size(0)], (targets > 0).float())
                        val_breakdown['exist'] += loss.item()
                    
                    val_loss += loss.item() * batch[etype].edge_label_index.size(1)
                    val_examples += batch[etype].edge_label_index.size(1)
                    val_batches += 1
                    
                    val_pbar.update(1)
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_pbar.close()
        
        # Average validation metrics
        if val_batches > 0:
            val_breakdown['exist'] /= val_batches
            val_breakdown['prob'] /= val_batches
        
        avg_val_loss = val_loss / val_examples if val_examples > 0 else float('inf')
        val_metric = avg_val_loss
        metric_name = 'Loss'
        
        # Logging
        print(f"\nEpoch {epoch+1:03d}/{cfg.pretrain.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Exist: {loss_breakdown['exist']:.4f}, Prob: {loss_breakdown['prob']:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f} (Exist: {val_breakdown['exist']:.4f}, Prob: {val_breakdown['prob']:.4f})")
        
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_exist_loss": loss_breakdown['exist'],
                "train_prob_loss": loss_breakdown['prob'],
                "val_loss": avg_val_loss,
                "val_exist_loss": val_breakdown['exist'],
                "val_prob_loss": val_breakdown['prob']
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
    
    # 6. Test Evaluation
    print(f"\n🧪 Starting TEST Evaluation...")
    
    # Clean test targets (NaNs)
    for etype in test_splits:
        if test_splits[etype].get('edge_attr') is not None:
            if torch.isnan(test_splits[etype]['edge_attr']).any():
                test_splits[etype]['edge_attr'] = torch.nan_to_num(test_splits[etype]['edge_attr'], nan=1.0)
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / "pretrained_best.pt"))
    
    # Convert test_splits to format expected by evaluate_link_prediction
    test_edges = {}
    for etype, data_dict in test_splits.items():
        test_edges[etype] = {
            'edge_index': data_dict['edge_index'],
            'edge_attr': data_dict['edge_attr']
        }
    
    test_metrics = evaluate_link_prediction(
        model, temporal_graph, test_edges,
        edge_loss_config, device,
        num_neg_per_pos=1
    )
    
    print_metrics(test_metrics, prefix="Test")
    
    if cfg.wandb.enabled:
        wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb.finish()
    
    print(f"\n✅ Pretraining Complete!")
    print(f"   Best Val {metric_name}: {best_val_metric:.4f}")
    print(f"   Model saved to: {output_dir / 'pretrained_best.pt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Event-based self-supervised pretraining with causal temporal sampling")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional key=value options")
    args = parser.parse_args()
    
    # Load config and merge CLI args
    cfg = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(cfg, cli_conf)
    
    main(cfg)

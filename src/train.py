#!/usr/bin/env python3
"""
Main training script for Temporal HGT link prediction.
Refactored for modularity and explicit Test Split evaluation.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from torch_geometric.loader import LinkNeighborLoader

from data.temporal_loader import load_event_graph, get_temporal_masks, filter_graph_by_time
from models.utils import build_hgt_model, count_parameters
from benchmark.evaluator import Evaluator


def train_one_epoch(model, loader, optimizer, device, supervision_edge_type, src_type, dst_type):
    model.train()
    total_loss = 0
    total_examples = 0
    
    pbar = tqdm(loader, desc="Training")
    
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred_scores = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[supervision_edge_type].edge_label_index,
            src_type,
            dst_type
        )
        
        # Prepare targets (MSE Regression)
        # edge_label_index contains [Positives, Negatives]
        num_pos = batch[supervision_edge_type].edge_label.size(0)
        full_batch_size = batch[supervision_edge_type].edge_label_index.size(1)
        num_neg = full_batch_size - num_pos
        
        pos_targets = batch[supervision_edge_type].edge_label.float()
        neg_targets = torch.zeros(num_neg, device=device)
        
        targets = torch.cat([pos_targets, neg_targets])
        
        # Loss
        # Slice prediction to match targets just in case
        curr_pred = pred_scores[:targets.size(0)]
        loss = F.mse_loss(curr_pred, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * full_batch_size
        total_examples += full_batch_size
        
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / total_examples


@torch.no_grad()
def validate_regression(model, loader, device, supervision_edge_type, src_type, dst_type):
    model.eval()
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        pred_scores = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch[supervision_edge_type].edge_label_index,
            src_type,
            dst_type
        )
        
        num_pos = batch[supervision_edge_type].edge_label.size(0)
        full_batch_size = batch[supervision_edge_type].edge_label_index.size(1)
        num_neg = full_batch_size - num_pos
        
        pos_targets = batch[supervision_edge_type].edge_label.float()
        neg_targets = torch.zeros(num_neg, device=device)
        targets = torch.cat([pos_targets, neg_targets])
        
        curr_pred = pred_scores[:targets.size(0)]
        loss = F.mse_loss(curr_pred, targets)
        
        total_loss += loss.item() * full_batch_size
        total_examples += full_batch_size
        
    return total_loss / total_examples


def main(config_path: str):
    print("\n" + "="*80)
    print("TEMPORAL HGT: TRAINING & EVALUATION")
    print("="*80 + "\n")
    
    # 1. Config
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        project_root = os.path.dirname(os.path.dirname(__file__))
        base_cfg = OmegaConf.load(os.path.join(project_root, "config/benchmark_config.yaml"))
        cfg = OmegaConf.merge(base_cfg, cfg)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load Data
    project_root = os.path.dirname(os.path.dirname(__file__))
    temporal_graph_path = os.path.join(project_root, cfg.data.temporal_graph_file)
    hetero_data = load_event_graph(temporal_graph_path, attach_features=True, embedding_dim=cfg.model.node_features.embedding_dim)
    
    # 3. Splits
    train_year = cfg.data.temporal_split.train_year
    val_year = cfg.data.temporal_split.val_year
    test_year = cfg.data.temporal_split.test_year
    
    # Create Snapshots for loaders
    train_snapshot = filter_graph_by_time(hetero_data, train_year)
    val_snapshot = filter_graph_by_time(hetero_data, val_year) # Used as context for Test? OR context for Val?
    # Actually:
    # Train Loader Context: train_snapshot (<= 2020)
    # Val Loader Context: train_snapshot (<= 2020) -> Predicts 2021 edges
    # Test Inference Context: val_snapshot (<= 2021) -> Predicts 2022 edges
    
    train_context = train_snapshot
    test_context = val_snapshot 
    
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
    # Note: hetero_data has ALL edges. 
    # train_edge_index comes from train_snapshot
    train_edge_index = train_context[supervision_edge_type].edge_index
    if 'edge_attr' in train_context[supervision_edge_type]:
        train_labels = train_context[supervision_edge_type].edge_attr.squeeze()
    else: train_labels = torch.ones(train_edge_index.size(1))
    
    # Val Edges: (train_year < t <= val_year)
    masks = get_temporal_masks(hetero_data, train_year, val_year)
    _, val_mask, test_mask_raw = masks[supervision_edge_type] # val_mask is correct
    
    # We need explicit test_mask for test_year (val_year < t <= test_year)
    # get_temporal_masks returns test_mask as > val_year. 
    # If we want strictly test_year, we should check max year. 
    # But usually > val_year is enough if data ends at test_year.
    test_mask = test_mask_raw
    
    val_edge_index = hetero_data[supervision_edge_type].edge_index[:, val_mask]
    val_labels = hetero_data[supervision_edge_type].edge_attr.squeeze()[val_mask]
    
    print(f"Stats:")
    print(f"  Train edges (<= {train_year}): {train_edge_index.size(1):,}")
    print(f"  Val edges   ({train_year} < t <= {val_year}): {val_edge_index.size(1):,}")
    print(f"  Test edges  (> {val_year}): {test_mask.sum():,}")
    
    # 6. Loaders
    print("\n🚚 Creating Loaders...")
    
    # Extract edge times for temporal sampling
    if 'edge_time' in train_context[supervision_edge_type]:
        train_edge_times = train_context[supervision_edge_type].edge_time
    else:
        # Fallback: use zeros if no temporal info
        train_edge_times = torch.zeros(train_edge_index.size(1), dtype=torch.long)
    
    train_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, train_edge_index),
        edge_label=train_labels,
        edge_label_time=train_edge_times - 1,  # Sample neighbors BEFORE this edge
        time_attr='edge_time',  # Attribute name for temporal filtering
        temporal_strategy='last',  # Use most recent neighbors within time window
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Validation Loader (Regression)
    # Context: train_context. Labels: val_edge_index
    if 'edge_time' in hetero_data[supervision_edge_type]:
        val_edge_times = hetero_data[supervision_edge_type].edge_time[val_mask]
    else:
        val_edge_times = torch.zeros(val_edge_index.size(1), dtype=torch.long)
    
    val_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, val_edge_index),
        edge_label=val_labels,
        edge_label_time=val_edge_times - 1,  # Sample neighbors BEFORE this edge
        time_attr='edge_time',
        temporal_strategy='last',
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    # 7. Model
    model = build_hgt_model(
        train_context, # Initialize with train graph schema
        hidden_dim=cfg.model.hgt.hidden_dim,
        num_heads=cfg.model.hgt.num_heads,
        num_layers=cfg.model.hgt.num_layers,
        dropout=cfg.model.hgt.dropout,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    evaluator = Evaluator(k_values=cfg.eval.k_values, output_dir=f"runs/{cfg.get('experiment_name', 'default')}")
    
    # 8. Training Loop
    print("\n🔄 Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, supervision_edge_type, src_type, dst_type)
        val_loss = validate_regression(model, val_loader, device, supervision_edge_type, src_type, dst_type)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{evaluator.output_dir}/best_model.pt")
            
    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")
    
    # 9. Test Evaluation (Novel Truth Ranking)
    print(f"\n🧪 Starting TEST Evaluation (Exhaustive Ranking)...")
    
    # Load Best Model
    model.load_state_dict(torch.load(f"{evaluator.output_dir}/best_model.pt"))
    model.eval()
    
    # Context: test_context (edges <= val_year)
    # Ground Truth: hetero_data (full)
    # Eval Mask: test_mask (edges > val_year)
    # Exclusion: edges <= val_year (ALL history)
    
    # Temporal masks returns (train, val, test)
    # exclusion_mask should be logical OR of train and val?
    # get_temporal_masks returns BOOLEAN masks on the FULL data.
    # So exclusion mask = ~test_mask (everything NOT in test) assuming strictly temporal
    # Or explicitly (train_mask | val_mask).
    
    ms = get_temporal_masks(hetero_data, train_year, val_year)[supervision_edge_type]
    exclusion_mask = (ms[0] | ms[1]) # Train | Val
    
    metrics = evaluator.evaluate_ranking_exhaustive(
        model,
        inference_data=test_context,     # <= 2021
        ground_truth_data=hetero_data,   # All data
        edge_type=supervision_edge_type,
        eval_mask=test_mask,            # 2022 edges
        exclusion_mask=exclusion_mask,  # <= 2021 edges
        device=device
    )
    
    print("\n✅ Final Test Metrics:")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark_config.yaml")
    args = parser.parse_args()
    main(args.config)

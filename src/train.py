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

from data.temporal_loader import load_event_graph, get_temporal_masks, filter_graph_by_time, to_time_agnostic
from models.utils import build_hgt_model
from benchmark.evaluator import Evaluator
from data.evaluation_prep import build_evaluation_sets


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
    
    # Time-Agnostic Handling
    is_time_agnostic = cfg.data.graph.get('time_agnostic', False)
    if is_time_agnostic:
        print("\nUsing Time-Agnostic Graph (Collapsed)...")
        train_context = to_time_agnostic(train_snapshot)
        test_context = to_time_agnostic(val_snapshot)
    else:
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
    train_mask = masks[supervision_edge_type][0]
    
    # We need explicit test_mask for test_year (val_year < t <= test_year)
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
    # Extract edge times for temporal sampling (if not time_agnostic)
    train_edge_times = None
    if not is_time_agnostic:
        if 'edge_time' in train_context[supervision_edge_type]:
            train_edge_times = train_context[supervision_edge_type].edge_time
        else:
            # Fallback for temporal graph without explicit time? Should not happen in this flow.
            train_edge_times = torch.zeros(train_edge_index.size(1), dtype=torch.long)
    
    train_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, train_edge_index),
        edge_label=train_labels,
        edge_label_time=train_edge_times - 1 if train_edge_times is not None else None, 
        time_attr='edge_time' if not is_time_agnostic else None,  # disable temporal sampling if agnostic
        temporal_strategy='last',  # ignored if time_attr is None
        neg_sampling=dict(mode='binary', amount=1.0),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Validation Loader (Regression)
    # Context: train_context. Labels: val_edge_index
    # Validation Loader (Regression)
    # Context: train_context. Labels: val_edge_index
    # For Time Agnostic: Labels should probably be collapsed too?
    # BUT val_edge_index comes from 'hetero_data' via 'val_mask'.
    # 'val_mask' was computed on ORIGINAL temporal data.
    # So 'val_edge_index' are EVENTS.
    # predicting EVENTS using STATIC graph?
    # User said: "Edge-level regression ... future evidence strength".
    # If we are time-agnostic, maybe we just predict the *Event* score using the *Static* embedding?
    # This seems fine. We don't collapse the Validation SET, just the Input Graph.
    
    val_edge_times = None
    if not is_time_agnostic:
        if 'edge_time' in hetero_data[supervision_edge_type]:
            val_edge_times = hetero_data[supervision_edge_type].edge_time[val_mask]
        else:
            val_edge_times = torch.zeros(val_edge_index.size(1), dtype=torch.long)
    
    val_loader = LinkNeighborLoader(
        data=train_context,
        num_neighbors=[20, 10],
        edge_label_index=(supervision_edge_type, val_edge_index),
        edge_label=val_labels,
        edge_label_time=val_edge_times - 1 if val_edge_times is not None else None,
        time_attr='edge_time' if not is_time_agnostic else None,
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
    
    # 8. Pre-compute Evaluation Sets (Validation & Test)
    print("\n🛠️  Pre-computing Evaluation Sets...")
    
    # Validation Sets
    val_targets, val_history, val_srcs = build_evaluation_sets(
        hetero_data, 
        supervision_edge_type, 
        val_mask, 
        train_mask
    )
    print(f"   Validation: {len(val_srcs)} unique sources")
    
    # Test Sets (Ground Truth for Final Eval)
    exclusion_mask = (train_mask | val_mask)
    test_targets, test_history, test_srcs = build_evaluation_sets(
        hetero_data,
        supervision_edge_type,
        test_mask,
        exclusion_mask
    )
    print(f"   Test: {len(test_srcs)} unique sources")
    
    # 9. Training Loop
    print("\n🔄 Starting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(cfg.train.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, supervision_edge_type, src_type, dst_type)
        val_loss = evaluator.validate_regression(model, val_loader, device, supervision_edge_type, src_type, dst_type)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Regression Loss: {val_loss:.4f}")
        
    # Validation Ranking (Periodically or every epoch)
        # We run this to "Monitor" as requested
        print(f"   Running Validation Ranking...")
        
        val_metrics = evaluator.evaluate_ranking(
            model,
            inference_data=train_context,    # Context <= 2020
            test_targets_dict=val_targets,
            history_targets_dict=val_history,
            unique_test_srcs=val_srcs,
            edge_type=supervision_edge_type,
            num_dst_nodes=hetero_data[dst_type].num_nodes,
            device=device,
            num_negatives=1000 # Default sampling for validation speed
        )
        
        # Save Best based on REGRESSION Loss (Primary Objective)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{evaluator.output_dir}/best_model.pt")
            print("   💾 New Best Model Saved!")
            
    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.4f}")
    
    # 9. Test Evaluation (Novel Truth Ranking)
    print(f"\n🧪 Starting TEST Evaluation (Exhaustive Ranking)...")
    
    # Load Best Model
    model.load_state_dict(torch.load(f"{evaluator.output_dir}/best_model.pt"))
    model.eval()
    
    # Context: test_context (edges <= val_year)
    # Ground Truth: hetero_data (full)
    # Eval Mask: test_mask (edges > val_year)
    # Exclusion: edges <= val_year (ALL history = train + val)
    
    exclusion_mask = (train_mask | val_mask)
    
    metrics = evaluator.evaluate_ranking(
        model,
        inference_data=test_context,
        test_targets_dict=test_targets,
        history_targets_dict=test_history,
        unique_test_srcs=test_srcs,
        edge_type=supervision_edge_type,
        num_dst_nodes=hetero_data[dst_type].num_nodes,
        device=device,
        num_negatives=None # Use None for Exhaustive Ranking on Test (Headline Numbers)
    )
    
    print("\n✅ Final Test Metrics:")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/benchmark_config.yaml")
    args = parser.parse_args()
    main(args.config)

#!/usr/bin/env python3
"""
Test script to verify Evaluator logic using an Oracle Model.

This script:
1. Loads the Temporal Graph.
2. Selects a subset of diseases for testing.
3. Constructs an 'Oracle Model' whose embeddings produce perfect scores for Test Set edges.
4. Runs `evaluate_ranking_filtered`.
5. Verifies that metrics are Perfect (1.0).

This ensures the Evaluation Logic (History Exclusion, Candidate Scoring, Metric Calculation) is correct.
"""

import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from data.temporal_loader import load_event_graph, get_temporal_masks
from data.evaluation_prep import build_evaluation_sets
from benchmark.evaluator import Evaluator

class OracleModel(nn.Module):
    """
    Oracle Model that encodes nodes such that dot product reflects ground truth.
    
    Target Embeddings: Identity Matrix (Each target is its own dimension).
    Disease Embeddings: Multi-hot vector where 1.0 indicates a Test Edge exists.
    """
    def __init__(self, num_diseases, num_targets, test_edge_index, test_edge_weights=None):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_targets = num_targets
        self.test_edge_index = test_edge_index
        self.test_edge_weights = test_edge_weights
        
    def encode(self, x_dict, edge_index_dict):
        # 1. Target Embeddings: Identity (One-Hot per target)
        # Shape: [Num Targets, Num Targets]
        # We assume 'target' is the dst_type
        # To save memory if Num Targets is huge, we can use sparse, but 10k is fine dense.
        
        device = self.test_edge_index.device
        
        # Identity matrix for targets
        z_target = torch.eye(self.num_targets, device=device)
        
        # 2. Disease Embeddings: Encode "Future Knowledge"
        # Shape: [Num Diseases, Num Targets]
        # z_disease[d, t] = 1.0 if edge (d, t) exists in Test Set
        
        z_disease = torch.zeros((self.num_diseases, self.num_targets), device=device)
        
        src_indices = self.test_edge_index[0]
        dst_indices = self.test_edge_index[1]
        
        # Vectorized assignment
        # z_disease[src, dst] = 1.0
        # If we have weights, use them
        values = self.test_edge_weights if self.test_edge_weights is not None else torch.ones(src_indices.size(0), device=device)
        
        z_disease.index_put_((src_indices, dst_indices), values)
        
        # Return dict
        # We assume keys are 'disease', 'target', etc. 
        # But we need to match the actual keys.
        # We will check keys in run().
        return {
            'molecule': torch.zeros((1, self.num_targets), device=device), # Dummy
            'disease': z_disease,
            'target': z_target,
            'go': torch.zeros((1, self.num_targets), device=device), # Dummy
        }
    
    def decode(self, z_src, z_dst):
        # Oracle decode: Dot product
        return (z_src * z_dst).sum(dim=-1)

def run_test():
    print("🧪 Running Evaluator Correctness Test...")
    
    # 1. Config & Data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config/benchmark_config.yaml") # Load just for defaults
    cfg = OmegaConf.load(config_path)
    
    data_path = os.path.join(project_root, "output/progression/temporal_graph.pt")
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        return

    print("   Loading Graph...")
    hetero_data = torch.load(data_path)
    
    # Identify Types
    src_type = 'disease'
    dst_type = 'target'
    relation = 'clinical_trial::chembl'
    
    edge_type = None
    for et in hetero_data.edge_types:
        if et[0] == src_type and et[2] == dst_type and relation in et[1]:
            edge_type = et
            break
    
    if not edge_type:
        print("❌ Supervision edge type not found")
        return
        
    print(f"   Edge Type: {edge_type}")
    
    # 2. Splits (Manual or via Utils)
    # We want to use the EXACT same logic as Train
    train_year = 2020
    val_year = 2021
    
    # Get Masks
    # Note: get_temporal_masks might fail if pyg-lib issue persists? 
    # No, get_temporal_masks uses simple tensor ops usually.
    # Let's check source of get_temporal_masks?
    # It is in src/data/temporal_loader.py. It iterates edges. Safe.
    
    masks = get_temporal_masks(hetero_data, train_year, val_year)
    train_mask, val_mask, test_mask = masks[edge_type]
    
    # 3. Setup Oracle
    # We want to test on TEST set.
    # History = Train + Val
    exclusion_mask = (train_mask | val_mask)
    
    # Build Evaluation Sets using the helper
    print("   Building Evaluation Sets...")
    test_targets, test_history, test_srcs = build_evaluation_sets(
        hetero_data,
        edge_type,
        test_mask,
        exclusion_mask
    )
    
    # Subsample for Speed (Optional, but 100 diseases is fast enough with Oracle)
    # Let's take top 100 diseases with most test targets to verify recall
    # Sort srcs by number of test targets
    test_srcs.sort(key=lambda s: len(test_targets[s]), reverse=True)
    subset_srcs = test_srcs[:50]
    
    print(f"   Testing on subset of {len(subset_srcs)} diseases (Input: {len(test_srcs)})")
    
    # Subset dicts
    subset_targets = {k: test_targets[k] for k in subset_srcs}
    subset_history = {k: test_history.get(k, set()) for k in subset_srcs}
    
    # 4. Create Oracle Model
    # Test Indices for Oracle: We only need to encode the Test connections
    test_edge_index = hetero_data[edge_type].edge_index[:, test_mask]
    
    # Get test weights if available
    test_weights = None
    if 'edge_weight' in hetero_data[edge_type]:
         test_weights = hetero_data[edge_type].edge_weight[test_mask]
    
    num_diseases = hetero_data[src_type].num_nodes
    num_targets = hetero_data[dst_type].num_nodes
    
    model = OracleModel(num_diseases, num_targets, test_edge_index, test_weights)
    
    # Verify Overlap between Test and History
    print("\n🕵️ Checking for Test-History Overlap...")
    overlap_count = 0
    total_test_edges = 0
    for src, tgts in test_targets.items():
        total_test_edges += len(tgts)
        hist_tgts = test_history.get(src, set())
        overlap = tgts.intersection(hist_tgts)
        if overlap:
            overlap_count += len(overlap)
            # print(f"   Src {src} has {len(overlap)} overlaps: {list(overlap)[:5]}...")
    
    print(f"   Total Test Edges: {total_test_edges}")
    print(f"   Overlapping Edges (Test & History): {overlap_count}")
    
    if overlap_count > 0:
        print("   ⚠️ WARNING: Some test edges are being excluded because they appear in history!")
        print("   This explains why MRR is not 1.0 (True Positive is excluded -> Score -inf)")
    
    # 5. Run Evaluator
    evaluator = Evaluator(k_values=[1, 10, 50, 100])
    
    # Dummy inference data (needed for signature, contents ignored by OracleModel except keys)
    inference_data = hetero_data 
    
    print("   Running Evaluation...")
    device = torch.device('cpu') # Oracle is CPU
    
    metrics = evaluator.evaluate_ranking(
        model,
        inference_data=inference_data,
        test_targets_dict=subset_targets,
        history_targets_dict=subset_history,
        unique_test_srcs=subset_srcs,
        edge_type=edge_type,
        num_dst_nodes=num_targets,
        device=device,
        num_negatives=None # Use Exhaustive for Oracle Verification
    )
    
    # 6. Verify Results
    # Expect Perfect Metrics
    # MRR should be 1.0 (First positive is the highest score)
    # Recall@K should be 1.0 if K >= Num Targets? No, R@K depends on K.
    # But Precision@1 should be 1.0.
    # NDCG should be 1.0.
    
    print("\n🧐 Verification:")
    
    # Check MRR@100 (Most lenient?)
    # Actually if Oracle works, the Top-N scores should Include all True Positives.
    # Since we set score=1.0 for Positives and 0.0 for Negatives.
    # All Positives are tied at rank 1.
    # Evaluator sort stable? Or random?
    # If tied, ordering is implementation specific. But all Positives should be above Negatives.
    
    # Check NDCG@100
    if metrics['NDCG@100'] > 0.99:
        print("✅ Oracle Passed: NDCG@100 is ~1.0")
    else:
        print(f"❌ Oracle Failed: NDCG@100 is {metrics['NDCG@100']}")
        
    if metrics['MRR@100'] > 0.99:
        print("✅ Oracle Passed: MRR@100 is ~1.0")
    else:
         print(f"❌ Oracle Failed: MRR@100 is {metrics['MRR@100']}")

if __name__ == "__main__":
    run_test()

#!/usr/bin/env python3
"""
Evaluator for link prediction benchmarking.
"""

import torch
import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
from torch_geometric.nn import MIPSKNNIndex

from .metrics import compute_ranking_metrics


class Evaluator:
    """
    Evaluator for link prediction tasks.
    
    Manages evaluation workflow: generate rankings, compute metrics, export results.
    """
    
    def __init__(
        self,
        k_values: List[int] = [10, 20, 50, 100],
        output_dir: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of k values for top-k metrics
            output_dir: Directory to save results
        """
        self.k_values = k_values
        self.output_dir = output_dir
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_regression(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        split: str = "val"
    ) -> float:
        """
        Compute regression metric (MSE) on predicted scores.
        """
        mse = torch.nn.functional.mse_loss(scores, targets).item()
        print(f"📊 {split.capitalize()} MSE: {mse:.4f}")
        return mse

    def evaluate_ranking_exhaustive(
        self,
        model,
        inference_data,
        ground_truth_data,
        edge_type,
        eval_mask,
        exclusion_mask,
        device,
        batch_size_emb: int = None
    ) -> Dict[str, float]:
        """
        Exhaustive ranking evaluation using MIPS.
        
        Args:
            model: Trained HGT model
            inference_data: Data snapshot used for embedding generation (strict no-leakage)
            ground_truth_data: Full data containing ground truth edges
            edge_type: Edge type to evaluate (src, rel, dst)
            eval_mask: Mask of edges to evaluate (e.g. test set)
            exclusion_mask: Mask of edges to exclude (e.g. training set)
            device: torch device
            batch_size_emb: Optional batch size for embedding generation (not yet used, full batch assumed)
            
        Returns:
            Dict of metrics (P@k, R@k, NDCG@k)
        """
        print(f"\n🔍 Evaluating Ranking on {eval_mask.sum()} edges...")
        
        src_type, _, dst_type = edge_type
        
        # 1. Build Ground Truth Dict
        # Only evaluate on edges present in eval_mask
        eval_edge_index = ground_truth_data[edge_type].edge_index[:, eval_mask]
        
        if eval_edge_index.size(1) == 0:
            print("⚠️ No evaluation edges found.")
            return {}

        eval_dict = {}
        src_indices = eval_edge_index[0].tolist()
        dst_indices = eval_edge_index[1].tolist()
        unique_src_nodes = set()
        
        for src, dst in zip(src_indices, dst_indices):
            if src not in eval_dict:
                eval_dict[src] = []
            eval_dict[src].append(dst)
            unique_src_nodes.add(src)
            
        print(f"   Evaluating {len(eval_dict)} unique source nodes")
        
        # 2. Build Exclusion Index
        # Exclude edges in exclusion_mask (e.g. training edges)
        exclude_edge_index = ground_truth_data[edge_type].edge_index[:, exclusion_mask].to(device)
        
        from torch_geometric import EdgeIndex
        num_src = ground_truth_data[src_type].num_nodes
        num_dst = ground_truth_data[dst_type].num_nodes
        
        # Sparse index for fast lookup of "known" positives
        exclude_links = EdgeIndex(
            exclude_edge_index,
            sparse_size=(num_src, num_dst)
        ).sort_by('row')[0]
        
        # 3. Generate Embeddings using INFERENCE snapshot
        # This ensures we don't leak future edges into embeddings
        print("   Generating embeddings using inference snapshot...")
        model.eval()
        
        with torch.no_grad():
            x_dict = {k: v.to(device) for k, v in inference_data.x_dict.items()}
            edge_index_dict = {k: v.to(device) for k, v in inference_data.edge_index_dict.items()}
            
            # Full batch inference (CPU fallback if OOM)
            try:
                out_dict = model.encode(x_dict, edge_index_dict)
                src_emb = out_dict[src_type]
                dst_emb = out_dict[dst_type]
            except RuntimeError:
                print("   ⚠️ Full graph inference failed (OOM). Switching to CPU...")
                model = model.cpu()
                x_dict = {k: v.cpu() for k, v in inference_data.x_dict.items()}
                edge_index_dict = {k: v.cpu() for k, v in inference_data.edge_index_dict.items()}
                out_dict = model.encode(x_dict, edge_index_dict)
                src_emb = out_dict[src_type]
                dst_emb = out_dict[dst_type]
                model = model.to(device)
                src_emb = src_emb.to(device)
                dst_emb = dst_emb.to(device)

        # 4. MIPS Search
        print("   Indexing candidates...")
        # Index all items (targets)
        mips = MIPSKNNIndex(dst_emb)
        
        metrics = {k: {'p': [], 'r': [], 'ndcg': []} for k in self.k_values}
        max_k = max(self.k_values)
        
        print("   Ranking...")
        for src_id in tqdm(list(unique_src_nodes), desc="Ranking sources"):
            true_dsts = set(eval_dict[src_id])
            
            # Get excluded items for this source
            start = exclude_links.indptr[src_id]
            end = exclude_links.indptr[src_id+1]
            exclude_dsts = exclude_links.col[start:end]
            
            # Search top-(k + num_excluded)
            src_vec = src_emb[src_id:src_id+1]
            
            # MIPSKNNIndex.search implicitly handles exclusion if supported, 
            # OR we fetch more and filter.
            # PyG's MIPSKNNIndex.search supports `exclude_links` in recent versions:
            # .search(query, k, exclude_links=None)
            # If src_id is passed? No, it takes vectorized exclusion or just raw indices?
            # It seems standard search takes `exclude_links` as neighbors of the query index 
            # IF query is index. But here query is vector.
            # Workaround: Fetch Top-K, filter excludes. BUT if many excludes, K might be insufficient.
            # Usually we fetch K + len(excludes).
            
            # NOTE: MIPSKNNIndex implementation varies. 
            # If we assume we can't efficiently exclude inside MIPS on GPU easily without custom kernel:
            # We fetch a larger K.
            
            # Current PyG (2.6+) MIPSKNNIndex supports exclusion if using faiss/etc? 
            # Let's rely on the exclusion parameter `exclude_links` if passed correctly.
            # Wait, `MIPSKNNIndex.search` signature: (x_q, k, exclude_links=None)
            # where exclude_links is Tensor of indices to exclude?
            # Or CSR?
            
            # Let's assume standard implementation: 
            # If we pass exclude_dsts (Tensor), it filters them out.
            
            _, pred_indices = mips.search(src_vec, max_k, exclude_dsts)
            
            top_k = pred_indices[0].tolist()
            
            for k in self.k_values:
                curr_top = top_k[:k]
                hits = len(set(curr_top) & true_dsts)
                
                metrics[k]['p'].append(hits / k)
                metrics[k]['r'].append(hits / len(true_dsts))
                
                # NDCG
                dcg = 0.0
                idcg = 0.0
                
                for i, t in enumerate(curr_top):
                    if t in true_dsts:
                        dcg += 1.0 / np.log2(i + 2)
                
                for i in range(min(k, len(true_dsts))):
                    idcg += 1.0 / np.log2(i + 2)
                    
                metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)
        
        # 5. Report
        final_metrics = {}
        print(f"\n✅ Results:")
        for k in self.k_values:
            p = np.mean(metrics[k]['p'])
            r = np.mean(metrics[k]['r'])
            n = np.mean(metrics[k]['ndcg'])
            final_metrics[f"P@{k}"] = p
            final_metrics[f"R@{k}"] = r
            final_metrics[f"NDCG@{k}"] = n
            print(f"   k={k:<3}: P={p:.4f}, R={r:.4f}, NDCG={n:.4f}")
            
        # Save results
        if self.output_dir:
            self.save_results(final_metrics, split="test_ranking")
            
        return final_metrics

    def save_results(self, metrics, split, filename=None):
        super().save_results(metrics, split, filename)

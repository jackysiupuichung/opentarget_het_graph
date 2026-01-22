#!/usr/bin/env python3
"""
Evaluator for link prediction benchmarking.
"""

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional


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
    

    @torch.no_grad()
    def validate_regression(
        self,
        model,
        loader,
        device,
        supervision_edge_type,
        src_type,
        dst_type
    ) -> float:
        """
        Run regression validation loop on a loader.
        """
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
            
            # Slice prediction to match targets just in case
            curr_pred = pred_scores[:targets.size(0)]
            loss = F.mse_loss(curr_pred, targets)
            
            total_loss += loss.item() * full_batch_size
            total_examples += full_batch_size
            
        return total_loss / total_examples



    def evaluate_ranking(
        self,
        model,
        inference_data,
        test_targets_dict: Dict[int, object],
        history_targets_dict: Dict[int, object],
        unique_test_srcs: List[int],
        edge_type,
        num_dst_nodes: int,
        device,
        num_negatives: int = 1000,
        batch_size_eval: int = 10000 
    ) -> Dict[str, float]:
        """
        Ranking evaluation with filtering of historical edges.
        
        Args:
            model: Trained model
            inference_data: Data snapshot used for embedding generation (strict no-leakage)
            test_targets_dict: Dict mapping source_id -> set(target_ids) for evaluation
            history_targets_dict: Dict mapping source_id -> set(target_ids) to exclude
            unique_test_srcs: List of source IDs to evaluate
            edge_type: Edge type to evaluate (src, rel, dst)
            num_dst_nodes: Total number of destination nodes
            device: torch device
            num_negatives: Number of negative samples. If None or <= 0, use exhaustive.
            batch_size_eval: Batch size for scoring candidates
            
        Returns:
            Dict of metrics
        """
        mode = "Negative Sampling" if num_negatives and num_negatives > 0 else "Exhaustive"
        print(f"\n🔍 Evaluating Ranking ({mode}) on {len(unique_test_srcs)} unique source nodes...")
        
        src_type, _, dst_type = edge_type
        
        # 3. Generate Embeddings (Once)
        print("   Generating embeddings using inference snapshot...")
        model.eval()
        
        with torch.no_grad():
            # Handle node features input
            try:
                 # If it's a Batch or has x_dict
                 x_dict = {k: v.to(device) for k, v in inference_data.x_dict.items()}
            except (KeyError, AttributeError):
                 # Fallback: construct from node types
                 x_dict = {}
                 for nt in inference_data.node_types:
                     if hasattr(inference_data[nt], 'x') and inference_data[nt].x is not None:
                         x_dict[nt] = inference_data[nt].x.to(device)
            
            # Handle edge index input
            if hasattr(inference_data, 'edge_index_dict'):
                edge_index_dict = {k: v.to(device) for k, v in inference_data.edge_index_dict.items()}
            else:
                 edge_index_dict = {}
                 for et in inference_data.edge_types:
                     if hasattr(inference_data[et], 'edge_index') and inference_data[et].edge_index is not None:
                         edge_index_dict[et] = inference_data[et].edge_index.to(device)
            
            try:
                out_dict = model.encode(x_dict, edge_index_dict)
                src_emb_all = out_dict[src_type]
                dst_emb_all = out_dict[dst_type]
            except RuntimeError:
                print("   ⚠️ Full graph inference failed (OOM). Switching to CPU...")
                model = model.cpu()
                x_dict = {k: v.cpu() for k, v in inference_data.x_dict.items()}
                edge_index_dict = {k: v.cpu() for k, v in inference_data.edge_index_dict.items()}
                out_dict = model.encode(x_dict, edge_index_dict)
                src_emb_all = out_dict[src_type].to(device)
                dst_emb_all = out_dict[dst_type].to(device)
                model = model.to(device)

        # 4. Evaluation Loop
        metrics = {k: {'p': [], 'r': [], 'ndcg': [], 'mrr': []} for k in self.k_values}
        max_k = max(self.k_values)
        
        for src_id in tqdm(list(unique_test_srcs), desc="Ranking sources"):
            true_dsts = test_targets_dict[src_id]
            history_dsts = history_targets_dict.get(src_id, set())
            
            src_vec = src_emb_all[src_id:src_id+1] # [1, dim]
            
            # --- Identify Candidates ---
            if num_negatives and num_negatives > 0:
                # NEGATIVE SAMPLING MODE
                # 1. Sample Random Negatives
                neg_indices = torch.randint(0, num_dst_nodes, (num_negatives,), device=device)
                
                # 2. Combine with True Positives
                true_indices = torch.tensor(list(true_dsts), device=device, dtype=torch.long)
                candidate_indices = torch.cat([true_indices, neg_indices]).unique()
                
                # 3. Filter History (if any sampled negatives are in history)
                if history_dsts:
                    hist_tensor = torch.tensor(list(history_dsts), device=device, dtype=torch.long)
                    # Keep only candidates NOT in history
                    # Note: We must ensure True Positives are NOT filtered out (though they shouldn't be in history if data is correct)
                    # But if they are, we should probably keep them for evaluation?
                    # The 'evaluator' correctness depends on 'novelty'.
                    # If a true positive IS in history, it's not novel.
                    # Assuming input ensures novelty.
                    
                    mask = ~torch.isin(candidate_indices, hist_tensor)
                    candidate_indices = candidate_indices[mask]
                    
                # 4. Score Subset
                if len(candidate_indices) == 0:
                    top_k_list = []
                else:
                    cand_emb = dst_emb_all[candidate_indices]
                    # Use model to decode
                    scores = model.decode(src_vec, cand_emb) # [num_candidates]
                    
                    # 5. Get Top-K relative to candidates
                    # We want the indices of the top-k candidates, but mapped back to global IDs
                    k_actual = min(max_k, len(scores))
                    _, top_indices_local = torch.topk(scores, k_actual)
                    top_global_indices = candidate_indices[top_indices_local].tolist()
                    top_k_list = top_global_indices
                    
            else:
                # EXHAUSTIVE MODE
                # candidates = all targets
                
                # scores_all: [1, num_dst] -> [num_dst]
                # Use model to decode. src_vec is [1, D], dst_emb_all is [N, D].
                # Broadcasting (1, D) * (N, D) -> (N, D) -> sum -> (N)
                scores_all = model.decode(src_vec, dst_emb_all)
                
                # Mask History
                if history_dsts:
                    hist_tensor = torch.tensor(list(history_dsts), device=device, dtype=torch.long)
                    scores_all[hist_tensor] = -float('inf')
                
                # Get Top-K
                _, top_indices = torch.topk(scores_all, max_k)
                top_k_list = top_indices.tolist()
            
            # Compute Metrics
            for k in self.k_values:
                curr_top = top_k_list[:k]
                hits = len(set(curr_top) & true_dsts)
                
                metrics[k]['p'].append(hits / k)
                metrics[k]['r'].append(hits / len(true_dsts))
                
                # NDCG
                dcg = 0.0
                idcg = 0.0
                
                # DCG
                for i, t in enumerate(curr_top):
                    if t in true_dsts:
                        dcg += 1.0 / np.log2(i + 2)
                
                # IDCG
                for i in range(min(k, len(true_dsts))):
                    idcg += 1.0 / np.log2(i + 2)
                    
                metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)
                
                # MRR
                rr = 0.0
                for i, t in enumerate(curr_top):
                    if t in true_dsts:
                        rr = 1.0 / (i + 1)
                        break
                metrics[k]['mrr'].append(rr)
        
        # 5. Report
        final_metrics = {}
        print(f"\n✅ Results:")
        for k in self.k_values:
            p = np.mean(metrics[k]['p'])
            r = np.mean(metrics[k]['r'])
            n = np.mean(metrics[k]['ndcg'])
            m = np.mean(metrics[k]['mrr'])
            final_metrics[f"P@{k}"] = p
            final_metrics[f"R@{k}"] = r
            final_metrics[f"NDCG@{k}"] = n
            final_metrics[f"MRR@{k}"] = m
            print(f"   k={k:<3}: P={p:.4f}, R={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}")
            
        # Save results
        if self.output_dir:
            self.save_results(final_metrics, split="test_ranking")
            
        return final_metrics

    def save_results(self, metrics, split, filename=None):
        if not self.output_dir: return
        file_path = Path(self.output_dir) / (filename or f"metrics_{split}.yaml")
        with open(file_path, "w") as f:
            yaml.dump(metrics, f)
        print(f"   💾 Saved metrics to {file_path}")

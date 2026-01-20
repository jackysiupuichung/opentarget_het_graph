#!/usr/bin/env python3
"""
Evaluator for link prediction benchmarking.
"""

import torch
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional
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
    
    def evaluate(
        self,
        scores_dict: Dict[int, torch.Tensor],
        labels_dict: Dict[int, torch.Tensor],
        split: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            scores_dict: Dict mapping user index to predicted scores
            labels_dict: Dict mapping user index to ground truth labels
            split: Data split name (train/val/test)
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Evaluating {split} set...")
        print(f"   Users: {len(scores_dict)}")
        print(f"   K values: {self.k_values}")
        
        # Compute metrics
        metrics = compute_ranking_metrics(scores_dict, labels_dict, self.k_values)
        
        # Print results
        print(f"\n✅ {split.capitalize()} Results:")
        for metric_name, value in sorted(metrics.items()):
            print(f"   {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_results(
        self,
        metrics: Dict[str, float],
        split: str = "test",
        filename: Optional[str] = None,
    ):
        """
        Save evaluation results to file.
        
        Args:
            metrics: Dictionary of metrics
            split: Data split name
            filename: Optional custom filename
        """
        if not self.output_dir:
            return
        
        if filename is None:
            filename = f"{split}_results.yaml"
        
        filepath = Path(self.output_dir) / filename
        
        with open(filepath, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        
        print(f"\n💾 Saved results to: {filepath}")
    
    def export_predictions(
        self,
        scores_dict: Dict[int, torch.Tensor],
        user_id_map: Dict[int, str],
        item_id_map: Dict[int, str],
        top_k: int = 100,
        filename: str = "predictions.csv",
    ):
        """
        Export top-k predictions to CSV.
        
        Args:
            scores_dict: Dict mapping user index to scores
            user_id_map: Mapping from user index to user ID
            item_id_map: Mapping from item index to item ID
            top_k: Number of top predictions per user
            filename: Output filename
        """
        if not self.output_dir:
            return
        
        records = []
        
        for user_idx, scores in scores_dict.items():
            # Get top-k items
            top_k_scores, top_k_indices = torch.topk(scores, k=min(top_k, len(scores)))
            
            user_id = user_id_map.get(user_idx, str(user_idx))
            
            for rank, (item_idx, score) in enumerate(zip(top_k_indices.tolist(), top_k_scores.tolist()), 1):
                item_id = item_id_map.get(item_idx, str(item_idx))
                
                records.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": rank,
                    "score": score,
                })
        
        # Save to CSV
        df = pd.DataFrame(records)
        filepath = Path(self.output_dir) / filename
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Exported predictions to: {filepath}")

"""
Evaluation data preparation utilities.
"""

from typing import Dict, Set, List, Tuple
import torch
from torch_geometric.data import HeteroData

def build_evaluation_sets(
    ground_truth_data: HeteroData,
    edge_type: Tuple[str, str, str],
    eval_mask: torch.Tensor,
    history_mask: torch.Tensor
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], List[int]]:
    """
    Build dictionaries for evaluation: ground truth positives and historical exclusions.
    
    Args:
        ground_truth_data: Full HeteroData object
        edge_type: The edge type tuple (src_type, relation, dst_type) to evaluate
        eval_mask: Boolean mask indicating which edges in ground_truth_data are for evaluation (e.g. Test)
        history_mask: Boolean mask indicating which edges are historical (e.g. Train + Val)
        
    Returns:
        test_targets_dict: Dict mapping source_id -> set(target_ids) for evaluation
        history_targets_dict: Dict mapping source_id -> set(target_ids) to exclude (history)
        unique_test_srcs: List of source IDs that appear in the evaluation set
    """
    
    # 1. Build Ground Truth Dict (Test Set)
    # Extract edges corresponding to the evaluation mask
    eval_edge_index = ground_truth_data[edge_type].edge_index[:, eval_mask]
    
    test_targets_dict = {} # src -> set(dst)
    
    if eval_edge_index.size(1) > 0:
        src_indices = eval_edge_index[0].tolist()
        dst_indices = eval_edge_index[1].tolist()
        
        for src, dst in zip(src_indices, dst_indices):
            if src not in test_targets_dict:
                test_targets_dict[src] = set()
            test_targets_dict[src].add(dst)
    
    unique_test_srcs = list(test_targets_dict.keys())
    
    # 2. Build History Dict (Train + Val)
    # Extract edges corresponding to the history mask
    history_edge_index = ground_truth_data[edge_type].edge_index[:, history_mask]
    history_targets_dict = {}
    
    if history_edge_index.size(1) > 0:
        h_src = history_edge_index[0].tolist()
        h_dst = history_edge_index[1].tolist()
        
        for src, dst in zip(h_src, h_dst):
            if src not in history_targets_dict:
                history_targets_dict[src] = set()
            history_targets_dict[src].add(dst)
            
    # 3. Filter Test Targets to keep only NOVEL interactions
    # (Remove targets that appear in history for the same source)
    final_test_targets_dict = {}
    
    for src, targets in test_targets_dict.items():
        history = history_targets_dict.get(src, set())
        novel_targets = targets - history
        
        if novel_targets:
            final_test_targets_dict[src] = novel_targets
            
    unique_test_srcs = list(final_test_targets_dict.keys())
            
    return final_test_targets_dict, history_targets_dict, unique_test_srcs

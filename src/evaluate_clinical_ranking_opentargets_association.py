#!/usr/bin/env python3
"""
Novel Target Prioritization Evaluator using OpenTargets Association Scores.

Uses pre-computed association scores from OpenTargets parquet files instead of 
trained model predictions for baseline evaluation.
"""

import sys
import torch
import pandas as pd
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from glob import glob

from torch_geometric.data import HeteroData


def extract_labels_from_graph(graph, split_year, node_mappings):
    """
    Dynamically extract labels from graph edges up to a specific year.
    (Duplicated from train_clinical_multitask.py for self-containment)
    """
    print(f"   Extracting labels up to year {split_year}...")
    
    # Task to edge type mapping
    # Check if graph uses 'relation::datasource' or 'relation_only'
    # Try to find 'clinical_trial_positive' first
    
    tasks = ['pos', 'unmet', 'adv', 'op']
    relations = {
        'pos': 'clinical_trial_positive',
        'unmet': 'clinical_trial_unmet_efficacy',
        'adv': 'clinical_trial_adverse_effects',
        'op': 'clinical_trial_Unknown/Operational'
    }
    
    task_edge_map = {}
    for t in tasks:
        base_rel = relations[t]
        # Check simplified name first
        if ('disease', base_rel, 'target') in graph.edge_types:
            task_edge_map[t] = base_rel
        # Check suffixed name
        elif ('disease', f"{base_rel}::chembl", 'target') in graph.edge_types:
            task_edge_map[t] = f"{base_rel}::chembl"
            
    print(f"   Detected {len(task_edge_map)} valid clinical trial edge types.")
    
    label_data = {}
    
    for task, edge_type_name in task_edge_map.items():
        etype = ('disease', edge_type_name, 'target')
        
        if etype not in graph.edge_types:
            continue
            
        edge_store = graph[etype]
        edge_index = edge_store.edge_index
        edge_attr = edge_store.edge_attr
        edge_time = edge_store.edge_time if hasattr(edge_store, 'edge_time') else None
        
        if edge_time is None:
             mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        else:
             mask = edge_time <= int(split_year)
        
        filtered_indices = edge_index[:, mask]
        filtered_attr = edge_attr[mask] if edge_attr is not None else torch.ones(mask.sum(), 1)
        
        num_edges = filtered_indices.size(1)
        
        src_indices = filtered_indices[0].cpu().numpy()
        dst_indices = filtered_indices[1].cpu().numpy()
        scores = filtered_attr.squeeze().cpu().numpy()
        if scores.ndim == 0: scores = np.array([scores])
        
        for i in range(num_edges):
            d_idx = int(src_indices[i])
            t_idx = int(dst_indices[i])
            score = float(scores[i])
            
            key = (d_idx, t_idx)
            if key not in label_data:
                label_data[key] = {t: 0.0 for t in task_edge_map.keys()}
            label_data[key][task] = max(label_data[key][task], score)

    # Compute max score across all outcomes for each pair (ground truth)
    max_scores = {}
    for key, task_scores in label_data.items():
        max_scores[key] = max(task_scores.values())
    
    return label_data, max_scores


def load_opentargets_associations(parquet_dir, node_mappings, max_year=None):
    """
    Load all parquet files iteratively and create disease-target score mapping.
    Optimized for memory usage by processing one file at a time.
    
    Args:
        parquet_dir: Directory containing OpenTargets association parquet files
        node_mappings: Dict with 'disease' and 'target' ID to index mappings
        max_year: Optional year cutoff - only include associations up to this year
        
    Returns:
        dict: {(disease_idx, target_idx): score}
    """
    print(f"\n📂 Loading OpenTargets associations from {parquet_dir}...")
    if max_year is not None:
        print(f"   Filtering to associations <= year {max_year}")
    
    parquet_files = glob(str(Path(parquet_dir) / "*.parquet"))
    print(f"   Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {parquet_dir}")
    
    # Map IDs to graph indices using sets for faster lookup
    disease_mapping = node_mappings['disease']
    target_mapping = node_mappings['target']
    
    valid_diseases = set(disease_mapping.keys())
    valid_targets = set(target_mapping.keys())
    
    association_scores = {}
    total_processed = 0
    total_mapped = 0
    
    # Process files iteratively
    for pf in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            # Read only essential columns
            columns = ['diseaseId', 'targetId', 'score']
            if max_year is not None:
                columns.append('year')
                
            df = pd.read_parquet(pf, columns=columns)
            
            if df.empty: continue
            
            # Filter by year if needed
            if max_year is not None:
                df = df[df['year'] <= max_year]
                
            if df.empty: continue
            
            # Pre-filter rows where both IDs are in our mapping
            # This drastically reduces rows before the slow iterrows loop
            mask = df['diseaseId'].isin(valid_diseases) & df['targetId'].isin(valid_targets)
            df_filtered = df[mask].copy()
            
            if df_filtered.empty: continue
            
            # Handle NaN scores
            df_filtered['score'] = df_filtered['score'].fillna(0.0)
            
            total_processed += len(df)
            total_mapped += len(df_filtered)
            
            # Update dictionary
            # Using vectorization via mapping then manual update is faster but dict update is easier
            # Let's do a semi-vectorized approach for speed
            
            # Map IDs to indices
            # We can use map() but let's be careful with missing keys (already filtered)
            d_indices = df_filtered['diseaseId'].map(disease_mapping)
            t_indices = df_filtered['targetId'].map(target_mapping)
            scores = df_filtered['score'].values
            
            # Update dict
            for d_idx, t_idx, score in zip(d_indices, t_indices, scores):
                key = (d_idx, t_idx)
                if key in association_scores:
                    association_scores[key] = max(association_scores[key], float(score))
                else:
                    association_scores[key] = float(score)
            
            # Explicit garbage collection hint
            del df, df_filtered
            
        except Exception as e:
            print(f"⚠️ Error processing {pf}: {e}")
            continue

    print(f"   Total associations processed: {total_processed:,}")
    print(f"   Mapped associations: {len(association_scores):,}")
    
    return association_scores


def evaluate_ranking_with_scores(
    association_scores,
    train_pairs, 
    test_pairs, 
    num_disease_nodes, 
    num_target_nodes, 
    k_values=[100, 200, 500]
):
    """
    Evaluate ranking metrics per disease using pre-computed association scores.
    
    For each disease, creates a candidate set of novel targets (excluding train+val history),
    scores only those candidates using association scores, and computes ranking metrics.
    """
    # Pre-compute metrics storage
    metrics = {k: {'precision': [], 'recall': [], 'hits': [], 'mrr': [], 'ndcg': []} for k in k_values}
    
    # Identify test diseases (diseases that have at least one test pair)
    test_diseases = set(d for d, t in test_pairs.keys())
    
    print(f"\n🔍 Evaluating Ranking on {len(test_diseases)} diseases...")
    print(f"   K values: {k_values}")
    
    # Pre-organize ground truth: disease -> set(target_indices)
    # test_pairs now contains max_scores (float values)
    test_ground_truth = {}
    for (d, t), max_score in test_pairs.items():
        if max_score > 0:  # Only consider pairs with actual clinical trial activity
            if d not in test_ground_truth: test_ground_truth[d] = set()
            test_ground_truth[d].add(t)
        
    # Pre-organize history: disease -> set(target_indices)
    history_map = {}
    for (d, t) in train_pairs.keys():
        if d not in history_map: history_map[d] = set()
        history_map[d].add(t)
    
    all_target_indices = set(range(num_target_nodes))
    
    # Loop over diseases
    for d_idx in tqdm(test_diseases):
        true_targets = test_ground_truth.get(d_idx, set())
        if len(true_targets) == 0:
            continue  # Skip diseases with no positive test pairs
            
        history = history_map.get(d_idx, set())
        
        # Build candidate set: all targets EXCEPT those in train+val history
        candidate_targets = all_target_indices - history
        candidate_list = sorted(list(candidate_targets))
        
        if len(candidate_list) == 0:
            continue  # Skip if no candidates
        
        # Get scores for candidates using association scores (default 0.0 if not found)
        candidate_scores = []
        for t_idx in candidate_list:
            score = association_scores.get((d_idx, t_idx), 0.0)
            candidate_scores.append(score)
        
        # Rank candidates by score
        candidate_scores = np.array(candidate_scores)
        sorted_indices = np.argsort(-candidate_scores)  # Descending order
        
        # Get top-k
        max_k = min(max(k_values), len(candidate_list))
        top_k_local_indices = sorted_indices[:max_k]
        top_k_indices = [candidate_list[i] for i in top_k_local_indices]
        
        # Metrics
        for k in k_values:
            k_actual = min(k, len(top_k_indices))
            curr_top = top_k_indices[:k_actual]
            intersects = len(set(curr_top) & true_targets)
            
            # Recall@K
            if len(true_targets) > 0:
                recall = intersects / len(true_targets)
            else:
                recall = 0.0
            metrics[k]['recall'].append(recall)
            
            # Precision@K
            precision = intersects / k_actual if k_actual > 0 else 0.0
            metrics[k]['precision'].append(precision)
            
            # Hits@K
            metrics[k]['hits'].append(1.0 if intersects > 0 else 0.0)
            
            # MRR
            rr = 0.0
            for rank, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    rr = 1.0 / (rank + 1)
                    break
            metrics[k]['mrr'].append(rr)
            
            # NDCG
            dcg = 0.0
            idcg = 0.0
            
            # DCG
            for i, t_idx in enumerate(curr_top):
                if t_idx in true_targets:
                    dcg += 1.0 / np.log2(i + 2)
            
            # IDCG (Perfect ranking)
            num_relevant = min(k_actual, len(true_targets))
            for i in range(num_relevant):
                idcg += 1.0 / np.log2(i + 2)
                
            metrics[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0)

    # Average metrics
    final_results = {}
    print(f"\n📊 Ranking Results (OpenTargets Association Scores):")
    print(f"\n   MACRO-AVERAGED (average of per-disease metrics):")
    for k in k_values:
        avg_rec = np.mean(metrics[k]['recall'])
        avg_prec = np.mean(metrics[k]['precision'])
        avg_mrr = np.mean(metrics[k]['mrr'])
        avg_ndcg = np.mean(metrics[k]['ndcg'])
        
        final_results[f'Macro-Recall@{k}'] = float(avg_rec)
        final_results[f'Macro-Precision@{k}'] = float(avg_prec)
        final_results[f'Macro-MRR@{k}'] = float(avg_mrr)
        final_results[f'Macro-NDCG@{k}'] = float(avg_ndcg)
        
        print(f"   K={k:<3}: Recall={avg_rec:.4f} | Precision={avg_prec:.4f} | MRR={avg_mrr:.4f} | NDCG={avg_ndcg:.4f}")
    
    # Compute micro-averaged metrics (aggregate first, then compute)
    print(f"\n   MICRO-AVERAGED (aggregate across all diseases):")
    
    # Re-organize data for micro-averaging
    test_ground_truth = {}
    for (d, t), max_score in test_pairs.items():
        if max_score > 0:
            if d not in test_ground_truth: test_ground_truth[d] = set()
            test_ground_truth[d].add(t)
    
    history_map = {}
    for (d, t) in train_pairs.keys():
        if d not in history_map: history_map[d] = set()
        history_map[d].add(t)
    
    test_diseases = set(d for d, t in test_pairs.keys())
    all_target_indices = set(range(num_target_nodes))
    
    for k in k_values:
        total_hits = 0
        total_true_targets = 0
        total_k = 0
        
        for d_idx in test_diseases:
            true_targets = test_ground_truth.get(d_idx, set())
            if len(true_targets) == 0:
                continue
            
            history = history_map.get(d_idx, set())
            candidate_targets = all_target_indices - history
            candidate_list = sorted(list(candidate_targets))
            
            if len(candidate_list) == 0:
                continue
            
            # Get scores
            candidate_scores = []
            for t_idx in candidate_list:
                score = association_scores.get((d_idx, t_idx), 0.0)
                candidate_scores.append(score)
            
            candidate_scores = np.array(candidate_scores)
            sorted_indices = np.argsort(-candidate_scores)
            
            # Get top-k
            k_actual = min(k, len(candidate_list))
            top_k_local = sorted_indices[:k_actual]
            top_k_targets = [candidate_list[i] for i in top_k_local]
            
            # Count hits
            hits = len(set(top_k_targets) & true_targets)
            total_hits += hits
            total_true_targets += len(true_targets)
            total_k += k_actual
        
        # Micro-averaged metrics
        micro_recall = total_hits / total_true_targets if total_true_targets > 0 else 0
        micro_precision = total_hits / total_k if total_k > 0 else 0
        
        final_results[f'Micro-Recall@{k}'] = float(micro_recall)
        final_results[f'Micro-Precision@{k}'] = float(micro_precision)
        
        print(f"   K={k:<3}: Recall={micro_recall:.4f} | Precision={micro_precision:.4f}")
        
    return final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config (yaml)")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.eval.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Novel Target Prioritization Evaluator (OpenTargets Baseline)")
    print(f"   Config: {args.config}")
    
    # 1. Load Data
    print(f"\n📂 Loading graph data...")
    graph = torch.load(cfg.data.graph_file, weights_only=False)
    mappings = torch.load(cfg.data.mappings_file, weights_only=False)
    node_mappings = mappings['node_mapping']
    
    # Load validation diseases for filtering
    val_diseases_path = cfg.data.validation_diseases_file
    print(f"📋 Loading validation diseases from {val_diseases_path}...")
    val_diseases_df = pd.read_csv(val_diseases_path)
    # Filter out diseases not in graph (graph_node_idx == -1)
    val_diseases_df = val_diseases_df[val_diseases_df['graph_node_idx'] != -1]
    validation_disease_indices = set(val_diseases_df['graph_node_idx'].tolist())
    print(f"   Loaded {len(validation_disease_indices)} validation diseases for benchmark")
    
    # 2. Extract Edges
    ts = cfg.data.temporal_split
    
    if hasattr(ts, 'val') and ts.val is not None:
        history_year = ts.val[1] # End of validation (e.g., 2020)
        print(f"   Using Validation End Year ({history_year}) as History Cutoff.")
    else:
        history_year = ts.train[1]
        print(f"   Using Training End Year ({history_year}) as History Cutoff (No val split found).")
        
    test_year = ts.test[1]
    
    print(f"\n📊 Extracting History (All edges <= {history_year})...")
    history_data, _ = extract_labels_from_graph(graph, history_year, node_mappings)
    
    print(f"📊 Extracting Test Candidates (Edges <= {test_year})...")
    full_test_data, full_test_max_scores = extract_labels_from_graph(graph, test_year, node_mappings)
    
    # Novel Test = (Edges <= Test) - (Edges <= History)
    # Ground truth uses max score across all outcomes
    # Filter to only validation diseases
    test_novel_data = {
        k: full_test_max_scores[k] for k in full_test_data.keys() 
        if k not in history_data and k[0] in validation_disease_indices
    }
    
    print(f"   Total Edges in Test Period Window: {len(full_test_data):,}")
    print(f"   Known History Edges: {len(history_data):,}")
    print(f"   Strictly Novel Test Edges: {len(test_novel_data):,} (History Removed, Validation Diseases Only)")
    print(f" Number of diseases with novel edges: {len(set([k[0] for k in test_novel_data.keys()]))}")
    if len(test_novel_data) == 0:
        print("❌ No novel edges found in test split! Check temporal splits.")
        sys.exit(1)

    # Load OpenTargets associations up to history cutoff (no temporal leakage)
    association_dir = cfg.data.association_dir
    association_scores = load_opentargets_associations(association_dir, node_mappings, max_year=history_year)

    # 3. Evaluate
    results = evaluate_ranking_with_scores(
        association_scores,
            history_data, # Use full history (Train+Val) to mask
        test_novel_data, 
        graph['disease'].num_nodes,
        graph['target'].num_nodes,
        k_values=[100, 200, 500]
    )
    
    # 4. Save
    out_file = output_dir / "results_ranking_opentargets_baseline.yaml"
    with open(out_file, 'w') as f:
        yaml.dump(results, f)
    print(f"\n✅ Saved ranking results to {out_file}")


if __name__ == "__main__":
    main()

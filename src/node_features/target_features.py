#!/usr/bin/env python3
"""
Build target features from raw data files.
Generates .pt files in data/node_features/processed/
"""

import os
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

target_prioritisation_features = [
        'tissueDistribution', 'geneticConstraint', 'mouseOrthologMaxIdentityPercentage'
    ]

def build_target_prioritisation_features(
    input_path: str
) -> dict:
    print(f"\n🏗️  Building Target Prioritisation Features...")
    print(f"   Input: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"   ❌ Input not found: {input_path}")
        return {}

    df = pd.read_parquet(input_path)
    print(f"   Loaded shape: {df.shape}")
        
    # Filter to available columns
    available_cols = [c for c in target_prioritisation_features if c in df.columns]
    print(f"   Using columns: {available_cols}")
    
    df_feats = df.set_index('targetId')[available_cols]
    
    # Convert to Tensor Dictionary
    feature_dict = {}
    for target_id, row in df_feats.iterrows():
        tensor = torch.tensor(row.values, dtype=torch.float)
        feature_dict[str(target_id)] = tensor
        
    print(f"   Processed {len(feature_dict)} targets.")
    print(f"   Feature dim: {len(available_cols)}")
    
    return feature_dict



def calculate_jss(df_pivot):
    """
    Calculate Jensen-Shannon Specificity (JSS) for each gene.
    """
    print("   Calculating JSS for validation...")
    
    # Normalize to probabilities (p_i)
    p = df_pivot.values + 1e-9
    row_sums = p.sum(axis=1, keepdims=True)
    p = p / row_sums
    
    # Uniform distribution (q_i)
    n_cell_tests = p.shape[1]
    q = np.full_like(p, 1.0 / n_cell_tests)
    
    # Calculate JSD (using scipy, default base e) -> convert to base 2
    # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    m = 0.5 * (p + q)
    
    # Manual JSD base 2
    kl_pm = np.sum(p * np.log2(p / m), axis=1)
    kl_qm = np.sum(q * np.log2(q / m), axis=1)
    
    jsd = 0.5 * (kl_pm + kl_qm)
    jsd = np.maximum(jsd, 0.0)
    
    jss = np.sqrt(jsd)
    
    print(f"   JSS stats: Min={jss.min():.4f}, Max={jss.max():.4f}, Mean={jss.mean():.4f}")
    return jss

def build_rna_expression_features(
    input_path: str
) -> dict:
    print(f"\n🏗️  Building RNA Expression Features...")
    print(f"   Input: {input_path}")

    if not os.path.exists(input_path):
        print(f"   ❌ Input not found: {input_path}")
        return {}

    try:
        df = pd.read_csv(input_path, sep="\t", compression="zip")
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return {}
        
    print(f"   Loaded shape: {df.shape}")
    
    # Pivot
    pivot = df.pivot_table(index='Gene', columns='Cell type group', values='nCPM', fill_value=0.0)
    print(f"   Pivoted shape: {pivot.shape}")
    
    # Calculate JSS for validation
    calculate_jss(pivot)
    
    # Log transform for features
    pivot_log = np.log1p(pivot)
    
    # Convert to Tensor Dictionary
    feature_dict = {}
    for gene_id, row in pivot_log.iterrows():
        tensor = torch.tensor(row.values, dtype=torch.float)
        feature_dict[str(gene_id)] = tensor
        
    print(f"   Processed {len(feature_dict)} genes.")
    print(f"   Feature dim: {pivot.shape[1]}")
    
    return feature_dict



def build_integrated_features(base_dir: Path, output_path: str, target_ids: list = None):
    print(f"\n🔗 Building INTEGRATED TARGET FEATURES...")
    
    # 1. Load Prio (Dict[str, Tensor])
    prio_path = str(base_dir / "target_prioritisation")
    # Check if it's a directory with parquet files or a single file
    if Path(prio_path).is_dir():
        from glob import glob
        prio_files = glob(f"{prio_path}/part-*.parquet")
        if prio_files:
            print(f"   Loading {len(prio_files)} prioritisation files...")
            import pandas as pd
            prio_dfs = [pd.read_parquet(f) for f in prio_files]
            prio_df = pd.concat(prio_dfs, ignore_index=True)
            # Convert to dict format
            prio_feats = {}
            for _, row in prio_df.iterrows():
                target_id = str(row['targetId'])
                # Extract ONLY the specified target_prioritisation_features
                feat_cols = [c for c in target_prioritisation_features if c in prio_df.columns]
                feat_vals = row[feat_cols].values.astype(float)  # Convert to float first
                prio_feats[target_id] = torch.tensor(feat_vals, dtype=torch.float)
            print(f"   Loaded {len(prio_feats)} targets from prioritisation")
            print(f"   Using features: {feat_cols}")
        else:
            raise ValueError(f"No prioritisation files found at {prio_path}")
    else:
        raise ValueError(f"Prioritisation path {prio_path} is not a directory")

    # 2. Load RNA (Dict[str, Tensor])
    rna_path = str(base_dir / "rna_expression/rna_single_cell_type_group.tsv.zip")
    if not os.path.exists(rna_path):
        raise ValueError(f"RNA path {rna_path} does not exist")
    rna_feats = build_rna_expression_features(
        input_path=rna_path
    )
    
    if not prio_feats and not rna_feats:
        print("❌ No features available.")
        return

    # Determine dimensions
    dim_prio = 0
    if prio_feats:
        dim_prio = next(iter(prio_feats.values())).shape[0]
        
    dim_rna = 0
    if rna_feats:
        dim_rna = next(iter(rna_feats.values())).shape[0]
        
    print(f"   Combining: Prio ({dim_prio}) + RNA ({dim_rna}) -> Total ({dim_prio + dim_rna})")
    
    # Filter to requested target IDs
    target_set = set(target_ids)
    all_keys = target_set
    print(f"\n📊 Filtering to {len(target_set):,} requested target IDs")
    
    # Compute Global Means for Imputation
    print(f"\n   Computing global means for imputation...")
    
    if prio_feats:
        all_prio = torch.stack(list(prio_feats.values()))
        mean_prio = all_prio.mean(dim=0)
    else:
        mean_prio = torch.zeros(dim_prio)
        
    if rna_feats:
        all_rna = torch.stack(list(rna_feats.values()))
        mean_rna = all_rna.mean(dim=0)
    else:
        mean_rna = torch.zeros(dim_rna)
        
    # Construct Integrated Dict
    integrated_dict = {}
    missing_prio = 0
    missing_rna = 0
    
    for key in all_keys:
        # Get Prio (or Mean)
        if key in prio_feats:
            v_prio = prio_feats[key]
        else:
            v_prio = mean_prio
            missing_prio += 1
            
        # Get RNA (or Mean)
        if key in rna_feats:
            v_rna = rna_feats[key]
        else:
            v_rna = mean_rna
            missing_rna += 1
        
        combined = torch.cat([v_prio, v_rna])
        integrated_dict[key] = combined
    
    # Compute per-column statistics for Prio features ONLY
    if prio_feats:
        print(f"\n   📊 Prio Feature Statistics (target_prioritisation_features only):")
        all_prio_stacked = torch.stack(list(prio_feats.values()))  # [num_targets, num_features]
        
        # Use the global target_prioritisation_features list
        for i, col_name in enumerate(target_prioritisation_features):
            if i >= all_prio_stacked.shape[1]:
                print(f"      [{i}] {col_name:45s} | NOT FOUND IN DATA")
                continue
                
            col_values = all_prio_stacked[:, i]
            
            # Filter out NaNs for statistics computation
            valid_values = col_values[~torch.isnan(col_values)]
            nan_count = torch.isnan(col_values).sum().item()
            
            if len(valid_values) > 0:
                col_mean = valid_values.mean().item()
                col_std = valid_values.std().item()
                col_min = valid_values.min().item()
                col_max = valid_values.max().item()
                
                print(f"      [{i}] {col_name:45s} | "
                      f"mean={col_mean:8.4f}, std={col_std:8.4f}, "
                      f"min={col_min:8.4f}, max={col_max:8.4f}, "
                      f"NaNs={nan_count}/{len(col_values)} ({nan_count/len(col_values)*100:.1f}%)")
            else:
                print(f"      [{i}] {col_name:45s} | ALL NaN ({len(col_values)} values)")
    
    print(f"\n   Feature Coverage:")
    print(f"   - Missing Prio: {missing_prio}/{len(all_keys)} ({missing_prio/len(all_keys)*100:.1f}%)")
    print(f"   - Missing RNA:  {missing_rna}/{len(all_keys)} ({missing_rna/len(all_keys)*100:.1f}%)")
        
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(integrated_dict, output_path)
    print(f"   ✅ Saved {len(integrated_dict)} integrated vectors to {output_path}")

if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data/node_features")
    parser.add_argument("--output-dir", default="data/node_features/processed")
    parser.add_argument("--target-ids-file", default=None, help="Parquet file with target IDs to filter to")
    args = parser.parse_args()
    
    # Load target IDs if provided
    target_ids = None
    if args.target_ids_file and Path(args.target_ids_file).exists():
        print(f"Loading target IDs from {args.target_ids_file}")
        df = pd.read_parquet(args.target_ids_file)
        target_ids = df['id'].tolist()
        print(f"Filtering to {len(target_ids):,} target IDs")
    
    build_integrated_features(
        base_dir=Path(args.base_dir),
        output_path=str(Path(args.output_dir) / "integrated_target_features.pt"),
        target_ids=target_ids
    )


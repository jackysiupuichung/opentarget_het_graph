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
    
    # Select features
    cols = [
        'isInMembrane', 'isSecreted', 'maxClinicalTrialPhase', 
        'tissueSpecificity', 'tissueDistribution', 
        'geneticConstraint', 'mouseOrthologMaxIdentityPercentage'
    ]
    
    # Filter to available columns
    available_cols = [c for c in cols if c in df.columns]
    print(f"   Using columns: {available_cols}")
    
    df_feats = df.set_index('targetId')[available_cols]
    
    # Handle NaNs: 
    # Mean imputation for numeric columns
    print("   Performing mean imputation for missing values...")
    df_feats = df_feats.fillna(df_feats.mean())
    
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



def build_integrated_features(base_dir: Path, output_path: str):
    print(f"\n🔗 Building INTEGRATED TARGET FEATURES...")
    
    # 1. Load Prio (Dict[str, Tensor])
    prio_feats = build_target_prioritisation_features(
        input_path=str(base_dir / "target_prioritisation/subset.parquet")
    )

    # 2. Load RNA (Dict[str, Tensor])
    rna_feats = build_rna_expression_features(
        input_path=str(base_dir / "rna_expression/rna_single_cell_type_group.tsv.zip")
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
    
    # Analyze Overlap
    keys_prio = set(prio_feats.keys())
    keys_rna = set(rna_feats.keys())
    all_keys = keys_prio | keys_rna
    
    print(f"\n📊 ID Overlap Analysis:")
    print(f"   Total Unique IDs: {len(all_keys)}")
    print(f"   Common (Both):    {len(keys_prio & keys_rna)}")
    
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
    
    for key in all_keys:
        # Get Prio (or Mean)
        v_prio = prio_feats.get(key, mean_prio)
        # Get RNA (or Mean)
        v_rna = rna_feats.get(key, mean_rna)
        
        combined = torch.cat([v_prio, v_rna])
        integrated_dict[key] = combined
        
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(integrated_dict, output_path)
    print(f"   ✅ Saved {len(integrated_dict)} integrated vectors to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data/node_features")
    parser.add_argument("--output-dir", default="data/node_features/processed")
    args = parser.parse_args()
    
    build_integrated_features(
        base_dir=Path(args.base_dir),
        output_path=str(Path(args.output_dir) / "integrated_target_features.pt")
    )

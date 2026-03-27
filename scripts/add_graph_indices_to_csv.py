#!/usr/bin/env python3
"""
Add graph node index mapping to validation_diseases.csv.

This script adds a 'graph_node_idx' column to the validation diseases CSV
with the mapped node indices from the temporal graph.
"""

import pandas as pd
import torch
from pathlib import Path


def main():
    print("\n" + "="*80)
    print("ADDING GRAPH NODE INDEX TO VALIDATION DISEASES CSV")
    print("="*80 + "\n")
    
    project_root = Path(__file__).parent.parent
    
    # Paths
    validation_csv = project_root / "data" / "validation_diseases.csv"
    mapping_file = project_root / "output" / "progression" / "temporal_graph_mappings.pt"
    output_csv = validation_csv  # Overwrite original
    
    # Load validation diseases
    print(f"📋 Loading {validation_csv}")
    val_df = pd.read_csv(validation_csv)
    print(f"   Found {len(val_df)} diseases")
    
    # Load node mapping
    print(f"\n🗺️  Loading {mapping_file}")
    mappings = torch.load(mapping_file, weights_only=False)
    disease_mapping = mappings['node_mapping']['disease']
    print(f"   Found {len(disease_mapping)} disease nodes in graph")
    
    # Map to indices
    print(f"\n🔗 Mapping disease IDs to graph indices...")
    graph_indices = []
    mapped_count = 0
    missing_count = 0
    
    for idx, row in val_df.iterrows():
        disease_id = row['EFO_ID']
        if disease_id in disease_mapping:
            graph_idx = disease_mapping[disease_id]
            graph_indices.append(graph_idx)
            mapped_count += 1
        else:
            graph_indices.append(-1)  # Use -1 for missing
            missing_count += 1
            print(f"   ⚠️  Missing: {row['Disease']} ({disease_id})")
    
    # Add column
    val_df['graph_node_idx'] = graph_indices
    
    print(f"\n✅ Mapping complete:")
    print(f"   Mapped: {mapped_count} / {len(val_df)} ({mapped_count/len(val_df)*100:.1f}%)")
    print(f"   Missing: {missing_count} (marked as -1)")
    
    # Save
    print(f"\n💾 Saving to {output_csv}")
    val_df.to_csv(output_csv, index=False)
    
    print(f"\n📊 Updated CSV columns: {list(val_df.columns)}")
    print(f"\nFirst 5 rows:")
    print(val_df[['Disease', 'EFO_ID', 'graph_node_idx']].head())
    
    print("\n" + "="*80)
    print("✅ COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

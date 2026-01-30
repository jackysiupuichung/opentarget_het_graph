#!/usr/bin/env python3
"""
Master script to build all node features.
Extracts node IDs from graph mappings and invokes individual feature builders.
"""

import argparse
import subprocess
import sys
import torch
import pandas as pd
from pathlib import Path

def run_script(script_path: str, args: list):
    cmd = [sys.executable, script_path] + args
    print(f"\n🚀 Running {script_path} {' '.join(args)}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed to run {script_path}")
        # We don't exit here, to allow partial completion if needed


def subset_embeddings(
    full_embeddings_path: str,
    node_ids_file: str,
    output_path: str,
    entity_type: str
):
    """
    Subset learned embeddings to only include nodes in the graph.
    Imputes missing nodes with mean embedding.
    
    Args:
        full_embeddings_path: Path to full embeddings .pt file
        node_ids_file: Path to parquet with node IDs in graph
        output_path: Path to save subset embeddings
        entity_type: "go" or "reactome" (for logging)
    """
    print(f"\n   📊 Subsetting {entity_type} embeddings to graph nodes...")
    
    # Load full embeddings
    full_emb = torch.load(full_embeddings_path, weights_only=False)
    print(f"      Full embeddings: {len(full_emb):,} nodes")
    
    # Load graph node IDs
    df = pd.read_parquet(node_ids_file)
    graph_ids = set(df['id'].tolist())
    print(f"      Graph nodes: {len(graph_ids):,} nodes")
    
    # Subset embeddings (nodes that exist in both)
    subset_emb = {nid: emb for nid, emb in full_emb.items() if nid in graph_ids}
    print(f"      Found embeddings: {len(subset_emb):,} nodes")
    
    # Find missing nodes
    missing_ids = graph_ids - set(subset_emb.keys())
    
    if missing_ids:
        print(f"      ⚠️  Missing {len(missing_ids):,} nodes ({len(missing_ids)/len(graph_ids)*100:.1f}%)")
        
        # Compute mean embedding for imputation
        all_embeddings = torch.stack(list(full_emb.values()))
        mean_embedding = all_embeddings.mean(dim=0)
        
        # Impute missing nodes
        for nid in missing_ids:
            subset_emb[nid] = mean_embedding
        
        print(f"      ✅ Imputed missing nodes with mean embedding")
    
    print(f"      Final embeddings: {len(subset_emb):,} nodes")
    
    # Save subset
    torch.save(subset_emb, output_path)
    print(f"      ✅ Saved to {output_path}")
    
    return len(subset_emb)


def extract_node_ids_from_mappings(mappings_file: str, output_dir: str):
    """
    Extract node IDs from graph mappings and save as parquet files.
    
    Args:
        mappings_file: Path to temporal_graph_mappings.pt
        output_dir: Directory to save node ID parquets
    
    Returns:
        dict: Paths to saved node ID files {node_type: path}
    """
    print(f"\n📋 Extracting node IDs from mappings: {mappings_file}")
    
    # Load mappings
    mappings = torch.load(mappings_file, weights_only=False)
    
    if 'node_mapping' not in mappings:
        raise ValueError("Mappings file does not contain 'node_mapping'")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    node_files = {}
    
    for node_type, node_map in mappings['node_mapping'].items():
        # Extract node IDs (keys from the mapping)
        node_ids = list(node_map.keys())
        
        # Save as parquet
        df = pd.DataFrame({'id': node_ids})
        output_file = output_path / f"{node_type}.parquet"
        df.to_parquet(output_file, index=False)
        
        node_files[node_type] = str(output_file)
        print(f"   {node_type}: {len(node_ids):,} nodes → {output_file}")
    
    return node_files


def main():
    parser = argparse.ArgumentParser(description="Build all node features from graph mappings")
    parser.add_argument("--mappings-file", required=True, help="Path to temporal_graph_mappings.pt")
    parser.add_argument("--evidence-dir", default="data/evidenceDated_subset/23.06", help="Evidence directory with full node data")
    parser.add_argument("--feature-data-dir", default="data/node_features", help="Raw feature data directory (for targets)")
    parser.add_argument("--output-dir", default="output/features/processed", help="Output directory for features")
    parser.add_argument("--temp-dir", default="output/features/temp_nodes", help="Temporary directory for extracted node IDs")
    args = parser.parse_args()
    
    mappings_file = args.mappings_file
    evidence_dir = args.evidence_dir
    feature_dir = args.feature_data_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir
    
    print(f"\n{'='*60}")
    print("BUILDING NODE FEATURES FROM GRAPH MAPPINGS")
    print(f"{'='*60}")
    print(f"Mappings file: {mappings_file}")
    print(f"Evidence directory: {evidence_dir}")
    print(f"Feature data directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    
    # Extract node IDs from mappings
    node_files = extract_node_ids_from_mappings(mappings_file, temp_dir)
    
    # 1. Target Features (Static + RNA) - ALIGNED TO GRAPH NODES
    print(f"\n{'='*60}")
    print("1. TARGET FEATURES")
    print(f"{'='*60}") 
    
    if 'target' in node_files:
        target_args = [
            "--base-dir", feature_dir,
            "--output-dir", output_dir,
            "--target-ids-file", node_files['target']
        ]
        print(f"✅ Using {node_files['target']}")
        run_script("src/node_features/target_features.py", target_args)
    else:
        print("⚠️  No target nodes in graph")
    
    # 2. Disease Features (Text) - ALIGNED TO GRAPH NODES
    print(f"\n{'='*60}")
    print("2. DISEASE FEATURES")
    print(f"{'='*60}")
    
    if 'disease' in node_files:
        disease_args = [
            "--disease-dir", f"{evidence_dir}/diseases",
            "--output-dir", output_dir,
            "--parquet-glob", "part-*.parquet",
            "--batch-size", "128",
            "--kg-ids-file", node_files['disease']
        ]
        print(f"✅ Using {node_files['disease']}")
        print(f"   Looking up data in: {evidence_dir}/diseases")
        run_script("src/node_features/disease_description.py", disease_args)
    else:
        print("⚠️  No disease nodes in graph")
    
    # 3. GO and Reactome Embeddings (Autoencoder) - TRAIN ON FULL GRAPH
    print(f"\n{'='*60}")
    print("3. GO & REACTOME AUTOENCODER EMBEDDINGS")
    print(f"{'='*60}")
    
    # Determine static edges directory from mappings file path
    mappings_path = Path(mappings_file)
    base_output = mappings_path.parent.parent
    static_edges_dir = base_output / "evidences" / "static_edges"
    
    print(f"   Static edges directory: {static_edges_dir}")
    
    # Train GO embeddings
    go_edges_file = static_edges_dir / "go_is_subtype_of_go_gene_ontology.parquet"
    if go_edges_file.exists():
        print(f"\n   🧬 Training GO autoencoder on full graph...")
        go_full_path = f"{output_dir}/go_embeddings_full.pt"
        go_args = [
            "--edges-file", str(go_edges_file),
            "--output-path", go_full_path,
            "--entity-type", "go",
            "--embedding-dim", "64",
            "--num-epochs", "100",
        ]
        run_script("src/node_features/pathway_go_autoencoder.py", go_args)
        
        # Subset to graph nodes if GO nodes exist in graph
        if 'go' in node_files:
            subset_embeddings(
                full_embeddings_path=go_full_path,
                node_ids_file=node_files['go'],
                output_path=f"{output_dir}/go_embeddings.pt",
                entity_type="go"
            )
        else:
            print(f"   ⚠️  No GO nodes in graph, skipping subset")
    else:
        print(f"   ⚠️  GO edges not found: {go_edges_file}")
    
    # Train Reactome embeddings
    reactome_edges_file = static_edges_dir / "reactome_is_subpathway_of_reactome_reactome.parquet"
    if reactome_edges_file.exists():
        print(f"\n   🧬 Training Reactome autoencoder on full graph...")
        reactome_full_path = f"{output_dir}/reactome_embeddings_full.pt"
        reactome_args = [
            "--edges-file", str(reactome_edges_file),
            "--output-path", reactome_full_path,
            "--entity-type", "reactome",
            "--embedding-dim", "64",
            "--num-epochs", "100",
        ]
        run_script("src/node_features/pathway_go_autoencoder.py", reactome_args)
        
        # Subset to graph nodes if Reactome nodes exist in graph
        if 'reactome' in node_files:
            subset_embeddings(
                full_embeddings_path=reactome_full_path,
                node_ids_file=node_files['reactome'],
                output_path=f"{output_dir}/reactome_embeddings.pt",
                entity_type="reactome"
            )
        else:
            print(f"   ⚠️  No Reactome nodes in graph, skipping subset")
    else:
        print(f"   ⚠️  Reactome edges not found: {reactome_edges_file}")
    
    # 4. Molecule Features (Morgan Fingerprints) - ALIGNED TO GRAPH NODES
    print(f"\n{'='*60}")
    print("4. MOLECULE FEATURES")
    print(f"{'='*60}")
    
    if 'molecule' in node_files:
        molecule_args = [
            "--drug-dir", f"{evidence_dir}/molecule",
            "--output-dir", output_dir,
            "--parquet-glob", "part-*.parquet",
            "--id-col", "id",
            "--smiles-col", "canonicalSmiles",
            "--kg-ids-file", node_files['molecule']
        ]
        print(f"✅ Using {node_files['molecule']}")
        print(f"   Looking up data in: {evidence_dir}/molecule")
        run_script("src/node_features/molecule_structure.py", molecule_args)
    else:
        print("⚠️  No molecule nodes in graph")
    
    print(f"\n{'='*60}")
    print("✅ ALL NODE FEATURES BUILT")
    print(f"{'='*60}")
    print(f"\nFeatures saved to: {output_dir}")
    print(f"Temporary node files: {temp_dir}")

if __name__ == "__main__":
    main()

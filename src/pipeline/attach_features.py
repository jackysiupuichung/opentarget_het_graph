#!/usr/bin/env python3
"""
Attach node features to a HeteroData graph.
Loads features from processed directory and attaches them to graph nodes.
"""

import argparse
import sys
import torch
from pathlib import Path


def load_feature_embeddings(feature_dir: Path, node_type: str):
    """
    Load feature embeddings for a specific node type.
    
    Args:
        feature_dir: Directory containing processed feature files
        node_type: Node type (disease, target, go, reactome, molecule)
    
    Returns:
        dict: {node_id: tensor} mapping, or None if not found
    """
    # Map node types to their feature files
    feature_files = {
        'disease': 'disease_embeddings.pt',
        'target': 'integrated_target_features.pt',
        'go': 'go_embeddings.pt',
        'reactome': 'reactome_embeddings.pt',
        'molecule': 'molecule_morgan_fingerprints.pt'
    }
    
    if node_type not in feature_files:
        print(f"   ⚠️  Unknown node type: {node_type}")
        return None
    
    feature_path = feature_dir / feature_files[node_type]
    
    if not feature_path.exists():
        print(f"   ⚠️  Feature file not found: {feature_path}")
        return None
    
    print(f"   Loading {node_type} features from {feature_path.name}")
    embeddings = torch.load(feature_path, weights_only=False)
    print(f"      Loaded {len(embeddings):,} embeddings")
    
    return embeddings


def attach_features_to_graph(data, feature_dir: Path):
    """
    Attach node features to HeteroData graph.
    
    Args:
        data: HeteroData graph
        feature_dir: Directory with processed feature .pt files
    
    Returns:
        HeteroData: Graph with features attached
    """
    print(f"\n📦 Attaching features from {feature_dir}")
    
    for node_type in data.node_types:
        print(f"\n   Processing {node_type}...")
        
        # Load embeddings
        embeddings = load_feature_embeddings(feature_dir, node_type)
        
        if embeddings is None:
            print(f"      ❌ No features available for {node_type}")
            data[node_type].x = None
            continue
        
        # Get node IDs from graph (assumes node IDs are stored in data[node_type].node_id)
        if not hasattr(data[node_type], 'node_id'):
            print(f"      ⚠️  No node_id attribute found for {node_type}")
            print(f"      Available attributes: {data[node_type].keys()}")
            data[node_type].x = None
            continue
        
        node_ids = data[node_type].node_id
        num_nodes = len(node_ids)
        
        # Build feature matrix
        feature_list = []
        missing_count = 0
        
        for node_id in node_ids:
            if node_id in embeddings:
                feature_list.append(embeddings[node_id])
            else:
                # Use zero vector for missing nodes
                if len(feature_list) > 0:
                    feature_list.append(torch.zeros_like(feature_list[0]))
                else:
                    # If first node is missing, we need to know the dimension
                    # Get dimension from first available embedding
                    sample_emb = next(iter(embeddings.values()))
                    feature_list.append(torch.zeros_like(sample_emb))
                missing_count += 1
        
        if len(feature_list) == 0:
            print(f"      ❌ No features could be loaded")
            data[node_type].x = None
            continue
        
        # Stack into tensor
        feature_matrix = torch.stack(feature_list)
        data[node_type].x = feature_matrix
        
        print(f"      ✅ Attached features: {feature_matrix.shape}")
        if missing_count > 0:
            print(f"      ⚠️  Missing embeddings: {missing_count}/{num_nodes} ({missing_count/num_nodes*100:.1f}%)")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Attach node features to HeteroData graph")
    parser.add_argument("--graph-file", required=True, help="Input .pt graph file")
    parser.add_argument("--output-file", required=True, help="Output .pt graph file with features")
    parser.add_argument("--feature-dir", required=True, help="Directory with processed feature .pt files")
    
    args = parser.parse_args()
    
    # Validate inputs
    graph_path = Path(args.graph_file)
    feature_dir = Path(args.feature_dir)
    output_path = Path(args.output_file)
    
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        sys.exit(1)
    
    if not feature_dir.exists():
        print(f"❌ Feature directory not found: {feature_dir}")
        sys.exit(1)
    
    print(f"\n🔗 Attaching Features to Graph")
    print(f"{'='*60}")
    print(f"Graph file: {graph_path}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output file: {output_path}")
    
    # Load graph
    print(f"\n📊 Loading graph...")
    data = torch.load(graph_path, weights_only=False)
    print(f"   Node types: {data.node_types}")
    print(f"   Edge types: {data.edge_types}")
    print(f"   Total nodes: {data.num_nodes:,}")
    print(f"   Total edges: {data.num_edges:,}")
    
    # Attach features
    data = attach_features_to_graph(data, feature_dir)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    print(f"\n✅ Saved graph with features to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("FEATURE SUMMARY")
    print(f"{'='*60}")
    for node_type in data.node_types:
        if data[node_type].x is not None:
            print(f"   {node_type:15s}: {str(data[node_type].x.shape):20s} ✅")
        else:
            print(f"   {node_type:15s}: {'None':20s} ❌")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

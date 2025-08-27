#!/usr/bin/env python3
import os
import argparse
import glob
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData


def load_nodes(node_dir):
    """
    Load node parquet files into a dict and build id→type lookup.
    Returns:
      - nodes: dict of {node_type: DataFrame}
      - id_to_type: dict mapping node_id → node_type
    """
    nodes = {}
    id_to_type = {}

    for fname in os.listdir(node_dir):
        if fname.endswith(".parquet"):
            node_type = os.path.splitext(fname)[0]
            node_df = pd.read_parquet(os.path.join(node_dir, fname))
            nodes[node_type] = node_df
            print(f"  - {node_type}: {len(node_df)} nodes")

            for nid in node_df["id"].astype(str).tolist():
                id_to_type[nid] = node_type

    return nodes, id_to_type

def load_edges(edge_dir):
    """Load all edge parquet files (including subdirectories) into a single DataFrame."""
    files = glob.glob(os.path.join(edge_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {edge_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def deduplicate_edges(edges, relation_mode="datatype"):
    """
    Deduplicate edges by (src, tgt, relation) keeping the one with max score.
    Preserves source_type and target_type.
    """
    if relation_mode == "datatype":
        edges["rel_key"] = edges["relation"]
    elif relation_mode == "source":
        edges["rel_key"] = edges["sourceId"]
    else:
        raise ValueError("relation_mode must be 'datatype' or 'source'")

    # Sort so highest score is first
    edges = edges.sort_values("score", ascending=False)

    # Deduplicate while keeping source_type & target_type
    keep_cols = ["source", "target", "rel_key", "score", "source_type", "target_type"]
    if "year" in edges.columns:
        keep_cols.append("year")
    if "datasourceId" in edges.columns:
        keep_cols.append("datasourceId")

    edges = edges.drop_duplicates(subset=["source", "target", "rel_key"], keep="first")[keep_cols]

    return edges



def temporal_split(edges, cutoff, test_horizon=5):
    """
    Split into:
      - train: edges with publicationYear <= cutoff
      - test:  edges in (cutoff, cutoff+test_horizon], chembl only
    """
    train_edges = edges[edges["year"] <= cutoff]
    test_edges = edges[
        (edges["year"] > cutoff)
        & (edges["year"] <= cutoff + test_horizon)
        & (edges["datasourceId"] == "chembl")
    ]
    return train_edges, test_edges

def temporal_user_item_split():
    # placeholder for user-item split logic
    pass


def build_heterodata(nodes, train_edges, test_edges):
    """
    Build a PyG HeteroData object from nodes and train/test edges.
    Expects edge DataFrames to include: source, target, relation, source_type, target_type.
    """
    data = HeteroData()

    # === Nodes ===
    id_maps = {}
    for node_type, node_df in nodes.items():
        ids = node_df["id"].astype(str).tolist()
        id_map = {nid: i for i, nid in enumerate(ids)}
        id_maps[node_type] = id_map
        node_df["mapped_id"] = node_df["id"].map(id_map)
        nodes[node_type] = node_df
        data[node_type].num_nodes = len(node_df)

    # === Edges ===
    for split_name, edge_df in [("train", train_edges), ("test", test_edges)]:
        for (src_type, rel_name, dst_type), group in edge_df.groupby(
            ["source_type", "rel_key", "target_type"]
        ):
            # map IDs to integer indices
            src_ids = [id_maps[src_type][s] for s in group["source"].astype(str)]
            dst_ids = [id_maps[dst_type][t] for t in group["target"].astype(str)]
            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

            # add to heterodata
            data[(src_type, rel_name, dst_type)].edge_index = edge_index

            # add score if available
            if "score" in group.columns:
                data[(src_type, rel_name, dst_type)].edge_score = torch.tensor(
                    group["score"].values, dtype=torch.float
                )

            # add split mask
            if split_name == "train":
                mask = torch.ones(edge_index.size(1), dtype=torch.bool)
            else:
                mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

            data[(src_type, rel_name, dst_type)].split_mask = mask

    return data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-dir", required=True, help="Directory with edge parquet files")
    parser.add_argument("--node-dir", required=True, help="Directory with node parquet files")
    parser.add_argument("--cutoff", type=int, default=2015, help="Training cutoff year")
    parser.add_argument("--test-horizon", type=int, default=5, help="Number of years after cutoff for test set")
    parser.add_argument("--relation-mode", choices=["datatype", "source"], default="datatype")
    parser.add_argument("--out", required=True, help="Output torch file path (.pt)")
    args = parser.parse_args()

    print("📂 Loading nodes...")
    nodes, id_to_type = load_nodes(args.node_dir)

    print("📂 Loading edges...")
    edges = load_edges(args.edge_dir)
    print(f"✅ Loaded {len(edges)} edges")

    print("🔗 Annotating edge types from node lookup...")
    edges["source_type"] = edges["source"].astype(str).map(id_to_type)
    edges["target_type"] = edges["target"].astype(str).map(id_to_type)

    missing_src = edges["source_type"].isna().sum()
    missing_tgt = edges["target_type"].isna().sum()
    if missing_src or missing_tgt:
        print(f"⚠️ Missing type for {missing_src} sources, {missing_tgt} targets")

    print("⏳ Splitting into train/test...")
    train_edges, test_edges = temporal_split(edges, cutoff=args.cutoff, test_horizon=args.test_horizon)
    print(f"✅ Train edges (raw): {len(train_edges)}, Test edges (raw): {len(test_edges)}")

    print("🧹 Deduplicating train/test separately...")
    train_edges = deduplicate_edges(train_edges, relation_mode=args.relation_mode)
    test_edges = deduplicate_edges(test_edges, relation_mode=args.relation_mode)
    print(f"✅ Train edges after dedup: {len(train_edges)}, Test edges after dedup: {len(test_edges)}")



if __name__ == "__main__":
    main()

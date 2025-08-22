#!/usr/bin/env python3
import os
import argparse
import duckdb
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def load_edges(edge_dir):
    """Load all edge parquet files into a single DataFrame."""
    con = duckdb.connect()
    query = f"""
        SELECT * FROM parquet_scan('{os.path.join(edge_dir, '*.parquet')}')
    """
    df = con.execute(query).fetch_df()
    con.close()
    return df


def deduplicate_edges(edges, relation_mode="datatype"):
    """
    Deduplicate edges by (src, tgt, relation) keeping the one with max score.
    relation_mode: 'datatype' (relation col) or 'source' (sourceId col)
    """
    if relation_mode == "datatype":
        edges["rel_key"] = edges["relation"]
    elif relation_mode == "source":
        edges["rel_key"] = edges["sourceId"]
    else:
        raise ValueError("relation_mode must be 'datatype' or 'source'")

    # Keep max score per (src, tgt, rel_key)
    edges = (
        edges.sort_values("score", ascending=False)
        .drop_duplicates(subset=["source", "target", "rel_key"], keep="first")
    )
    return edges


def temporal_split(edges, cutoff, test_horizon=5):
    """
    Split into:
      - train: edges with publicationYear <= cutoff
      - test:  edges in (cutoff, cutoff+test_horizon], chembl only
    """
    train_edges = edges[edges["publicationYear"] <= cutoff]
    test_edges = edges[
        (edges["publicationYear"] > cutoff)
        & (edges["publicationYear"] <= cutoff + test_horizon)
        & (edges["sourceId"] == "chembl")
    ]
    return train_edges, test_edges


def build_heterodata(nodes, train_edges, test_edges):
    """
    Build a PyG HeteroData object from nodes and train/test edges.
    """
    data = HeteroData()

    # === Nodes ===
    for node_type, node_df in nodes.items():
        node_ids = pd.Series(node_df["id"].astype("category").cat.codes.values)
        id_map = dict(zip(node_df["id"], node_ids))
        node_df["mapped_id"] = node_df["id"].map(id_map)
        nodes[node_type] = node_df
        data[node_type].num_nodes = len(node_df)

    # === Edges ===
    for split_name, edge_df in [("train", train_edges), ("test", test_edges)]:
        for rel_key, group in edge_df.groupby("rel_key"):
            src_type = "target" if group["source"].iloc[0].startswith("ENSG") else "disease"
            dst_type = "disease" if src_type == "target" else "target"

            src_ids = nodes[src_type].set_index("id").loc[group["source"]]["mapped_id"].values
            dst_ids = nodes[dst_type].set_index("id").loc[group["target"]]["mapped_id"].values

            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

            rel_name = str(rel_key)
            data[(src_type, rel_name, dst_type)].edge_index = edge_index

            # Save scores as edge attributes
            if "score" in group.columns:
                data[(src_type, rel_name, dst_type)].edge_score = torch.tensor(
                    group["score"].values, dtype=torch.float
                )

            # Keep split info
            split_mask = torch.ones(edge_index.size(1), dtype=torch.bool) if split_name == "train" else torch.zeros(edge_index.size(1), dtype=torch.bool)
            data[(src_type, rel_name, dst_type)].set_default_key("split_mask", split_mask)

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

    print("📂 Loading edges...")
    edges = load_edges(args.edge_dir)
    print(f"✅ Loaded {len(edges)} edges")

    print("🧹 Deduplicating edges...")
    edges = deduplicate_edges(edges, relation_mode=args.relation_mode)
    print(f"✅ {len(edges)} edges remain after deduplication")

    print("⏳ Splitting into train/test...")
    train_edges, test_edges = temporal_split(edges, cutoff=args.cutoff)
    print(f"✅ Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")

    print("📂 Loading nodes...")
    nodes = {}
    for fname in os.listdir(args.node_dir):
        if fname.endswith(".parquet"):
            node_type = os.path.splitext(fname)[0]
            nodes[node_type] = pd.read_parquet(os.path.join(args.node_dir, fname))
            print(f"  - {node_type}: {len(nodes[node_type])} nodes")

    print("⚙️ Building HeteroData object...")
    data = build_heterodata(nodes, train_edges, test_edges)

    print(f"💾 Saving graph object → {args.out}")
    torch.save(data, args.out)
    print("🚀 Done!")


if __name__ == "__main__":
    main()

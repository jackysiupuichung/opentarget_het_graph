#!/usr/bin/env python3
"""
Serialize Graph to CSV
======================
Exports the heterogeneous graph (.pt) to two CSV files:
  - output/graph/csv/edges.csv        : full edge list (src_type, src_idx, rel, dst_type, dst_idx)
  - output/graph/csv/node_mappings.csv: node_id string → integer index per node type

Run once (requires torch/PyG environment), then use the CSVs for downstream
analysis without needing PyTorch.

Usage:
    python scripts/serialize_graph_to_csv.py
"""

import sys
import torch
import pandas as pd
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
GRAPH_FILE    = ROOT / "output/graph/hetero_graph_with_features.pt"
MAPPINGS_FILE = ROOT / "output/graph/temporal_graph_mappings.pt"
OUT_DIR       = ROOT / "output/graph/csv"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading graph from {GRAPH_FILE} ...")
    data = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)

    print(f"Loading mappings from {MAPPINGS_FILE} ...")
    mappings = torch.load(MAPPINGS_FILE, weights_only=False)

    # ------------------------------------------------------------------
    # 1. Edge list CSV
    # ------------------------------------------------------------------
    print("Serialising edges ...")
    edge_rows = []
    for src_type, rel, dst_type in data.edge_types:
        ei = data[(src_type, rel, dst_type)].edge_index  # [2, E]
        src_arr = ei[0].numpy()
        dst_arr = ei[1].numpy()
        n = len(src_arr)
        edge_rows.append(pd.DataFrame({
            "src_type": src_type,
            "src_idx":  src_arr,
            "rel":      rel,
            "dst_type": dst_type,
            "dst_idx":  dst_arr,
        }))
        print(f"  {src_type} --[{rel}]--> {dst_type}  ({n:,} edges)")

    edges_df = pd.concat(edge_rows, ignore_index=True)
    edges_path = OUT_DIR / "edges.csv"
    edges_df.to_csv(edges_path, index=False)
    print(f"  Saved {len(edges_df):,} total edges → {edges_path}")

    # ------------------------------------------------------------------
    # 2. Node mapping CSV
    # ------------------------------------------------------------------
    print("Serialising node mappings ...")
    mapping_rows = []
    for node_type, id_to_idx in mappings["node_mapping"].items():
        for node_id, node_idx in id_to_idx.items():
            mapping_rows.append({
                "node_type": node_type,
                "node_id":   node_id,
                "node_idx":  node_idx,
            })
    mappings_df = pd.DataFrame(mapping_rows)
    mappings_path = OUT_DIR / "node_mappings.csv"
    mappings_df.to_csv(mappings_path, index=False)
    print(f"  Saved {len(mappings_df):,} node mappings → {mappings_path}")

    print("\nDone. CSVs written to:", OUT_DIR)


if __name__ == "__main__":
    main()

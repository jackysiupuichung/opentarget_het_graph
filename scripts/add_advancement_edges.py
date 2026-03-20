"""
Add clinical trial advancement edges to the heterogeneous graph.

Loads hetero_graph_with_features.pt and injects a new ('target', 'advancement', 'disease')
edge type from the train/test advancement CSVs, saving as hetero_graph_with_advancement.pt.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main(args):
    graph_path = Path(args.graph)
    mappings_path = Path(args.mappings)
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    output_path = Path(args.output)

    log.info("Loading graph from %s", graph_path)
    data = torch.load(graph_path, weights_only=False)

    log.info("Loading mappings from %s", mappings_path)
    mappings = torch.load(mappings_path, weights_only=False)
    node_mapping = mappings["node_mapping"]

    log.info("Reading advancement CSVs")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    log.info("  train rows: %d", len(train_df))
    log.info("  test rows:  %d", len(test_df))

    df = pd.concat([train_df, test_df], ignore_index=True)
    log.info("  merged rows: %d", len(df))

    # Filter to nodes present in the graph
    target_set = set(node_mapping["target"].keys())
    disease_set = set(node_mapping["disease"].keys())
    mask = df["target_id"].isin(target_set) & df["disease_id"].isin(disease_set)
    n_dropped = (~mask).sum()
    if n_dropped:
        log.warning("Dropping %d rows whose IDs are not in the graph", n_dropped)
        missing_targets = set(df.loc[~mask, "target_id"]) - target_set
        missing_diseases = set(df.loc[~mask, "disease_id"]) - disease_set
        if missing_targets:
            log.warning("  Missing targets (%d): %s", len(missing_targets),
                        list(missing_targets)[:5])
        if missing_diseases:
            log.warning("  Missing diseases (%d): %s", len(missing_diseases),
                        list(missing_diseases)[:5])
    df = df[mask].reset_index(drop=True)
    log.info("Edges to add: %d", len(df))

    # Build tensors
    src = torch.tensor(
        [node_mapping["target"][t] for t in df["target_id"]], dtype=torch.long
    )
    dst = torch.tensor(
        [node_mapping["disease"][d] for d in df["disease_id"]], dtype=torch.long
    )
    edge_time = torch.tensor(df["transition_year"].astype(int).values, dtype=torch.long)
    edge_attr = torch.tensor(
        df["outcome"].astype(float).values, dtype=torch.float
    ).unsqueeze(-1)

    edge_type = ("target", "advancement", "disease")
    data[edge_type].edge_index = torch.stack([src, dst], dim=0)
    data[edge_type].edge_attr = edge_attr
    data[edge_type].edge_time = edge_time

    log.info("Edge type added: %s", edge_type)
    log.info("  edge_index shape: %s", tuple(data[edge_type].edge_index.shape))
    log.info("  edge_attr shape:  %s", tuple(data[edge_type].edge_attr.shape))
    log.info("  edge_time shape:  %s", tuple(data[edge_type].edge_time.shape))
    log.info("  outcome=True:  %d", int(edge_attr.sum().item()))
    log.info("  outcome=False: %d", int((edge_attr == 0).sum().item()))
    log.info("  year range: %d – %d", edge_time.min().item(), edge_time.max().item())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving enriched graph to %s", output_path)
    torch.save(data, output_path)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add advancement edges to the graph.")
    parser.add_argument(
        "--graph",
        default="output/graph/hetero_graph_with_features.pt",
        help="Path to input HeteroData .pt file",
    )
    parser.add_argument(
        "--mappings",
        default="output/graph/temporal_graph_mappings.pt",
        help="Path to node/edge mappings .pt file",
    )
    parser.add_argument(
        "--train-csv",
        default="data/clinical_trial_advancement/23.06/train_dataset.csv",
        help="Path to train advancement CSV",
    )
    parser.add_argument(
        "--test-csv",
        default="data/clinical_trial_advancement/23.06/test_dataset.csv",
        help="Path to test advancement CSV",
    )
    parser.add_argument(
        "--output",
        default="output/graph/hetero_graph_with_advancement.pt",
        help="Path for output enriched graph",
    )
    main(parser.parse_args())

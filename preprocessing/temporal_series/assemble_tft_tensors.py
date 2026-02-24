#!/usr/bin/env python3
"""
Assemble final TFT tensor dictionary from:
  - tft_longitudinal.parquet  (dynamic source-level features)
  - tft_anchors.parquet       (partition + outcome labels)
  - feature .pt files         (static node priors from Step 03/04)

Output: tft_tensors.pt
{
    "static":    Tensor[N, D_static],
    "temporal":  Tensor[N, T, num_sources*2],
    "outcome":   Tensor[N],
    "partition": List[str],
    "td_pairs":  List[Tuple[str, str]],
}
"""
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "temporal_graph" / "pipeline"))
from attach_features import load_feature_embeddings


def load_static_features(
    anchors: pd.DataFrame,
    feature_dir: Path,
    id_cols: list,
) -> torch.Tensor:
    """
    Load static node features (disease + target embeddings) from .pt files
    produced by Step 03 (collecting_node_features_03.sh).

    For each TD pair in anchors, concatenate disease + target embeddings.
    Missing embeddings are filled with zeros.
    """
    disease_col, target_col = id_cols[0], id_cols[1]

    disease_embs = load_feature_embeddings(feature_dir, 'disease')
    target_embs  = load_feature_embeddings(feature_dir, 'target')

    if disease_embs is None or target_embs is None:
        print("   ⚠️ Static feature files not found — static tensor will be zeros.")
        return torch.zeros(len(anchors), 1)

    # Determine embedding dims from first entry
    d_dim = next(iter(disease_embs.values())).shape[0]
    t_dim = next(iter(target_embs.values())).shape[0]

    static_rows = []
    for _, row in tqdm(anchors.iterrows(), total=len(anchors), desc="Building static features"):
        d_emb = disease_embs.get(row[disease_col], torch.zeros(d_dim))
        t_emb = target_embs.get(row[target_col],  torch.zeros(t_dim))
        static_rows.append(torch.cat([d_emb, t_emb]))

    return torch.stack(static_rows)


def build_temporal_tensor(
    longitudinal: pd.DataFrame,
    anchors: pd.DataFrame,
    id_cols: list,
    lookback: int,
) -> torch.Tensor:
    """
    Build temporal tensor of shape [N, T, F] where:
        N = number of TD pairs
        T = lookback window
        F = number of feature columns (_S and _N columns)
    """
    feature_cols = sorted([c for c in longitudinal.columns if any(c.endswith(s) for s in ['_S', '_N', '_P'])])
    T = lookback
    N = len(anchors)
    F = len(feature_cols)

    print(f"   Building temporal tensor [{N}, {T}, {F}]")

    # Index longitudinal by (TD pair, relative_year)
    idx = longitudinal.set_index(id_cols + ['relative_year'])

    tensor = torch.zeros(N, T, F, dtype=torch.float32)
    for i, (_, anchor_row) in enumerate(tqdm(anchors.iterrows(), total=N, desc="Building temporal tensor")):
        pair_key = tuple(anchor_row[c] for c in id_cols)
        for t_idx, rel_year in enumerate(range(-lookback, 0)):
            try:
                vals = idx.loc[pair_key + (rel_year,)][feature_cols].values
                tensor[i, t_idx] = torch.tensor(vals, dtype=torch.float32)
            except KeyError:
                pass  # zero-padded already

    return tensor


def main():
    parser = argparse.ArgumentParser(description="Assemble TFT tensor dictionary")
    parser.add_argument("--tft-dir",     default="output/tft_dataset",
                        help="Directory with tft_longitudinal.parquet and tft_anchors.parquet")
    parser.add_argument("--feature-dir", default="output/features/processed",
                        help="Directory with static .pt feature files (from Step 03/04)")
    parser.add_argument("--output",      default="output/tft_dataset/tft_tensors.pt",
                        help="Output tensor file path")
    parser.add_argument("--lookback",    type=int, default=10, help="Lookback window")
    args = parser.parse_args()

    tft_dir     = Path(args.tft_dir)
    feature_dir = Path(args.feature_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load tabular data
    print("📂 Loading parquet inputs...")
    longitudinal = pd.read_parquet(tft_dir / "tft_longitudinal.parquet")
    anchors      = pd.read_parquet(tft_dir / "tft_anchors.parquet")

    id_cols = [c for c in anchors.columns if c in longitudinal.columns and c not in
               ['anchor_year', 'partition', 'outcome', 'relative_year']]
    print(f"   TD identifier columns detected: {id_cols}")
    print(f"   Pairs: {len(anchors):,} | Time steps: {args.lookback} | Partitions: {anchors['partition'].value_counts().to_dict()}")

    # 1. Static tensor
    print("\n🧬 Building static features...")
    static_tensor = load_static_features(anchors, feature_dir, id_cols)

    # 2. Temporal tensor
    print("\n📈 Building temporal tensor...")
    temporal_tensor = build_temporal_tensor(longitudinal, anchors, id_cols, args.lookback)

    # 3. Labels and metadata
    outcome_tensor  = torch.tensor(anchors['outcome'].values, dtype=torch.float32)
    partitions      = anchors['partition'].tolist()
    td_pairs        = list(zip(anchors[id_cols[0]], anchors[id_cols[1]]))

    # 4. Save
    result = {
        "static":    static_tensor,
        "temporal":  temporal_tensor,
        "outcome":   outcome_tensor,
        "partition": partitions,
        "td_pairs":  td_pairs,
    }

    print(f"\n💾 Saving to {output_path}...")
    torch.save(result, output_path)

    print("\n✅ TFT Tensor Assembly Complete!")
    print(f"   static:   {static_tensor.shape}")
    print(f"   temporal: {temporal_tensor.shape}")
    print(f"   outcome:  {outcome_tensor.shape}")
    print(f"   Output:   {output_path}")


if __name__ == "__main__":
    main()

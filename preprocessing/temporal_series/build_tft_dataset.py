#!/usr/bin/env python3
"""
Step 05: Build TFT longitudinal dataset from pre-parsed edge parquets.

Inherits output from Step 01 (collecting_edges_01.sh):
  - raw edge parquets:  output/evidences/edges/
  - ChEMBL edges:       output/evidences/edges/target_clinical_trial_disease_chembl*.parquet

For each TD pair:
  - Anchor:  first year pair reaches Phase 2 (ChEMBL)
  - Mask:    all features use only evidence_year < anchor_year
  - Window:  T-{lookback} to T-1 (relative to anchor)
  - Outcome: 1 if pair reaches Phase 3+ in (anchor_year, outcome_max_year]

Output: longitudinal parquet keyed by (targetId, diseaseId, relative_year)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
from typing import Tuple

# --- Import shared utilities from Step 01 pipeline ---
_PIPELINE_DIR = str(Path(__file__).resolve().parents[1] / "temporal_graph" / "pipeline")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, _PIPELINE_DIR)

from build_event_list import harmonic_sum, aggregate_scores, add_datatype_info


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_edges(edges_dir: str, sample_ratio: float = None) -> pd.DataFrame:
    """Load all pre-parsed edge parquets from the Step 01 output directory."""
    parquet_files = glob(os.path.join(edges_dir, "*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {edges_dir}")

    print(f"📂 Loading {len(parquet_files)} edge parquet files from {edges_dir}")
    dfs = []
    for pf in tqdm(parquet_files, desc="Loading edges"):
        try:
            df = pd.read_parquet(pf)
            if sample_ratio:
                df = df.sample(frac=sample_ratio, random_state=42)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ Error reading {Path(pf).name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"   Loaded {len(combined):,} total edge records")
    return combined


# ──────────────────────────────────────────────
# Score → Clinical Phase Thresholds
# Open Targets ChEMBL score mapping:
#   Phase 1: score >= 0.1
#   Phase 2: score >= 0.2  ← anchor (T=0)
#   Phase 3: score >= 0.7  ← positive outcome
#   Phase 4: score == 1.0  ← positive outcome
# ──────────────────────────────────────────────
PHASE_THRESHOLDS  = {1: 0.1, 2: 0.2, 3: 0.7, 4: 1.0}
ANCHOR_SCORE_MIN  = PHASE_THRESHOLDS[2]  # 0.2 (Phase 2)
OUTCOME_SCORE_MIN = PHASE_THRESHOLDS[3]  # 0.7 (Phase 3)

def get_phase(score: float) -> int:
    """Map score to clinical phase using thresholds."""
    if score >= 1.0: return 4
    if score >= 0.7: return 3
    if score >= 0.2: return 2
    if score >= 0.1: return 1
    return 0

# ──────────────────────────────────────────────
# Anchor Table
# ──────────────────────────────────────────────

def build_anchor_table(
    edges: pd.DataFrame,
    train_max: int,
    val_max: int,
    test_max: int,
    outcome_max: int,
):
    """
    Identify T=0 (first Phase 2 entry) for each TD pair from ChEMBL.
    Phase 2 proxy: score >= ANCHOR_SCORE_MIN  (0.2)
    Phase 3 proxy: score >= OUTCOME_SCORE_MIN (0.7)

    Returns:
        (anchors DataFrame, id_cols list)
    """
    year_col  = 'year'  if 'year'  in edges.columns else None
    score_col = 'score' if 'score' in edges.columns else \
                'edge_weight' if 'edge_weight' in edges.columns else None

    if not year_col:
        raise ValueError("Edge parquets must have a 'year' column. Run Step 01 first.")
    if not score_col:
        raise ValueError("Edge parquets must have a 'score' or 'edge_weight' column.")

    # Filter to ChEMBL edges. Prefer `datasourceId` (used in KG pipeline outputs),
    # then `datasource`. Don't fallback to `sourceId` since that is a node id.
    chembl_mask = (
        edges['datasourceId'].astype(str).str.contains('chembl', case=False, na=False)
        if 'datasourceId' in edges.columns else
        edges['datasource'].astype(str).str.contains('chembl', case=False, na=False)
        if 'datasource' in edges.columns else
        pd.Series([False] * len(edges))
    )

    chembl = edges[chembl_mask & edges[year_col].notna()].copy()
    chembl[year_col]  = chembl[year_col].astype(int)
    chembl[score_col] = pd.to_numeric(chembl[score_col], errors='coerce')

    # Detect TD identifier columns
    id_cols = [c for c in ['sourceId', 'targetId'] if c in chembl.columns]
    if not id_cols:
        raise ValueError(f"Could not detect TD identifier columns. Available: {chembl.columns.tolist()}")

    # T=0: first year with score >= 0.2 (Phase 2 threshold)
    chembl['phase'] = chembl[score_col].apply(get_phase)
    
    phase2_plus = chembl[chembl['phase'] >= 2]
    print(f"   ChEMBL records with Phase 2+ (score ≥ {ANCHOR_SCORE_MIN}): {len(phase2_plus):,}")

    anchors = (
        phase2_plus.groupby(id_cols)[year_col]
        .min().reset_index()
        .rename(columns={year_col: 'anchor_year'})
    )
    
    # FILTER: Exclude pairs that reached Phase 3+ on or before their anchor year
    print("   Filtering out pairs already in Phase 3+ at or before anchor year...")
    pre_success = chembl[chembl['phase'] >= 3].groupby(id_cols)[year_col].min().reset_index()
    pre_success = pre_success.rename(columns={year_col: 'first_phase3_year'})
    
    anchors = anchors.merge(pre_success, on=id_cols, how='left')
    # If first_phase3_year <= anchor_year, it's not a "pure" advancement prediction
    anchors = anchors[~(anchors['first_phase3_year'] <= anchors['anchor_year'])].copy()
    print(f"   Pairs remaining after Phase 3+ pre-filter: {len(anchors):,}")

    # Partition tagging
    print(f"🗓  Partitions: Train ≤ {train_max}, Val [{train_max+1}–{val_max}], Test [{val_max+1}–{test_max}]")
    anchors['partition'] = 'excluded'
    anchors.loc[(anchors['anchor_year'] >= 1990) & (anchors['anchor_year'] <= train_max), 'partition'] = 'train'
    anchors.loc[(anchors['anchor_year'] > train_max) & (anchors['anchor_year'] <= val_max),  'partition'] = 'val'
    anchors.loc[(anchors['anchor_year'] > val_max)   & (anchors['anchor_year'] <= test_max), 'partition'] = 'test'

    anchors = anchors[anchors['partition'] != 'excluded'].copy()
    print(f"   Partition counts:\n{anchors['partition'].value_counts()}")

    # Outcome: reach Phase 3+ in (anchor_year, outcome_max]
    # Join chembl with anchors to check pair-specific timing
    future = chembl[chembl['phase'] >= 3].merge(anchors[id_cols + ['anchor_year']], on=id_cols)
    future = future[(future[year_col] > future['anchor_year']) & (future[year_col] <= outcome_max)]
    
    positive_pairs = set(map(tuple, future[id_cols].drop_duplicates().values))
    anchors['outcome'] = anchors.apply(
        lambda row: int(tuple(row[c] for c in id_cols) in positive_pairs), axis=1
    )

    pos_rate = anchors['outcome'].mean() * 100
    print(f"   Overall positive rate: {pos_rate:.1f}%")

    return anchors, id_cols


# ──────────────────────────────────────────────
# Dynamic Feature Extraction
# ──────────────────────────────────────────────

def build_dynamic_features(
    edges: pd.DataFrame,
    anchors: pd.DataFrame,
    id_cols: list,
    lookback: int,
) -> pd.DataFrame:
    """
    Build source-level harmonic association scores and novelty scores per TD pair per relative year.

    All features use ONLY evidence from evidence_year < anchor_year (pair-specific mask).

    Returns long-format DataFrame with columns:
      [*id_cols, relative_year, {source}_S, {source}_N, ...]
    """
    print("📈 Building dynamic source-level sequences...")

    year_col = 'year'
    source_col = 'datasource' if 'datasource' in edges.columns else \
                 'datasourceId' if 'datasourceId' in edges.columns else None
    score_col  = 'score' if 'score' in edges.columns else \
                 'edge_weight' if 'edge_weight' in edges.columns else None

    if not source_col or not score_col:
        print(f"   ⚠️ Could not find source ({source_col}) or score ({score_col}) columns; skipping dynamic features.")
        return pd.DataFrame()

    edges = edges[edges[year_col].notna()].copy()
    edges[year_col] = edges[year_col].astype(int)

    # Join anchor_year onto evidence (vectorized merge)
    merged = edges.merge(anchors[id_cols + ['anchor_year']], on=id_cols, how='inner')

    # Pair-specific mask: use only pre-anchor evidence
    merged = merged[merged[year_col] < merged['anchor_year']].copy()
    merged['relative_year'] = merged[year_col] - merged['anchor_year']
    merged = merged[merged['relative_year'] >= -lookback]

    if merged.empty:
        print("   ⚠️ No historical evidence found within lookback window.")
        return pd.DataFrame()

    print(f"   {len(merged):,} historical evidence records within lookback window")

    # Vectorized agg per (TD pair, source, relative_year)
    group_cols = id_cols + [source_col, 'relative_year']
    print("   Computing composite dynamic features (S, Phase)...")
    
    # Map score to phase for features
    merged['phase'] = merged[score_col].apply(get_phase)
    
    agg = merged.groupby(group_cols, as_index=False).agg(
        S=(score_col, lambda x: harmonic_sum(x.values)),
        P=('phase', 'max')  # Max phase reached by that source in that year
    )

    # Novelty score N = S_t - S_{t-1}
    agg = agg.sort_values(id_cols + [source_col, 'relative_year'])
    agg['S_prev'] = agg.groupby(id_cols + [source_col])['S'].shift(1).fillna(0)
    agg['N'] = (agg['S'] - agg['S_prev']).clip(lower=0)

    # Pivot: one row per (TD pair, relative_year), columns per source
    print("   Pivoting to wide format (source columns)...")
    s_wide = agg.pivot_table(index=id_cols + ['relative_year'], columns=source_col, values='S', fill_value=0)
    n_wide = agg.pivot_table(index=id_cols + ['relative_year'], columns=source_col, values='N', fill_value=0)
    p_wide = agg.pivot_table(index=id_cols + ['relative_year'], columns=source_col, values='P', fill_value=0)

    s_wide.columns = [f"{c}_S" for c in s_wide.columns]
    n_wide.columns = [f"{c}_N" for c in n_wide.columns]
    p_wide.columns = [f"{c}_P" for c in p_wide.columns]

    wide = s_wide.join(n_wide).join(p_wide).reset_index()
    return wide


# ──────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────

def build_tft_dataset(
    raw_edges_dir: str,
    output_dir: str,
    train_max: int = 2014,
    val_max:   int = 2015,
    test_max:  int = 2022,
    outcome_max: int = 2024,
    lookback:  int = 10,
    sample_ratio: float = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-parsed edge parquets from Step 01
    edges = load_edges(raw_edges_dir, sample_ratio=sample_ratio)

    # 2. Build anchor table (returns anchors + id_cols detected from ChEMBL edges)
    print("\n⚓ Building anchor table (T=0 = first Phase 2 entry, score ≥ 0.2)...")
    anchors, id_cols = build_anchor_table(edges, train_max, val_max, test_max, outcome_max)

    print(f"   TD identifier columns: {id_cols}")

    # 3. Build dynamic source-level sequences
    dynamic = build_dynamic_features(edges, anchors, id_cols, lookback)

    # 4. Fill time-series grid (ensure all relative years are present for each TD pair)
    print("🧩 Filling time-series grid...")
    grid = anchors[id_cols + ['anchor_year', 'partition', 'outcome']].copy()
    year_range = pd.DataFrame({'relative_year': range(-lookback, 0)})
    grid = grid.assign(_key=1).merge(year_range.assign(_key=1), on='_key').drop('_key', axis=1)

    if not dynamic.empty:
        final = grid.merge(dynamic, on=id_cols + ['relative_year'], how='left')
    else:
        final = grid.copy()

    # Fill NAs
    feature_cols = [c for c in final.columns if any(c.endswith(suffix) for suffix in ['_S', '_N', '_P'])]
    final[feature_cols] = final[feature_cols].fillna(0)

    # 5. Summary
    print(f"\n📊 DATASET SUMMARY")
    print("="*60)
    pairs = final.drop_duplicates(id_cols)
    summary = pairs.groupby(['partition', 'outcome']).size().unstack(fill_value=0)
    if 1 in summary.columns and 0 in summary.columns:
        summary['pos_rate (%)'] = (summary[1] / (summary[0] + summary[1]) * 100).round(2)
    print("\nPair Outcomes per Partition:")
    print(summary)
    print(f"\nTime steps per pair: {lookback}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:6]}{'...' if len(feature_cols) > 6 else ''}")
    print("="*60)

    # 6. Save
    out_file = output_path / "tft_longitudinal.parquet"
    print(f"\n💾 Saving to {out_file}...")
    final.to_parquet(out_file, index=False)

    anchors_file = output_path / "tft_anchors.parquet"
    anchors.to_parquet(anchors_file, index=False)
    print(f"💾 Anchor table saved to {anchors_file}")
    print("✅ Done!")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 05: Build TFT longitudinal dataset from Step 01 edge parquets"
    )
    parser.add_argument("--raw-edges-dir", default="output/evidences/edges",
                        help="Directory with pre-parsed edge parquets (output of kg_pipeline / build_event_list)")
    parser.add_argument("--output-dir",    default="output/tft_dataset",
                        help="Output directory for TFT dataset parquets")
    parser.add_argument("--train-max",  type=int, default=2014, help="Max anchor year for train set")
    parser.add_argument("--val-max",    type=int, default=2015, help="Max anchor year for val set")
    parser.add_argument("--test-max",   type=int, default=2022, help="Max anchor year for test set")
    parser.add_argument("--outcome-max", type=int, default=2024, help="Max year for outcome window")
    parser.add_argument("--lookback",   type=int, default=10,   help="Lookback window in years (T-N to T-1)")
    parser.add_argument("--sample-ratio", type=float, default=None,
                        help="Sample fraction of edges (e.g., 0.01 for 1%% for testing)")
    args = parser.parse_args()

    build_tft_dataset(
        raw_edges_dir = args.raw_edges_dir,
        output_dir    = args.output_dir,
        train_max     = args.train_max,
        val_max       = args.val_max,
        test_max      = args.test_max,
        outcome_max   = args.outcome_max,
        lookback      = args.lookback,
        sample_ratio  = args.sample_ratio,
    )


if __name__ == "__main__":
    main()

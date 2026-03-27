#!/usr/bin/env python3
"""
Build temporal event graph from progression edges.

Outputs single event-based graph with edge_time and edge_weight attributes.
Replaces per-year snapshot approach.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def harmonic_sum(scores, max_harmonic=1.644):
    """Compute harmonic sum of top-50 scores (Open Targets standard)."""
    if len(scores) == 0:
        return 0.0
    s = np.sort(scores)[::-1][:50]
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / max_harmonic


def max_score(scores):
    """Return the maximum score."""
    if len(scores) == 0:
        return 0.0
    return np.max(scores)


# Novelty decay parameters (Open Targets paper values)
NOVELTY_SCALE = 2.0   # logistic steepness k
NOVELTY_SHIFT = 3.0   # sigmoid midpoint m
NOVELTY_WINDOW = 5    # years after a peak to apply decay


def compute_novelty_series(year_scores: np.ndarray,
                            scale: float = NOVELTY_SCALE,
                            shift: float = NOVELTY_SHIFT,
                            window: int = NOVELTY_WINDOW) -> np.ndarray:
    """
    Given a score time series for one (sourceId, targetId, datasourceId) triplet,
    return a novelty time series of the same length.

    Novelty at year t = max over all peak years p <= t within window of:
        peak_p / (1 + exp(scale * ((t - p) - shift)))
    where peak_p = score_p - score_{p-1} > 0.

    Args:
        year_scores: 1-D array of cumulative scores, one per year in `years`
        years:       1-D int array of corresponding years (sorted ascending)
        scale:       logistic steepness k
        shift:       sigmoid midpoint m
        window:      number of years after peak year to apply decay

    Returns:
        novelty: 1-D float array, same length as year_scores
    """
    n = len(year_scores)
    novelty = np.zeros(n, dtype=float)

    for p in range(n):
        # Score delta vs previous year (first year compared against 0)
        prev_score = year_scores[p - 1] if p > 0 else 0.0
        peak = year_scores[p] - prev_score
        if peak <= 0:
            continue
        # Apply logistic decay over the window
        for w in range(window + 1):
            t = p + w
            if t >= n:
                break
            nov = peak / (1.0 + np.exp(scale * (w - shift)))
            if nov > novelty[t]:
                novelty[t] = nov

    return novelty


def compute_novelty_per_datasource(dynamic_edges: pd.DataFrame,
                                    start_year: int, end_year: int,
                                    aggregation_method: str = 'harmonic_sum') -> pd.DataFrame:
    """
    Compute per-datasource novelty scores for every (sourceId, targetId, datasourceId)
    triplet across the full year range.

    Uses the same cumulative score series as the main loop so novelty peaks align
    with actual score changes.

    Returns:
        DataFrame with columns: sourceId, targetId, source_type, target_type,
                                 relation, datasourceId, year, novelty
    """
    years = np.arange(start_year, end_year + 1)
    records = []

    id_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 'relation', 'datasourceId']
    groups = dynamic_edges.groupby(id_cols)

    for key, grp in tqdm(groups, desc="Computing novelty"):
        # Build cumulative score series for this triplet
        year_scores = np.zeros(len(years), dtype=float)
        for i, yr in enumerate(years):
            cumulative = grp[grp['year'] <= yr]['score']
            if len(cumulative):
                year_scores[i] = aggregate_scores(cumulative.values, method=aggregation_method)

        novelty_series = compute_novelty_series(year_scores)

        src_id, tgt_id, src_type, tgt_type, relation, datasource = key
        for i, yr in enumerate(years):
            if novelty_series[i] > 0:
                records.append({
                    'sourceId':    src_id,
                    'targetId':    tgt_id,
                    'source_type': src_type,
                    'target_type': tgt_type,
                    'relation':    relation,
                    'datasourceId': datasource,
                    'year':        yr,
                    'novelty':     novelty_series[i],
                })

    if not records:
        return pd.DataFrame(columns=id_cols + ['year', 'novelty'])
    return pd.DataFrame(records)


def aggregate_scores(scores, method='harmonic_sum'):
    """Aggregate scores using specified method.
    
    Args:
        scores: Array of scores
        method: 'harmonic_sum' or 'max'
    
    Returns:
        Aggregated score
    """
    if method == 'harmonic_sum':
        return harmonic_sum(scores)
    elif method == 'max':
        return max_score(scores)
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Use 'harmonic_sum' or 'max'")


def load_datatype_mapping(config_path):
    """Load datasource->datatype mapping with weights.
    
    Returns:
        dict: {datasourceId: {'datatype': datatype_id, 'weight': weight}}
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mapping = {}
    
    for datatype_id, datatype_info in config['datatypes'].items():
        for datasource_id, weight in datatype_info['datasources'].items():
            mapping[datasource_id] = {
                'datatype': datatype_id,
                'weight': weight,
                'label': datatype_info['label']
            }
    
    return mapping


def add_datatype_info(edges, datatype_mapping):
    """Add datatypeId and weight columns based on datasourceId."""
    edges = edges.copy()
    
    # Map datasourceId -> datatype and weight
    edges['datatypeId'] = edges['datasourceId'].map(
        lambda x: datatype_mapping.get(x, {}).get('datatype', x)
    )
    edges['datasource_weight'] = edges['datasourceId'].map(
        lambda x: datatype_mapping.get(x, {}).get('weight', 1.0)
    )
    
    # Warn about unmapped datasources
    unmapped = edges[~edges['datasourceId'].isin(datatype_mapping)]['datasourceId'].unique()
    if len(unmapped) > 0:
        print(f"⚠️  Unmapped datasources (weight=1.0): {unmapped.tolist()}")
    
    return edges


def load_all_edges(directory, sample_ratio=None):
    """Load all parquet files from directory.
    
    Args:
        directory: Directory containing parquet files
        sample_ratio: Optional float (0.0-1.0) to sample each file (e.g., 0.01 for 1:100)
    """
    dfs = []
    parquet_files = glob(os.path.join(directory, "*.parquet"))
    
    for pq in tqdm(parquet_files, desc="Loading edges"):
        try:
            df = pd.read_parquet(pq)
            if not df.empty:
                # Sample if ratio specified
                if sample_ratio is not None and 0 < sample_ratio < 1.0:
                    df = df.sample(frac=sample_ratio, random_state=42)
                dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {pq}: {e}")
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()





def build_event_list(
    input_dir: str,
    config_path: str,
    output_file: str,
    aggregation_method: str = 'harmonic_sum',
    sample_ratio: float = None,
    datatype_mapping_file: str = None,
):
    """
    Build temporal event graph from raw edges using cumulative aggregation.

    Always outputs edge_weight (harmonic sum score) and edge_novelty as separate columns.

    Args:
        input_dir: Directory with raw edges
        config_path: Path to progression config
        output_file: Output parquet file
        aggregation_method: 'harmonic_sum' or 'max' (default: 'harmonic_sum')
        sample_ratio: Optional float (0.0-1.0) to sample edges (e.g., 0.01 for 1:100 test)
        datatype_mapping_file: Optional path to datatype mapping YAML
    """
    print("\n" + "="*80)
    sample_info = f" [SAMPLE: {sample_ratio*100:.1f}%]" if sample_ratio else ""
    mode_str = "DATATYPE-level" if datatype_mapping_file else "datasource-level"
    print(f"BUILDING TEMPORAL EVENT GRAPH ({mode_str}, aggregation: {aggregation_method}){sample_info}")
    print("="*80)
    
    # Load datatype mapping
    datatype_mapping = None
    if datatype_mapping_file:
        print(f"\n📊 Loading datatype mapping from {datatype_mapping_file}...")
        datatype_mapping = load_datatype_mapping(datatype_mapping_file)
        print(f"✅ Loaded {len(datatype_mapping)} datasource->datatype mappings")
        print(f"   Datatypes: {set(d['datatype'] for d in datatype_mapping.values())}")
    
    # Load config
    print(f"\n📄 Loading config from {config_path}...")
    config = load_config(config_path)
    
    # Get time range from config
    if 'time_range' not in config:
        print("❌ No 'time_range' found in config!")
        return
    
    start_year = config['time_range']['first_year']
    end_year = config['time_range']['last_year']
    print(f"✅ Time range from config: {start_year} - {end_year}")
    
    # Load raw edges
    print(f"\n📂 Loading raw edges from {input_dir}...")
    if sample_ratio:
        print(f"   📊 Sampling {sample_ratio*100:.1f}% of each file for testing")
    edges = load_all_edges(input_dir, sample_ratio=sample_ratio)
    
    if edges.empty:
        print("❌ No edges found!")
        return
    
    print(f"✅ Loaded {len(edges):,} total edges")
    
    # Filter to dynamic edges only
    if 'year' not in edges.columns:
        print("❌ No 'year' column found!")
        return
    
    dynamic_edges = edges[edges['year'].notna()].copy()
    print(f"📊 Dynamic edges: {len(dynamic_edges):,}")
    
    # Add datatype info if using datatype aggregation
    if datatype_mapping:
        print(f"\n🏷️  Adding datatype information...")
        dynamic_edges = add_datatype_info(dynamic_edges, datatype_mapping)
        print(f"✅ Added datatypeId and datasource_weight columns")
    
    # Build cumulative temporal events
    print(f"\n🔢 Building cumulative temporal events...")
    print(f"   For each year {start_year}-{end_year}: include all evidences up to that year")
    
    # Group columns (without year)
    if datatype_mapping:
        # Datatype-level: group by datatypeId
        group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                      'relation', 'datatypeId']
        datasource_group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                                 'relation', 'datatypeId', 'datasourceId']
    else:
        # Datasource-level: group by datasourceId
        group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                      'relation', 'datasourceId']
        datasource_group_cols = group_cols
    
    # Store all year-score combinations
    all_events = []
    
    # For each year, calculate cumulative harmonic sum
    for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
        # Get all edges up to and including this year (CUMULATIVE)
        cumulative_edges = dynamic_edges[dynamic_edges['year'] <= year].copy()
        
        if cumulative_edges.empty:
            continue
            
        # Split into Clinical (MAX) and Others (Harmonic Sum)
        mask_clinical = cumulative_edges['relation'].str.contains('clinical_trial', case=False)
        clinical_edges = cumulative_edges[mask_clinical]
        other_edges = cumulative_edges[~mask_clinical]
        
        dfs_to_concat = []
        
        # 1. Clinical Trials -> MAX (always at datasource level, no datatype aggregation)
        if not clinical_edges.empty:
            clinical_agg = clinical_edges.groupby(
                ['sourceId', 'targetId', 'source_type', 'target_type', 'relation', 'datasourceId'],
                as_index=False
            )['score'].max()
            # Add datatypeId if needed
            if datatype_mapping:
                clinical_agg['datatypeId'] = clinical_agg['datasourceId']
            dfs_to_concat.append(clinical_agg)
            
        # 2. Others -> Process differently based on mode
        if not other_edges.empty:
            if datatype_mapping:
                # DATATYPE MODE:
                # Step 1: Harmonic sum per datasource (within each datatype group)
                datasource_scores = other_edges.groupby(datasource_group_cols, as_index=False).agg({
                    'score': lambda x: aggregate_scores(x.values, method=aggregation_method),
                    'datasource_weight': 'first'  # Keep weight
                })
                
                # Step 2: Weight datasource scores
                datasource_scores['weighted_score'] = (
                    datasource_scores['score'] * datasource_scores['datasource_weight']
                )
                
                # Step 3: Harmonic sum of weighted datasource scores -> datatype score
                datatype_agg = datasource_scores.groupby(group_cols, as_index=False).agg({
                    'weighted_score': lambda x: aggregate_scores(x.values, method=aggregation_method)
                }).rename(columns={'weighted_score': 'score'})
                
                dfs_to_concat.append(datatype_agg)
            else:
                # DATASOURCE MODE: Simple harmonic sum per datasource
                other_agg = other_edges.groupby(group_cols, as_index=False).agg({
                    'score': lambda x: aggregate_scores(x.values, method=aggregation_method)
                })
                dfs_to_concat.append(other_agg)
            
        if not dfs_to_concat:
            continue
            
        year_scores = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Add year column
        year_scores['year'] = year
        
        all_events.append(year_scores)
    
    if not all_events:
        print("❌ No events generated!")
        return
    
    # Combine all years
    events = pd.concat(all_events, ignore_index=True)
    print(f"✅ Generated {len(events):,} year-score combinations")
    
    # Keep only score-change events
    # For each combination, keep first year and years where score changes
    print(f"\n🗜️ Filtering to score-change events only...")
    
    events = events.sort_values(group_cols + ['year'])
    
    # Vectorized filtering: Keep if it's the first occurrence of a group OR if score changed
    # 1. Compare Current row with Previous row for the group columns
    # We use .shift(1) to get the value of the previous row
    group_changed = (events[group_cols] != events[group_cols].shift(1)).any(axis=1)
    
    # 2. Compare Current score with Previous score
    score_changed = (events['score'] != events['score'].shift(1))
    
    # Keep row if it's a new group OR if the score changed within the same group
    keep_mask = group_changed | score_changed
    
    events = events[keep_mask]
    print(f"✅ {len(events):,} events after filtering (score changes only)")

    # ── Novelty computation ────────────────────────────────────────────────────
    print(f"\n🔬 Computing per-datasource novelty...")
    # Always computed at datasource level then aggregated up
    novelty_ds = compute_novelty_per_datasource(
        dynamic_edges, start_year, end_year, aggregation_method=aggregation_method
    )
    print(f"   Novelty records: {len(novelty_ds):,}")

    if datatype_mapping:
        # Aggregate datasource novelties → datatype novelty (same two-level scheme as scores)
        novelty_ds = add_datatype_info(novelty_ds, datatype_mapping)
        novelty_ds['weighted_novelty'] = novelty_ds['novelty'] * novelty_ds['datasource_weight']
        novelty_agg = novelty_ds.groupby(group_cols + ['year'], as_index=False).agg(
            novelty=('weighted_novelty', lambda x: harmonic_sum(x.values))
        )
    else:
        novelty_agg = novelty_ds[group_cols + ['year', 'novelty']].copy()

    # Merge novelty into events (left join — keep all score events, fill missing novelty with 0)
    events = events.merge(novelty_agg, on=group_cols + ['year'], how='left')
    events['novelty'] = events['novelty'].fillna(0.0)
    print(f"✅ Novelty merged into events")

    # ── Rename columns ─────────────────────────────────────────────────────────
    events = events.rename(columns={
        'year': 'edge_time',
        'score': 'edge_weight',
        'novelty': 'edge_novelty',
    })
    
    # Save
    print(f"\n💾 Saving event graph...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_file, index=False)
    
    print(f"✅ Saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EVENT GRAPH SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTime range: {int(events['edge_time'].min())} - {int(events['edge_time'].max())}")
    print(f"Total events: {len(events):,}")
    print(f"Unique node pairs: {events[['sourceId', 'targetId']].drop_duplicates().shape[0]:,}")
    
    print(f"\n📈 Events per year:")
    year_counts = events.groupby('edge_time').size()
    for year, count in sorted(year_counts.items()):
        print(f"   {int(year)}: {count:,} events")
    
    print(f"\n{'='*80}")
    print(f"✅ EVENT GRAPH BUILD COMPLETE!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal event graph"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with raw edge parquet files"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to progression config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/progression/events.parquet",
        help="Output parquet file"
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="harmonic_sum",
        choices=["harmonic_sum", "max"],
        help="Score aggregation method: 'harmonic_sum' (default) or 'max'"
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Sample ratio for testing (e.g., 0.01 for 1:100 sample). Default: None (use all data)"
    )
    parser.add_argument(
        "--datatype-mapping",
        type=str,
        default=None,
        help="Path to datatype mapping YAML (enables datatype-level aggregation). Default: None (datasource-level)"
    )
    args = parser.parse_args()

    build_event_list(
        input_dir=args.input_dir,
        config_path=args.config,
        aggregation_method=args.aggregation_method,
        sample_ratio=args.sample_ratio,
        datatype_mapping_file=args.datatype_mapping,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

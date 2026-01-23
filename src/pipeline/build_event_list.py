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


def apply_cutoffs(edges, config):
    """Apply datasource-specific cutoffs using relation::datasource format."""
    if 'datasources' not in config:
        return edges
    
    # Create composite key: relation::datasourceId
    edges = edges.copy()
    edges['relation_datasource'] = edges['relation'] + '::' + edges['datasourceId']
    
    filtered = []
    
    for relation_datasource, params in config['datasources'].items():
        ds_edges = edges[edges['relation_datasource'] == relation_datasource].copy()
        
        if ds_edges.empty:
            continue
        
        if 'cutoff' in params and 'score' in ds_edges.columns:
            cutoff = params['cutoff']
            ds_edges = ds_edges[ds_edges['score'] >= cutoff]
            print(f"   {relation_datasource}: {len(ds_edges):,} edges (cutoff >= {cutoff})")
        else:
            print(f"   {relation_datasource}: {len(ds_edges):,} edges")
        
        filtered.append(ds_edges)
    
    # Include unconfigured datasources
    configured = set(config['datasources'].keys())
    unconfigured = edges[~edges['relation_datasource'].isin(configured)]
    if not unconfigured.empty:
        print(f"   Other relation::datasource combinations: {len(unconfigured):,} edges")
        filtered.append(unconfigured)
    
    result = pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()
    
    # Drop the temporary column
    if not result.empty and 'relation_datasource' in result.columns:
        result = result.drop(columns=['relation_datasource'])
    
    return result


def build_event_list(
    input_dir: str,
    config_path: str,
    output_file: str,
    aggregation_method: str = 'harmonic_sum',
    sample_ratio: float = None,
):
    """
    Build temporal event graph from raw edges using cumulative aggregation.
    
    Creates single event list with edge_time and edge_weight.
    
    Args:
        input_dir: Directory with raw edges
        config_path: Path to progression config
        output_file: Output parquet file
        aggregation_method: 'harmonic_sum' or 'max' (default: 'harmonic_sum')
        sample_ratio: Optional float (0.0-1.0) to sample edges (e.g., 0.01 for 1:100 test)
    """
    print("\n" + "="*80)
    sample_info = f" [SAMPLE: {sample_ratio*100:.1f}%]" if sample_ratio else ""
    print(f"BUILDING TEMPORAL EVENT GRAPH (aggregation: {aggregation_method}){sample_info}")
    print("="*80)
    
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
    
    # Apply cutoffs
    print(f"\n✂️ Applying datasource cutoffs...")
    dynamic_edges = apply_cutoffs(dynamic_edges, config)
    print(f"✅ {len(dynamic_edges):,} edges after cutoffs")
    
    # Build cumulative temporal events
    print(f"\n🔢 Building cumulative temporal events...")
    print(f"   For each year {start_year}-{end_year}: include all evidences up to that year")
    
    # Group columns (without year)
    group_cols = ['sourceId', 'targetId', 'source_type', 'target_type', 
                  'relation', 'datasourceId']
    
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
        
        # 1. Clinical Trials -> MAX
        if not clinical_edges.empty:
            clinical_agg = clinical_edges.groupby(group_cols, as_index=False)['score'].max()
            dfs_to_concat.append(clinical_agg)
            
        # 2. Others -> Harmonic Sum (or whatever aggregation_method is set to)
        if not other_edges.empty:
            # Vectorized harmonic sum is hard, stick to lambda for now or optimize later
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
    
    # Rename for clarity
    events = events.rename(columns={
        'year': 'edge_time',
        'score': 'edge_weight'
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
    
    args = parser.parse_args()
    
    build_event_list(
        input_dir=args.input_dir,
        config_path=args.config,
        aggregation_method=args.aggregation_method,
        sample_ratio=args.sample_ratio,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()

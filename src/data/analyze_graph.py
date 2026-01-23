#!/usr/bin/env python3
"""
Graph Analysis Script
=====================
Calculates and plots statistics for both Event-Based and Time-Agnostic views of the graph.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch_geometric.data import HeteroData
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.temporal_loader import load_event_graph, to_time_agnostic

def setup_plotting():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 12

def plot_counts(counts_dict, title, ylabel, filename, color='skyblue'):
    plt.figure()
    keys = list(counts_dict.keys())
    values = list(counts_dict.values())
    
    # Sort by value
    sorted_pairs = sorted(zip(keys, values), key=lambda x: x[1], reverse=True)
    keys = [x[0] for x in sorted_pairs]
    values = [x[1] for x in sorted_pairs]
    
    sns.barplot(x=values, y=keys, color=color)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   Saved {filename}")

def plot_degree_distribution(degrees, title, filename):
    plt.figure()
    plt.hist(degrees, bins=50, log=True, color='purple', alpha=0.7)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Count (Log Scale)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   Saved {filename}")

def plot_temporal_distribution(timestamps, title, filename):
    plt.figure()
    # Assuming timestamps are years
    years = pd.Series(timestamps).value_counts().sort_index()
    sns.barplot(x=years.index, y=years.values, color='coral')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"   Saved {filename}")

def analyze_graph(data: HeteroData, output_dir: str, prefix: str):
    print(f"\n🔍 Analyzing {prefix} Graph...")
    
    # 1. Node Counts
    node_counts = {nt: data[nt].num_nodes for nt in data.node_types}
    plot_counts(node_counts, f"{prefix}: Node Counts", "Count", f"{output_dir}/{prefix}_node_counts.png")
    
    # 2. Edge Counts
    edge_counts = {f"{et[0]}-{et[1]}-{et[2]}": data[et].edge_index.size(1) for et in data.edge_types}
    plot_counts(edge_counts, f"{prefix}: Edge Counts", "Count", f"{output_dir}/{prefix}_edge_counts.png")
    
    # 3. Degree Distribution (Target Nodes - Central Hubs)
    # Calculate total degree for 'target' nodes across all relations
    target_degrees = torch.zeros(data['target'].num_nodes)
    for et in data.edge_types:
        src, rel, dst = et
        if dst == 'target':
            # Add incoming edges
            deg = torch.bincount(data[et].edge_index[1], minlength=data['target'].num_nodes)
            target_degrees += deg.cpu()
        if src == 'target':
            # Add outgoing edges
            deg = torch.bincount(data[et].edge_index[0], minlength=data['target'].num_nodes)
            target_degrees += deg.cpu()
            
    plot_degree_distribution(target_degrees.numpy(), f"{prefix}: Target Degree Distribution", f"{output_dir}/{prefix}_target_degrees.png")
    
    # 4. Temporal Analysis (If applicable)
    if prefix == "Event":
        all_years = []
        for et in data.edge_types:
            if 'edge_time' in data[et]:
                all_years.append(data[et].edge_time.cpu().numpy())
        
        if all_years:
            all_years = np.concatenate(all_years)
            plot_temporal_distribution(all_years, "Event Distribution over Time", f"{output_dir}/event_temporal_dist.png")
            
            # Specific check for Clinical Trials (Supervision)
            # Try to find clinical trial edge
            clinical_et = None
            for et in data.edge_types:
                if "clinical_trial" in et[1]:
                    clinical_et = et
                    break
            
            if clinical_et and 'edge_time' in data[clinical_et]:
                clinical_years = data[clinical_et].edge_time.cpu().numpy()
                plot_temporal_distribution(clinical_years, "Clinical Trial Events over Time", f"{output_dir}/clinical_temporal_dist.png")

def main():
    parser = argparse.ArgumentParser(description="Analyze Graph Statistics")
    parser.add_argument("--file", type=str, default="output/progression/temporal_graph.pt", help="Path to temporal graph file")
    parser.add_argument("--output", type=str, default="output/analysis", help="Output directory for plots")
    args = parser.parse_args()
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    setup_plotting()
    
    # 1. Load Event Graph
    print(f"📂 Loading {args.file}...")
    event_data = load_event_graph(args.file, attach_features=False)
    
    # 2. Analyze Event Graph
    analyze_graph(event_data, args.output, "Event")
    
    # 3. Collapse to Time-Agnostic
    static_data = to_time_agnostic(event_data)
    
    # 4. Analyze Static Graph
    analyze_graph(static_data, args.output, "Static")
    
    # 5. Compare Collapsing Ratio
    print("\n📉 Collapsing Ratios:")
    for et in event_data.edge_types:
        orig = event_data[et].edge_index.size(1)
        collapsed = static_data[et].edge_index.size(1)
        if orig > 0:
            ratio = orig / collapsed
            print(f"   {et[1]}: {orig} -> {collapsed} (Ratio: {ratio:.2f}x)")
            
    print(f"\n✅ Analysis Complete. Plots saved to {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Master script to build all node features.
Invokes individual feature builders.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, args: list):
    cmd = [sys.executable, script_path] + args
    print(f"\n🚀 Running {script_path} {' '.join(args)}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Failed to run {script_path}")
        # We don't exit here, to allow partial completion if needed, or we can exit.
        # For now, print error.

def main():
    parser = argparse.ArgumentParser(description="Build all node features")
    parser.add_argument("--node-dir", default="output/nodes", help="Directory containing node parquets")
    parser.add_argument("--feature-data-dir", default="data/node_features", help="Raw feature data directory")
    parser.add_argument("--output-dir", default="data/node_features/processed", help="Output directory")
    args = parser.parse_args()
    
    node_dir = args.node_dir
    feature_dir = args.feature_data_dir
    output_dir = args.output_dir
    
    # 1. Target Features (Static + RNA)
    run_script("src/node_features/target_features.py", [
        "--base-dir", feature_dir,
        "--output-dir", output_dir
    ])
    
    # 2. Disease Features (Text)
    # Note: This is slow and requires GPU preferably.
    # We pass the parquet glob.
    run_script("src/node_features/disease_description.py", [
        "--disease-dir", node_dir,
        "--output-dir", output_dir,
        "--parquet-glob", "diseases.parquet",
        "--batch-size", "128" # Adjust based on hardware
    ])
    
    # 3. Molecule Features (Morgan Fingerprints)
    run_script("src/node_features/molecule_structure.py", [
        "--drug-dir", node_dir,
        "--output-dir", output_dir,
        "--parquet-glob", "molecule.parquet",
        "--id-col", "id", # Confirm col name in molecule.parquet
        "--smiles-col", "canonicalSmiles" # Confirm col name
    ])
    
    print("\n✅ All feature generation scripts invoked.")

if __name__ == "__main__":
    main()

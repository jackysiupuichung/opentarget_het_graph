#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List
import argparse
import yaml

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# -------------------------------------------------
# PARQUET INGESTION
# -------------------------------------------------
def load_drug_parquets(
    drug_dir: str,
    parquet_glob: str,
    id_col: str,
    smiles_col: str,
    kg_ids: list = None,
) -> pd.DataFrame:
    parquet_files = sorted(Path(drug_dir).glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {drug_dir} matching {parquet_glob}"
        )

    dfs = [
        pd.read_parquet(p, columns=[id_col, smiles_col])
        for p in parquet_files
    ]

    df = pd.concat(dfs, ignore_index=True)

    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""]

    # deterministic deduplication
    df = df.drop_duplicates(subset=[id_col], keep="first")
    
    # Filter to KG IDs if provided
    if kg_ids:
        kg_set = set(kg_ids)
        # We don't filter the DF strictly here because we might need to check if IDs exist at all
        # But for efficiency we can filter.
        # Imputation logic needs to know ALL kg_ids, even those NOT in this DF.
        df = df[df[id_col].isin(kg_set)]
        print(f"   Found data for {len(df):,} molecules (out of {len(kg_ids):,} requested)")

    return df.reset_index(drop=True)


# -------------------------------------------------
# MORGAN FINGERPRINT
# -------------------------------------------------
def smiles_to_morgan_fp(
    smiles: str,
    n_bits: int,
    radius: int,
) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Use modern generator to avoid deprecation warning
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)


    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def build_drug_embeddings(args: argparse.Namespace, kg_ids: list = None) -> Dict[str, np.ndarray]:
    df = load_drug_parquets(
        drug_dir=args.drug_dir,
        parquet_glob=args.parquet_glob,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
        kg_ids=kg_ids,
    )

    embeddings: Dict[str, np.ndarray] = {}
    skipped_stats = {
        "missing_smiles": 0,
        "rdkit_error": 0,
    }
    
    # 1. Generate valid embeddings first
    print("\n   Generating Morgan fingerprints for all molecules...")
    valid_vectors = []
    
    for _, row in df.iterrows():
        drug_id = row[args.id_col]
        smiles = row[args.smiles_col]
        
        # Check SMILES existence
        if not isinstance(smiles, str) or not smiles.strip():
            skipped_stats["missing_smiles"] += 1
            continue
            
        # Generate Fingerprint
        fp = smiles_to_morgan_fp(
            smiles,
            n_bits=args.fp_dim,
            radius=args.radius,
        )

        if fp is None:
            skipped_stats["rdkit_error"] += 1
            continue

        embeddings[drug_id] = fp
        valid_vectors.append(fp)

    # 2. Calculate Mean Vector
    if valid_vectors:
        mean_vector = np.mean(valid_vectors, axis=0).astype(np.float32)   
        print(f"   Calculated mean vector from {len(valid_vectors):,} molecules")
    else:
        print("   ⚠️ No valid molecules found! Using zero vector.")
        mean_vector = np.zeros((args.fp_dim,), dtype=np.float32)

    # 3. Impute Missing (for ALL requested KG IDs)
    if kg_ids:
        print("\n   Imputing missing molecules with mean vector...")
        imputed_count = 0
        
        for mid in kg_ids:
            if mid not in embeddings:
                embeddings[mid] = mean_vector
                imputed_count += 1
        
        print(f"   Imputed {imputed_count:,} molecules (Total requested: {len(kg_ids):,})")

    return embeddings, skipped_stats


# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
def save_outputs(
    embeddings: Dict[str, np.ndarray],
    skipped_stats: Dict[str, int],
    args: argparse.Namespace,
):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert all to tensors (ensure float for consistency if mean used)
    # The mean vector is float, but fingerprints were int8. 
    # To mix them, we should convert everything to float tensors.
    tensor_dict = {}
    for k, v in embeddings.items():
        tensor_dict[k] = torch.tensor(v, dtype=torch.float)

    # PyTorch tensor store
    torch.save(
        tensor_dict,
        out_dir / "molecule_morgan_fingerprints.pt",
    )

    # Coverage Report
    print("\n" + "=" * 60)
    print("MOLECULE FEATURE COVERAGE REPORT")
    print("=" * 60)
    print(f"Total molecules processed/imputed: {len(embeddings):,}")
    print("-" * 60)
    print("SKIPPED REASONS (imputed with mean):")
    for reason, count in skipped_stats.items():
        if count > 0:
            print(f"   ❌ {reason}: {count:,}")
    print("=" * 60 + "\n")

    # YAML metadata
    payload = {
        "meta": {
            "fp_type": "morgan",
            "radius": args.radius,
            "fp_dim": args.fp_dim,
            "num_drugs": len(embeddings),
            "skipped_stats": skipped_stats,
            "source_dir": args.drug_dir,
            "imputed": True
        },
    }

    with open(out_dir / "drug_morgan_fingerprints.yaml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


# -------------------------------------------------
# ARGPARSE
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Morgan fingerprint drug embeddings from parquet directory"
    )

    parser.add_argument("--drug-dir", required=True, help="Evidence directory with molecule parquets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kg-ids-file", default=None, help="Parquet file with KG molecule IDs to filter to")

    parser.add_argument("--parquet-glob", default="part-*.parquet")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--smiles-col", default="canonicalSmiles")

    parser.add_argument("--fp-dim", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)

    return parser.parse_args()


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
def main():
    args = parse_args()
    
    # Load KG IDs if provided
    kg_ids = None
    if args.kg_ids_file and Path(args.kg_ids_file).exists():
        print(f"Loading KG molecule IDs from {args.kg_ids_file}")
        kg_df = pd.read_parquet(args.kg_ids_file)
        kg_ids = kg_df['id'].tolist()
        print(f"Filtering to {len(kg_ids):,} KG molecule IDs")
    
    embeddings, skipped = build_drug_embeddings(args, kg_ids=kg_ids)
    save_outputs(embeddings, skipped, args)


if __name__ == "__main__":
    main()

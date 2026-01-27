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
from sklearn.decomposition import PCA


# -------------------------------------------------
# PARQUET INGESTION
# -------------------------------------------------
def load_drug_parquets(
    drug_dir: str,
    parquet_glob: str,
    id_col: str,
    smiles_col: str,
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

    return df.reset_index(drop=True)


# -------------------------------------------------
# BIOLOGIC / PEPTIDE HEURISTIC
# -------------------------------------------------
def is_biologic_or_peptide(smiles: str, max_len: int) -> bool:
    """
    Simple heuristic:
    - very long SMILES
    - RDKit cannot parse
    """
    if len(smiles) > max_len:
        return True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True

    return False


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

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
    )

    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def build_drug_embeddings(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    df = load_drug_parquets(
        drug_dir=args.drug_dir,
        parquet_glob=args.parquet_glob,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
    )

    embeddings: Dict[str, np.ndarray] = {}
    skipped_biologics: List[str] = []

    for _, row in df.iterrows():
        drug_id = row[args.id_col]
        smiles = row[args.smiles_col]

        if is_biologic_or_peptide(smiles, args.max_smiles_len):
            skipped_biologics.append(drug_id)
            continue

        fp = smiles_to_morgan_fp(
            smiles,
            n_bits=args.fp_dim,
            radius=args.radius,
        )

        if fp is None:
            skipped_biologics.append(drug_id)
            continue

        embeddings[drug_id] = fp

    return embeddings, skipped_biologics


# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
def save_outputs(
    embeddings: Dict[str, np.ndarray],
    skipped: List[str],
    args: argparse.Namespace,
):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch tensor store
    torch.save(
        {k: torch.tensor(v) for k, v in embeddings.items()},
        out_dir / "drug_morgan_fingerprints.pt",
    )

    # YAML metadata + (optional) PCA
    payload = {
        "meta": {
            "fp_type": "morgan",
            "radius": args.radius,
            "fp_dim": args.fp_dim,
            "num_drugs": len(embeddings),
            "num_skipped_biologics": len(skipped),
            "source_dir": args.drug_dir,
        },
        "skipped_biologics": skipped,
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

    parser.add_argument("--drug-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--parquet-glob", default="part-*.parquet")
    parser.add_argument("--id-col", default="drugId")
    parser.add_argument("--smiles-col", default="canonicalSmiles")

    parser.add_argument("--fp-dim", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)

    parser.add_argument(
        "--max-smiles-len",
        type=int,
        default=300,
        help="Above this length treated as biologic/peptide",
    )

    return parser.parse_args()


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
def main():
    args = parse_args()
    embeddings, skipped = build_drug_embeddings(args)
    save_outputs(embeddings, skipped, args)


if __name__ == "__main__":
    main()

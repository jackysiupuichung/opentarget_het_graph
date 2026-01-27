#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List
import argparse
import yaml

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# -------------------------------------------------
# PARQUET INGESTION
# -------------------------------------------------
def load_disease_parquets(
    disease_dir: str,
    parquet_glob: str,
    id_col: str,
    text_col: str,
) -> pd.DataFrame:
    parquet_files = sorted(Path(disease_dir).glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {disease_dir} matching {parquet_glob}"
        )

    dfs = [
        pd.read_parquet(p, columns=[id_col, text_col])
        for p in parquet_files
    ]

    df = pd.concat(dfs, ignore_index=True)
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df = df.drop_duplicates(subset=[id_col], keep="first")

    return df.reset_index(drop=True)


# -------------------------------------------------
# BATCH ENCODING
# -------------------------------------------------
@torch.no_grad()
def encode_batch_mean_pool(
    texts: List[str],
    tokenizer,
    model,
    max_length: int,
    device: str,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
def build_disease_embeddings(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    df = load_disease_parquets(
        disease_dir=args.disease_dir,
        parquet_glob=args.parquet_glob,
        id_col=args.id_col,
        text_col=args.text_col,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    ids = df[args.id_col].tolist()
    texts = df[args.text_col].tolist()

    embeddings: Dict[str, np.ndarray] = {}

    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        batch_ids = ids[i : i + args.batch_size]

        batch_embs = encode_batch_mean_pool(
            batch_texts,
            tokenizer,
            model,
            args.max_length,
            args.device,
        )

        for did, emb in zip(batch_ids, batch_embs):
            if emb.shape[0] != args.embedding_dim:
                emb = emb[: args.embedding_dim]
            embeddings[did] = emb

    return embeddings


# -------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------
def save_outputs(
    embeddings: Dict[str, np.ndarray],
    args: argparse.Namespace,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {k: torch.tensor(v) for k, v in embeddings.items()},
        output_dir / "disease_embeddings.pt",
    )

    with open(output_dir / "disease_embeddings.yaml", "w") as f:
        yaml.safe_dump(
            {
                "meta": {
                    "model": args.model_name,
                    "embedding_dim": args.embedding_dim,
                    "num_diseases": len(embeddings),
                    "source_dir": args.disease_dir,
                },
                "embeddings": {k: v.tolist() for k, v in embeddings.items()},
            },
            f,
            sort_keys=False,
        )


# -------------------------------------------------
# ARGPARSE
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GPT-based disease embeddings from parquet directory"
    )

    parser.add_argument("--disease-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--parquet-glob", default="part-*.parquet")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="description")

    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
def main():
    args = parse_args()
    embeddings = build_disease_embeddings(args)
    save_outputs(embeddings, args)


if __name__ == "__main__":
    main()

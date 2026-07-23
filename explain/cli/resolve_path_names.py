"""Rewrite PaGE-Link path strings from internal node indices to accession+name.

Reads per_pair_paths.parquet (whose `path` column has tokens like
`target#7886` / `disease#171`), resolves each (type, idx) to its external
accession via the mappings file's node_mapping, and to a human name via the KG
node parquets. CPU-only: torch.load the mappings dict + read small node parquets;
no graph/model. Prints a readable, paper-ready path table.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import torch


def _clean(x):
    s = "" if x is None else str(x)
    return s.strip()


def load_id_and_name_maps(mappings_file: str, node_base: str):
    m = torch.load(mappings_file, weights_only=False)
    # {node_type: {idx -> accession}}
    id_maps = {nt: {int(v): k for k, v in nm.items()}
               for nt, nm in m["node_mapping"].items()}
    nb = Path(node_base)
    name_maps = {}

    def read(p, idcol, namecol):
        df = pd.read_parquet(p, columns=[idcol, namecol])
        return {str(k): _clean(v) for k, v in zip(df[idcol], df[namecol]) if _clean(v)}

    nodes = nb / "evidences" / "nodes"
    if (nodes / "targets.parquet").exists():
        name_maps["target"] = read(nodes / "targets.parquet", "id", "name")
    if (nodes / "reactome.parquet").exists():
        name_maps["reactome"] = read(nodes / "reactome.parquet", "id", "name")
    if (nodes / "go_ontology_terms.parquet").exists():
        name_maps["go"] = read(nodes / "go_ontology_terms.parquet", "id", "name")
    # disease names: evidenceDated/diseases preferred, fallback nodes/diseases desc
    dmap = {}
    ed = nb / "evidenceDated" / "diseases"
    if ed.exists():
        try:
            dmap.update(read(ed, "id", "name"))
        except Exception:
            pass
    fb = nodes / "diseases.parquet"
    if fb.exists():
        cols = pd.read_parquet(fb).columns
        col = "description" if "description" in cols else ("name" if "name" in cols else None)
        if col:
            for k, v in read(fb, "id", col).items():
                dmap.setdefault(k, v)
    if dmap:
        name_maps["disease"] = dmap
    if (nodes / "molecule.parquet").exists() or (nb / "evidenceDated" / "molecule").exists():
        mp = nb / "evidenceDated" / "molecule"
        if mp.exists():
            try:
                name_maps["molecule"] = read(mp, "id", "name")
            except Exception:
                pass
    return id_maps, name_maps


def resolve_token(tok, id_maps, name_maps):
    m = re.match(r"([a-z_]+)#(\d+)", tok)
    if not m:
        return tok
    nt, idx = m.group(1), int(m.group(2))
    acc = id_maps.get(nt, {}).get(idx)
    if acc is None:
        return tok
    name = name_maps.get(nt, {}).get(str(acc))
    return f"{acc}" + (f" ({name})" if name else "")


def rewrite_path(path, id_maps, name_maps):
    return re.sub(r"[a-z_]+#\d+",
                  lambda m: resolve_token(m.group(0), id_maps, name_maps), path)


def main(a):
    id_maps, name_maps = load_id_and_name_maps(a.mappings_file, a.node_base)
    df = pd.read_parquet(a.paths_parquet)
    df = df[df["rank"] == 0] if a.best_only else df
    pd.set_option("display.max_colwidth", 400)
    for _, r in df.iterrows():
        t = resolve_token(f"target#{0}", id_maps, name_maps)  # noqa (placeholder)
        readable = rewrite_path(r["path"], id_maps, name_maps)
        print(f"\n[{r['target_id']} -> {r['disease_id']}] rank={r['rank']} "
              f"hops={r['n_hops']} mean_cost={r['mean_cost']:.3f}")
        print("  " + readable)


def _p():
    p = argparse.ArgumentParser()
    p.add_argument("--paths-parquet", required=True)
    p.add_argument("--mappings-file", required=True)
    p.add_argument("--node-base", required=True,
                   help="Dir containing evidences/nodes + evidenceDated.")
    p.add_argument("--best-only", action="store_true",
                   help="Only rank-0 path per pair.")
    return p.parse_args()


if __name__ == "__main__":
    main(_p())

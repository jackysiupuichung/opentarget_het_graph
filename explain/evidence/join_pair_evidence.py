"""Recover the underlying OT evidence rows for the edges of an already-explained
(target, disease) pair, WITHOUT re-running the model.

Why this exists
---------------
``explain_advancement.py`` only emits ``per_pair_evidence.parquet`` when run
with ``--raw-edges-dir``. The headline run was evidence-free, and the graph the
checkpoint was trained on has since been overwritten by a newer (68-relation)
build, so re-running the full explainer crashes on a state_dict shape mismatch.

But the evidence lookup never needed the model or the graph: it is a pure join
of (edge_type, sourceId, targetId) against the parsed evidence parquets under
``evidences/edges/``. This script does exactly that, driven from the existing
``per_pair_edges.parquet``.

Index → accession mapping comes from the mappings file's ``node_mapping``.
Because that file was rebuilt the same day as the graph, we VALIDATE up front
that the queried pair's own indices still resolve to the expected accessions;
if they don't, the mappings have drifted relative to the saved edges and we
abort rather than emit a silently-misjoined parquet.

Reverse edges (``rev_*``) are folded back to their canonical forward
orientation (strip ``rev_`` + swap src/dst) before keying, mirroring viz.py,
since the raw evidence parquets only carry forward (sourceId, targetId).

CPU only. Intended for ``computeshort`` via sbatch.
"""

from __future__ import annotations

import argparse
import os
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
import torch


def _build_id_maps(mappings_file: str) -> dict[str, dict[int, str]]:
    """{node_type: {internal_idx: external_accession}} from the mappings file."""
    mappings = torch.load(mappings_file, weights_only=False)
    return {
        nt: {int(v): k for k, v in nm.items()}
        for nt, nm in mappings["node_mapping"].items()
    }


class RawEdgeIndex:
    """Lazy index over parsed-edge parquets, grouped by
    (source_type, relation, target_type) and looked up by (sourceId, targetId).

    Lifted from explain_advancement._RawEdgeIndex but standalone so this script
    has no import-time dependency on the (graph-loading) driver module.
    """

    def __init__(self, raw_edges_dir: str):
        self.raw_edges_dir = raw_edges_dir
        self._files_by_triple: dict[tuple, list[str]] = {}
        self._loaded: dict[str, dict] = {}
        self._index_files()

    def _index_files(self) -> None:
        for path in sorted(glob(os.path.join(self.raw_edges_dir, "*.parquet"))):
            try:
                pf = pq.ParquetFile(path)
                batch = next(pf.iter_batches(
                    batch_size=1,
                    columns=["source_type", "relation", "target_type"],
                ))
                head = batch.to_pandas()
            except (StopIteration, Exception):
                continue
            if head.empty:
                continue
            triple = (
                str(head["source_type"].iloc[0]),
                str(head["relation"].iloc[0]),
                str(head["target_type"].iloc[0]),
            )
            self._files_by_triple.setdefault(triple, []).append(path)

    def triples(self) -> set[tuple]:
        return set(self._files_by_triple)

    def _ensure_loaded(self, path: str) -> dict:
        if path in self._loaded:
            return self._loaded[path]
        cols_meta = set(pq.ParquetFile(path).schema_arrow.names)
        evidence_col = "literature" if "literature" in cols_meta else (
            "id" if "id" in cols_meta else None
        )
        load_cols = ["sourceId", "targetId", "datasourceId", "year"]
        if "score" in cols_meta:
            load_cols.append("score")     # OT per-evidence score, for presentation
        if evidence_col:
            load_cols.append(evidence_col)
        df = pd.read_parquet(path, columns=load_cols)
        df["sourceId"] = df["sourceId"].astype(str)
        df["targetId"] = df["targetId"].astype(str)
        df = df.sort_values(["sourceId", "targetId"]).set_index(
            ["sourceId", "targetId"], drop=False
        )
        cache = {"evidence_col": evidence_col, "df": df}
        self._loaded[path] = cache
        return cache

    def get(self, edge_type: tuple, source_id: str, target_id: str,
            year_max: int | None) -> pd.DataFrame:
        files = self._files_by_triple.get(edge_type, [])
        if not files:
            return pd.DataFrame()
        chunks = []
        for path in files:
            cache = self._ensure_loaded(path)
            df = cache["df"]
            key = (str(source_id), str(target_id))
            try:
                hit = df.loc[[key]]
            except KeyError:
                continue
            if year_max is not None and "year" in hit.columns:
                hit = hit[hit["year"] <= year_max]
                if hit.empty:
                    continue
            cols = ["sourceId", "targetId", "datasourceId", "year"]
            if "score" in hit.columns:
                cols.append("score")
            if cache["evidence_col"]:
                cols.append(cache["evidence_col"])
            chunks.append(hit[cols].reset_index(drop=True).copy())
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)


def _canonicalise(edge_type: str, src_idx: int, dst_idx: int) -> tuple[str, str, str, int, int]:
    """Strip rev_ and flip src/dst so the edge points in the forward
    orientation the raw evidence parquets are keyed on.

    Returns (src_type, relation, dst_type, src_idx, dst_idx) in forward form.
    """
    src_type, rel, dst_type = edge_type.split("::", 2)
    if rel.startswith("rev_"):
        rel = rel[4:]
        src_type, dst_type = dst_type, src_type
        src_idx, dst_idx = dst_idx, src_idx
    return src_type, rel, dst_type, src_idx, dst_idx


def main(args: argparse.Namespace) -> None:
    id_maps = _build_id_maps(args.mappings_file)

    edges = pd.read_parquet(args.edges_parquet)
    edges = edges[(edges["target_id"] == args.target_id)
                  & (edges["disease_id"] == args.disease_id)].copy()
    if edges.empty:
        raise SystemExit(
            f"[join] no edges for {args.target_id} / {args.disease_id} in "
            f"{args.edges_parquet}")
    print(f"[join] {len(edges)} edges for the pair", flush=True)

    # ── Validation gate: confirm the mappings still reconcile with the saved
    # edges. The supervision pair's own node indices must round-trip to the
    # expected accessions; otherwise the mappings drifted and any join is
    # garbage. We find them via the target<->disease supervision-style edges.
    def _resolves(node_type: str, idx: int, expect: str) -> bool:
        return id_maps.get(node_type, {}).get(int(idx)) == expect

    ok = False
    for _, r in edges.iterrows():
        st, rel, dt, si, di = _canonicalise(r["edge_type"], int(r["src"]), int(r["dst"]))
        if st == "target" and dt == "disease":
            if _resolves("target", si, args.target_id) and _resolves("disease", di, args.disease_id):
                ok = True
                break
        if st == "disease" and dt == "target":
            if _resolves("disease", si, args.disease_id) and _resolves("target", di, args.target_id):
                ok = True
                break
    if not ok:
        raise SystemExit(
            "[join] ABORT: queried pair's indices do NOT resolve to the "
            "expected accessions under the current mappings file. The mappings "
            "have drifted relative to the saved per_pair_edges.parquet — the "
            "join would silently mis-key. Recover the mappings file that was "
            "in force when the edges were written before proceeding.")
    print("[join] validation OK: pair indices reconcile with mappings", flush=True)

    raw_index = RawEdgeIndex(args.raw_edges_dir)
    print(f"[join] indexed raw edges: {args.raw_edges_dir} "
          f"({len(raw_index.triples())} triples)", flush=True)

    out_rows = []
    miss_no_acc = 0
    for _, r in edges.iterrows():
        st, rel, dt, si, di = _canonicalise(r["edge_type"], int(r["src"]), int(r["dst"]))
        src_acc = id_maps.get(st, {}).get(int(si))
        dst_acc = id_maps.get(dt, {}).get(int(di))
        if src_acc is None or dst_acc is None:
            miss_no_acc += 1
            continue
        hits = raw_index.get((st, rel, dt), src_acc, dst_acc, year_max=args.year_max)
        if hits.empty:
            continue
        hits = hits.copy()
        hits["target_id"] = args.target_id
        hits["disease_id"] = args.disease_id
        hits["edge_type"] = r["edge_type"]          # original (with rev_) for traceability
        hits["fwd_edge_type"] = "::".join((st, rel, dt))
        hits["src"] = int(r["src"])
        hits["dst"] = int(r["dst"])
        hits["ig_total"] = r.get("ig_total")
        hits["attention"] = r.get("attention")
        if "literature" not in hits.columns:
            hits["literature"] = None
        if "id" not in hits.columns:
            hits["id"] = None
        out_rows.append(hits)

    cols = ["target_id", "disease_id", "edge_type", "fwd_edge_type", "src", "dst",
            "ig_total", "attention", "sourceId", "targetId", "datasourceId",
            "year", "score", "literature", "id"]
    if out_rows:
        out = pd.concat(out_rows, ignore_index=True)
        out = out[[c for c in cols if c in out.columns]]
    else:
        out = pd.DataFrame(columns=cols)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[join] edges with no accession mapping: {miss_no_acc}", flush=True)
    print(f"[join] wrote {len(out)} evidence rows -> {args.out}", flush=True)
    if len(out):
        print("[join] evidence rows per datasource:", flush=True)
        print(out["datasourceId"].value_counts().to_string(), flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--edges-parquet", required=True,
                   help="Existing per_pair_edges.parquet from the evidence-free run.")
    p.add_argument("--mappings-file", required=True,
                   help="temporal_graph_*_mappings.pt providing node_mapping.")
    p.add_argument("--raw-edges-dir", required=True,
                   help="Directory of parsed evidence parquets (.../evidences/edges/).")
    p.add_argument("--target-id", required=True)
    p.add_argument("--disease-id", required=True)
    p.add_argument("--year-max", type=int, default=None,
                   help="Optional: cap evidence to rows with year <= this "
                        "(the decision year, to stay causally clean).")
    p.add_argument("--out", required=True, help="Output parquet path.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

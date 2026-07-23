"""Export the significant-edge evidence of an explained pair as per-edge JSON.

Pipeline position
-----------------
    explain_advancement.py  (--raw-edges-dir)  ->  per_pair_evidence.parquet
        OR
    join_pair_evidence.py                       ->  <pair>_evidence.parquet
                                                        |
                                                        v
    export_pair_evidence_json.py  ->  <pair>_evidence.json        (case study)
                                  ->  <pair>_evidence_flat.parquet (feeds present_*)

What it produces
----------------
A ChronoMedKG-style (arXiv:2605.22734) decomposition of a single prediction:
the (target, disease) score is broken into its significant subgraph edges, and
each edge carries its OpenTargets evidence — datasource, year, score, the
citable OT evidence ``id``, and the cited papers' PMIDs — so a reader can
dereference every driver back to its source.

All data is LOCAL: it comes from the parsed 26.03 evidence parquets that
``join_pair_evidence.py`` already attached to the edges. No network calls.
(A later optional phase can enrich these records from the live OT GraphQL API,
keyed on the same ``id`` hash.)

The "significant" edges are the top-|IG| edges, selected with the SAME ranking
``present_pair_evidence.py`` uses for its Level-2 drill-down, so the JSON and the
markdown brief agree on which edges count.

CPU + IO only. Runnable directly; for a many-pair sweep submit via sbatch on
``computeshort``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from present_pair_evidence import (
    OT_EVIDENCE_URL,
    PUBMED_URL,
    _attach_ig,
    _fmt_relation,
    _pmids,
    _year_span,
    edge_col_of,
    top_edge_rows,
)


def _evidence_records(g: pd.DataFrame) -> list[dict]:
    """One record per distinct OT evidence ``id`` backing a single edge.

    Collapses the raw per-(id, year) rows to one entry per evidence id, with its
    year span, max score, PMIDs, and dereferenceable URLs.
    """
    has_id = "id" in g.columns
    has_score = "score" in g.columns
    has_lit = "literature" in g.columns

    records = []
    # Group by the citable evidence id (datasource carried along for the rare
    # case the same id appears under two datasource labels).
    keys = ["id"] if has_id else ["datasourceId"]
    keys = [k for k in keys if k in g.columns]
    for kv, sub in g.groupby(keys, dropna=False):
        kv = kv if isinstance(kv, tuple) else (kv,)
        d = dict(zip(keys, kv))
        evid = d.get("id")
        rec = {
            "id": None if evid is None or pd.isna(evid) else str(evid),
            "datasourceId": (str(sub["datasourceId"].iloc[0])
                             if "datasourceId" in sub.columns else None),
            "year": _year_span(sub.get("year", pd.Series(dtype=float))),
            "n_rows": int(len(sub)),
        }
        if has_score:
            sc = pd.to_numeric(sub["score"], errors="coerce").dropna()
            rec["score"] = float(sc.max()) if not sc.empty else None
        pmids = _pmids(sub["literature"]) if has_lit else []
        rec["literature"] = pmids
        rec["pmid_urls"] = [f"{PUBMED_URL}{p}" for p in pmids]
        rec["ot_url"] = f"{OT_EVIDENCE_URL}{rec['id']}" if rec["id"] else None
        records.append(rec)

    # Strongest-scored / most-cited first within an edge.
    records.sort(key=lambda r: (-(r.get("score") or 0.0), -len(r["literature"])))
    return records


def build_pair_json(ev: pd.DataFrame, target_id: str, disease_id: str,
                    top_edges: int) -> dict:
    """Assemble the per-pair decomposition: significant edges (by signed IG),
    each with its backing OT evidence records."""
    edge_col = edge_col_of(ev)
    keep = top_edge_rows(ev, top_edges)

    edges = []
    edge_ids = keep.drop_duplicates([edge_col, "src", "dst"])
    for _, ek in edge_ids.iterrows():
        sub = keep[(keep[edge_col] == ek[edge_col])
                   & (keep["src"] == ek["src"])
                   & (keep["dst"] == ek["dst"])]
        ig = pd.to_numeric(pd.Series([ek.get("ig_total")]), errors="coerce").iloc[0]
        attn = pd.to_numeric(pd.Series([ek.get("attention")]), errors="coerce").iloc[0]
        src_acc = sub["sourceId"].iloc[0] if "sourceId" in sub.columns else None
        dst_acc = sub["targetId"].iloc[0] if "targetId" in sub.columns else None
        edges.append({
            "relation": _fmt_relation(ek[edge_col]),
            "edge_type": str(ek[edge_col]),
            "src": None if src_acc is None else str(src_acc),
            "dst": None if dst_acc is None else str(dst_acc),
            "ig_total": None if pd.isna(ig) else float(ig),
            "attention": None if pd.isna(attn) else float(attn),
            "n_evidence": int(sub["id"].nunique()) if "id" in sub.columns else int(len(sub)),
            "evidence": _evidence_records(sub),
        })

    # Strongest positive driver first (matches the markdown brief ordering).
    edges.sort(key=lambda e: (e["ig_total"] if e["ig_total"] is not None
                              else float("-inf")), reverse=True)
    return {"target_id": target_id, "disease_id": disease_id,
            "n_edges": len(edges), "edges": edges}


def main(args: argparse.Namespace) -> None:
    ev = pd.read_parquet(args.evidence_parquet)
    if args.edges_parquet:
        ev = _attach_ig(ev, args.edges_parquet)
    if args.target_id and args.disease_id:
        ev = ev[(ev["target_id"] == args.target_id)
                & (ev["disease_id"] == args.disease_id)].copy()
    if ev.empty:
        raise SystemExit(f"[export] no evidence rows in "
                         f"{args.evidence_parquet} for the requested filter")

    pairs = ev[["target_id", "disease_id"]].drop_duplicates()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[export] {len(pairs)} pair(s) to export", flush=True)

    for _, pr in pairs.iterrows():
        t, d = pr["target_id"], pr["disease_id"]
        sub = ev[(ev["target_id"] == t) & (ev["disease_id"] == d)]
        stem = f"{t}__{d}".replace("/", "_")

        payload = build_pair_json(sub, t, d, args.top_edges)
        with open(os.path.join(args.out_dir, f"{stem}_evidence.json"), "w") as fh:
            json.dump(payload, fh, indent=2)

        # Flat parquet of just the significant-edge rows, in the schema
        # present_pair_evidence.py consumes (so the two scripts compose).
        flat = top_edge_rows(sub, args.top_edges)
        flat.to_parquet(
            os.path.join(args.out_dir, f"{stem}_evidence_flat.parquet"),
            index=False)

        n_ev = sum(len(e["evidence"]) for e in payload["edges"])
        n_pmid = sum(1 for e in payload["edges"] for r in e["evidence"]
                     if r["literature"])
        print(f"[export] {t} -> {d}: {payload['n_edges']} edges, "
              f"{n_ev} evidence records ({n_pmid} with PMIDs) -> {stem}_evidence.json",
              flush=True)
        if "datasourceId" in flat.columns and len(flat):
            print("[export]   rows per datasource:", flush=True)
            print(flat["datasourceId"].value_counts().to_string(), flush=True)

    print(f"[export] done -> {args.out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export significant-edge OT evidence of explained pairs as "
                    "per-edge JSON (local, no network).")
    p.add_argument("--evidence-parquet", required=True,
                   help="per_pair_evidence.parquet (explain_advancement "
                        "--raw-edges-dir) or join_pair_evidence.py output.")
    p.add_argument("--edges-parquet", default=None,
                   help="Sibling per_pair_edges.parquet. When the evidence file "
                        "lacks ig_total/attention, join them in from here.")
    p.add_argument("--out-dir", required=True,
                   help="Directory for <pair>_evidence.json + "
                        "<pair>_evidence_flat.parquet.")
    p.add_argument("--target-id", default=None,
                   help="Optional: export only this pair.")
    p.add_argument("--disease-id", default=None)
    p.add_argument("--top-edges", type=int, default=15,
                   help="How many top-|IG| edges to export (default 15). Matches "
                        "the present_pair_evidence.py Level-2 selection.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

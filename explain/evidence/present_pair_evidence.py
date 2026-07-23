"""Turn an explained (target, disease) pair into a human-readable evidence brief.

Pipeline position
-----------------
    explain_advancement.py  (--raw-edges-dir)  ->  per_pair_evidence.parquet
        OR
    join_pair_evidence.py                       ->  <pair>_evidence.parquet
                                                        |
                                                        v
    present_pair_evidence.py  ->  <pair>.md   (readable brief)
                              ->  <pair>_summary.parquet  (tidy table)

What it shows
-------------
We deliberately do NOT resolve to PMIDs (the id->literature join is lossy: many
OT evidence rows cite no paper). Instead each prediction driver is presented at
the level the explainer is faithful at:

  * the EDGE  -- which relation in the subgraph drove the score, ranked by the
    model's Integrated-Gradients attribution (``ig_total``), with attention as
    a secondary signal;
  * the DATASOURCE(S) backing that edge (``datasourceId``), so a reader sees
    whether a driver rests on a clinical trial, a genetic association, a
    pathway annotation, text-mining, etc.;
  * the OT evidence SCORE and YEAR for those backing rows, summarised
    (max / mean score, evidence count, year span).

So a driver reads as, e.g.:

    target_genetic_association_disease   IG=+0.412  attn=0.21
        eva                 n=3   score max 0.90 / mean 0.74   2018-2021
        gene_burden         n=1   score 0.55                   2020

i.e. "the genetic-association edge was the largest positive contributor, and it
is backed by 3 EVA rows (top score 0.90) and 1 gene-burden row".
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _fmt_relation(edge_type: str) -> str:
    """``target::genetic_association::disease`` (or fwd form) -> readable rel.

    Keep the full src::rel::dst but drop a leading ``rev_`` so the reader sees
    the canonical relation name.
    """
    parts = str(edge_type).split("::")
    if len(parts) == 3:
        st, rel, dt = parts
        rel = rel[4:] if rel.startswith("rev_") else rel
        return f"{st} --{rel}--> {dt}"
    return str(edge_type)


def _year_span(years: pd.Series) -> str:
    yrs = pd.to_numeric(years, errors="coerce").dropna()
    if yrs.empty:
        return "year n/a"
    lo, hi = int(yrs.min()), int(yrs.max())
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


# Where a reader dereferences the provenance handles we carry per edge.
OT_EVIDENCE_URL = "https://platform.opentargets.org/evidence/"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/"


def _pmids(literature: pd.Series) -> list[str]:
    """Flatten a column of PMID lists/scalars into a sorted unique str list.

    The ``literature`` column holds numpy arrays of PMID strings for the
    datasources that cite papers (europepmc, intact, gene_ontology) and ``None``
    elsewhere; normalise both to a flat de-duplicated list.
    """
    out: set[str] = set()
    for v in literature.dropna():
        if isinstance(v, (list, tuple, np.ndarray)):
            out.update(str(x) for x in v if x is not None and str(x) != "")
        elif str(v):
            out.add(str(v))
    return sorted(out)


def build_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (edge, datasource): IG/attention of the edge + score/year
    rollup of that datasource's backing evidence rows.

    Sorted by |ig_total| desc so the strongest prediction drivers come first.
    """
    # Edge identity = forward edge type + endpoints (so rev/forward fold together
    # only if join already canonicalised; we group on whatever edge_type is given).
    edge_col = "fwd_edge_type" if "fwd_edge_type" in ev.columns else "edge_type"
    has_score = "score" in ev.columns
    # ``attention`` is present on join_pair_evidence.py output but NOT on the
    # explainer's own per_pair_evidence.parquet (ig only). Guard so either feeds.
    has_attn = "attention" in ev.columns

    grp_keys = [edge_col, "src", "dst", "datasourceId"]
    grp_keys = [k for k in grp_keys if k in ev.columns]

    rows = []
    for keys, g in ev.groupby(grp_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        d = dict(zip(grp_keys, keys))
        ig = pd.to_numeric(g.get("ig_total"), errors="coerce")
        rec = {
            "edge": d.get(edge_col),
            "relation": _fmt_relation(d.get(edge_col, "")),
            "src": d.get("src"),
            "dst": d.get("dst"),
            "datasource": d.get("datasourceId"),
            "n_evidence": len(g),
            "ig_total": float(ig.dropna().iloc[0]) if ig.notna().any() else np.nan,
            "year_span": _year_span(g.get("year", pd.Series(dtype=float))),
        }
        if has_attn:
            attn = pd.to_numeric(g["attention"], errors="coerce")
            rec["attention"] = float(attn.dropna().iloc[0]) if attn.notna().any() else np.nan
        if has_score:
            sc = pd.to_numeric(g["score"], errors="coerce").dropna()
            rec["score_max"] = float(sc.max()) if not sc.empty else np.nan
            rec["score_mean"] = float(sc.mean()) if not sc.empty else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Order by SIGNED IG descending: the edges that pushed the prediction up
    # (largest positive attribution) lead; negative contributors sink to the
    # bottom. NaN IG (no attribution) sorts last.
    out["_ig"] = out["ig_total"].fillna(float("-inf"))
    out = out.sort_values(["_ig", "n_evidence"], ascending=False)
    return out.drop(columns="_ig").reset_index(drop=True)


def build_relation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """LEVEL 1 — one row per relation TYPE (collapse all its edges).

    Rolls the per-edge×datasource ``summary`` up to the relation, so the 985
    near-zero reactome ``affected_pathway`` edges become a single line whose
    ``ig_sum`` reflects their (tiny) total contribution. This is the figure
    legend: "which relation type moved the prediction, and by how much".
    """
    if summary.empty:
        return summary
    # IG is per-edge; de-dup to one IG per (relation, src, dst) before summing
    # so the same edge isn't counted once per datasource.
    has_attn = "attention" in summary.columns
    keep_cols = ["relation", "src", "dst", "ig_total"] + (["attention"] if has_attn else [])
    edges = summary.drop_duplicates(["relation", "src", "dst"])[keep_cols]
    aggs = dict(
        ig_sum=("ig_total", "sum"),
        ig_max=("ig_total", "max"),
        n_edges=("ig_total", "size"),
    )
    if has_attn:
        aggs["attn_max"] = ("attention", "max")
    agg = edges.groupby("relation").agg(**aggs).reset_index()
    # datasources + evidence count per relation (from the full summary).
    ds = summary.groupby("relation").agg(
        n_evidence=("n_evidence", "sum"),
        datasources=("datasource", lambda s: sorted(set(map(str, s.dropna())))),
    ).reset_index()
    out = agg.merge(ds, on="relation", how="left")
    return out.sort_values("ig_sum", ascending=False).reset_index(drop=True)


def edge_col_of(ev: pd.DataFrame) -> str:
    """Name of the column holding the (canonical) edge type."""
    return "fwd_edge_type" if "fwd_edge_type" in ev.columns else "edge_type"


def top_edge_rows(ev: pd.DataFrame, top_edges: int) -> pd.DataFrame:
    """Restrict ``ev`` to the rows of the ``top_edges`` highest-|IG| edges.

    Shared edge-ranking used by both the Level-2 drill-down and the JSON export
    so they agree on which edges count as "significant". Returns the evidence
    rows (not de-duplicated) with a numeric ``ig_total`` column.
    """
    edge_col = edge_col_of(ev)
    e = ev.copy()
    e["ig_total"] = pd.to_numeric(e.get("ig_total"), errors="coerce")
    edge_rank = (e.drop_duplicates([edge_col, "src", "dst"])
                  .assign(_abs=lambda d: d["ig_total"].abs())
                  .sort_values("_abs", ascending=False)
                  .head(top_edges)[[edge_col, "src", "dst"]])
    return e.merge(edge_rank, on=[edge_col, "src", "dst"], how="inner")


def build_edge_evidence(ev: pd.DataFrame, top_edges: int) -> pd.DataFrame:
    """LEVEL 2 — per individual edge (the figure's actual edges), the
    individual OT evidence rows that back it.

    One output row per (edge, evidence id): carries the entity accessions
    (``sourceId``/``targetId``), datasource, year, OT score, and the evidence
    ``id`` hash — the citable handle you reference a figure edge to. Restricted
    to the ``top_edges`` highest-|IG| edges so it stays referenceable.
    """
    edge_col = edge_col_of(ev)
    keep = top_edge_rows(ev, top_edges)
    cols = [edge_col, "src", "dst", "ig_total", "attention",
            "sourceId", "targetId", "datasourceId", "year"]
    if "score" in keep.columns:
        cols.append("score")
    if "id" in keep.columns:
        cols.append("id")
    if "literature" in keep.columns:
        cols.append("literature")
    cols = [c for c in cols if c in keep.columns]
    keep = keep[cols].copy()
    keep["relation"] = keep[edge_col].map(_fmt_relation)

    # Collapse to ONE row per distinct OT evidence id (the raw parquet has a
    # row per id×year and repeats ids); show the id's year span + score rollup
    # so a 136-row clinical_precedence edge becomes a handful of citable ids.
    group_keys = [edge_col, "src", "dst", "relation", "sourceId", "targetId",
                  "datasourceId"]
    group_keys = [k for k in group_keys if k in keep.columns]
    id_key = "id" if "id" in keep.columns else None
    if id_key:
        group_keys = group_keys + [id_key]

    def _collapse(g: pd.DataFrame) -> pd.Series:
        rec = {
            "ig_total": g["ig_total"].iloc[0] if "ig_total" in g else np.nan,
            "attention": g["attention"].iloc[0] if "attention" in g else np.nan,
            "year": _year_span(g.get("year", pd.Series(dtype=float))),
            "n_rows": len(g),
        }
        if "score" in g.columns:
            sc = pd.to_numeric(g["score"], errors="coerce").dropna()
            rec["score"] = float(sc.max()) if not sc.empty else np.nan
        if "literature" in g.columns:
            rec["pmids"] = _pmids(g["literature"])
        return pd.Series(rec)

    collapsed = keep.groupby(group_keys, dropna=False).apply(
        _collapse, include_groups=False).reset_index()
    collapsed["_abs"] = collapsed["ig_total"].abs().fillna(-1)
    return collapsed.sort_values(["_abs", "datasourceId", "year"],
                                 ascending=[False, True, True]).drop(columns="_abs")


def render_markdown(rel_summary: pd.DataFrame, edge_detail: pd.DataFrame,
                    target_id: str, disease_id: str, detail_edges: int) -> str:
    """Render the two-level brief: relation-type legend, then per-edge evidence
    drill-down (so a figure edge can be matched to its OT evidence ids)."""
    lines: list[str] = []
    lines.append(f"# Advancement explanation: {target_id} -> {disease_id}")
    lines.append("")
    if rel_summary.empty:
        lines.append("_No backing evidence rows found for this pair._")
        return "\n".join(lines)

    has_score = "score" in edge_detail.columns

    # ---- Level 1: relation-type legend ------------------------------------
    lines.append("## Drivers by relation type")
    lines.append("")
    lines.append("Signed Integrated-Gradients attribution summed over every "
                 "edge of each relation in the subgraph (strongest positive "
                 "first). This is the legend for the subgraph figure.")
    lines.append("")
    lines.append("| Relation | IG sum | IG max | #edges | #evidence | Datasources |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, r in rel_summary.iterrows():
        ds = ", ".join(r["datasources"]) if isinstance(r["datasources"], list) else ""
        lines.append(
            f"| {r['relation']} | {r['ig_sum']:+.4f} | {r['ig_max']:+.4f} "
            f"| {int(r['n_edges'])} | {int(r['n_evidence'])} | {ds} |")
    lines.append("")

    # ---- Level 2: per-edge evidence drill-down ----------------------------
    lines.append(f"## Evidence behind the top {detail_edges} edges")
    lines.append("")
    lines.append("Each individual edge in the subgraph, with the OpenTargets "
                 "evidence rows that back it. The evidence `id` links to the OT "
                 "platform record; **PMIDs** link to the cited papers (present "
                 "for literature/intact/GO datasources). Use (relation, "
                 "sourceId→targetId) to match an edge in the figure.")
    lines.append("")
    edge_col = "fwd_edge_type" if "fwd_edge_type" in edge_detail.columns else "edge_type"
    has_pmids = "pmids" in edge_detail.columns
    edge_keys = edge_detail.drop_duplicates([edge_col, "src", "dst"])
    shown = 0
    for _, ek in edge_keys.iterrows():
        if shown >= detail_edges:
            break
        shown += 1
        sub = edge_detail[(edge_detail[edge_col] == ek[edge_col])
                          & (edge_detail["src"] == ek["src"])
                          & (edge_detail["dst"] == ek["dst"])]
        ig = ek["ig_total"]
        ig_s = f"{ig:+.4f}" if pd.notna(ig) else "n/a"
        src_acc = sub["sourceId"].iloc[0] if "sourceId" in sub else ek["src"]
        dst_acc = sub["targetId"].iloc[0] if "targetId" in sub else ek["dst"]
        n_ev = sub["id"].nunique() if "id" in sub.columns else len(sub)
        lines.append(f"### {shown}. {ek['relation']}  (IG {ig_s})")
        lines.append(f"- Edge: `{src_acc}` → `{dst_acc}`  "
                     f"({n_ev} distinct evidence row(s))")
        hdr = ("    | datasource | years |" + (" score |" if has_score else "")
               + " evidence id |" + (" PMIDs |" if has_pmids else ""))
        sep = ("    |---|---:|" + ("---:|" if has_score else "") + "---|"
               + ("---|" if has_pmids else ""))
        lines.append(hdr)
        lines.append(sep)
        for _, r in sub.iterrows():
            yr = r.get("year") if pd.notna(r.get("year")) else ""
            row = f"    | {r['datasourceId']} | {yr} |"
            if has_score:
                sc = r.get("score")
                row += f" {sc:.2f} |" if pd.notna(sc) else " — |"
            evid = r.get("id") if "id" in sub.columns and pd.notna(r.get("id")) else None
            row += (f" [`{evid}`]({OT_EVIDENCE_URL}{evid}) |" if evid else " — |")
            if has_pmids:
                pmids = r.get("pmids")
                pmids = list(pmids) if isinstance(pmids, (list, tuple, np.ndarray)) else []
                if pmids:
                    links = ", ".join(f"[{p}]({PUBMED_URL}{p})" for p in pmids[:8])
                    if len(pmids) > 8:
                        links += f" +{len(pmids) - 8}"
                else:
                    links = "—"
                row += f" {links} |"
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def _attach_ig(ev: pd.DataFrame, edges_parquet: str) -> pd.DataFrame:
    """Join per-edge ``ig_total``/``attention`` onto evidence rows.

    ``explain_advancement.py``'s ``per_pair_evidence.parquet`` carries the edge
    identity (``edge_type``, ``src``, ``dst``) but not the attribution; that
    lives in the sibling ``per_pair_edges.parquet``. Both index in the same
    global node space, so a left-join on (target_id, disease_id, edge_type,
    src, dst) recovers IG/attention without re-running the model.
    """
    if {"ig_total", "attention"}.issubset(ev.columns) and ev["ig_total"].notna().any():
        return ev
    ed = pd.read_parquet(edges_parquet,
                         columns=["target_id", "disease_id", "edge_type",
                                  "src", "dst", "ig_total", "attention"])
    key = ["target_id", "disease_id", "edge_type", "src", "dst"]
    ed = ed.drop_duplicates(key)
    drop = [c for c in ("ig_total", "attention") if c in ev.columns]
    return ev.drop(columns=drop).merge(ed, on=key, how="left")


def main(args: argparse.Namespace) -> None:
    ev = pd.read_parquet(args.evidence_parquet)
    if args.edges_parquet:
        ev = _attach_ig(ev, args.edges_parquet)
    if args.target_id and args.disease_id:
        ev = ev[(ev["target_id"] == args.target_id)
                & (ev["disease_id"] == args.disease_id)].copy()
    if ev.empty:
        raise SystemExit(f"[present] no evidence rows in "
                         f"{args.evidence_parquet} for the requested filter")

    # If the file holds many pairs and none was requested, present each.
    pairs = ev[["target_id", "disease_id"]].drop_duplicates()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[present] {len(pairs)} pair(s) to present", flush=True)

    for _, pr in pairs.iterrows():
        t, d = pr["target_id"], pr["disease_id"]
        sub = ev[(ev["target_id"] == t) & (ev["disease_id"] == d)]
        stem = f"{t}__{d}".replace("/", "_")

        summary = build_summary(sub)                       # edge×datasource
        rel_summary = build_relation_summary(summary)      # Level 1
        edge_detail = build_edge_evidence(sub, args.top_edges)  # Level 2

        rel_summary.to_parquet(
            os.path.join(args.out_dir, f"{stem}_relation_summary.parquet"),
            index=False)
        edge_detail.to_parquet(
            os.path.join(args.out_dir, f"{stem}_edge_evidence.parquet"),
            index=False)

        md = render_markdown(rel_summary, edge_detail, t, d,
                             detail_edges=args.top_edges)
        with open(os.path.join(args.out_dir, f"{stem}.md"), "w") as fh:
            fh.write(md)
        n_rel = len(rel_summary)
        print(f"[present] {t} -> {d}: {n_rel} relation types, "
              f"{edge_detail.drop_duplicates(['src','dst']).shape[0]} detailed "
              f"edges -> {stem}.md", flush=True)

    print(f"[present] done -> {args.out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a readable evidence brief for explained pairs.")
    p.add_argument("--evidence-parquet", required=True,
                   help="per_pair_evidence.parquet (explain_advancement "
                        "--raw-edges-dir) or join_pair_evidence.py output.")
    p.add_argument("--edges-parquet", default=None,
                   help="Sibling per_pair_edges.parquet. When the evidence "
                        "file lacks ig_total/attention (explainer-written "
                        "per_pair_evidence.parquet), join them in from here.")
    p.add_argument("--out-dir", required=True,
                   help="Directory for <pair>.md + <pair>_summary.parquet.")
    p.add_argument("--target-id", default=None,
                   help="Optional: present only this pair.")
    p.add_argument("--disease-id", default=None)
    p.add_argument("--top-edges", type=int, default=15,
                   help="How many top-|IG| individual edges to detail in the "
                        "Level-2 evidence drill-down (default 15). The Level-1 "
                        "relation legend always covers all relations.")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

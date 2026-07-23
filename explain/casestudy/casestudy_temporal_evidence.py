"""Temporal evidence profile for the explainability case studies.

For a (target, disease) pair, extract every direct evidence record from the
dated Open Targets datasource parquets (evidenceDated/sourceId=*), plus the
INDIRECT (on-path) evidence: every intermediate (target, disease) edge the
model's fused explanation paths route through, excluding the direct query
pair. This is the full evidence context contained in the explanation path, so
it includes cross-disease detours (e.g. a bridge target's animal_model or
genetic edge to an intermediate disease), not just neighbour targets that
share the query disease. Build a cumulative max-score-vs-year line plot per
datasource, direct vs indirect, up to the decision year and on to present.

Run on a compute node (memory): srun --jobid=<J> .venv/bin/python scripts/casestudy_temporal_evidence.py
"""
import glob, json, os
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVDIR = "/gpfs/scratch/bty414/opentarget_evidences/26.03/evidenceDated"
OUT = "/data/home/bty414/opentarget_temporal_study/src/opentarget_het_graph/figures/results"
os.makedirs(OUT, exist_ok=True)

# Auto-discover every datasource partition on disk rather than hardcoding a
# subset (avoids silently dropping a datasource). These sourceId names are the
# raw Open Targets datasources; they roll up into the graph's datatype-level
# schema relations (e.g. eva/gwas/clingen/... -> genetic_association;
# impc -> animal_model; europepmc -> literature; clinical_precedence ->
# clinical_trial_* and modulated_by). See casestudy doc §"Evidence vs path".
SOURCES = sorted(
    os.path.basename(p).replace("sourceId=", "")
    for p in glob.glob(f"{EVDIR}/sourceId=*")
    if os.path.isdir(p)
)

GENE = {"ENSG00000078814": "MYH7B", "ENSG00000092054": "MYH7",
        "ENSG00000111245": "MYL2", "ENSG00000160808": "MYL3",
        "ENSG00000197616": "MYH6", "ENSG00000198336": "MYL4",
        "ENSG00000106631": "MYL7",
        "ENSG00000198851": "CD3E", "ENSG00000160654": "CD3G",
        "ENSG00000167286": "CD3D"}

# Each case study is a query pair; the INDIRECT neighbour targets are derived
# automatically from the model's fused explanation paths (indirect_targets_from_paths
# below), NOT hand-picked — so the direct/indirect temporal figure is fully
# reproducible and exactly matches the intermediates the explanation routes through.
CASES = {
    "CD274_meso": dict(
        target="ENSG00000120217", tname="CD274 (PD-L1)",
        disease="EFO_0000588", dname="mesothelioma", decision=2016,
    ),
    "IL17F_psa": dict(
        target="ENSG00000112116", tname="IL17F",
        disease="EFO_0003778", dname="psoriatic arthritis", decision=2016,
    ),
    "TIGIT_nsclc": dict(
        target="ENSG00000181847", tname="TIGIT",
        disease="EFO_0003060", dname="non-small cell lung carcinoma", decision=2018,
    ),
}

# Fused explanation paths (total-cost ranked) + the node mapping, used to derive
# the indirect bridge targets per pair.
PATHS_JSON = "/gpfs/scratch/bty414/biobridge_paths_totalcost.json"
MAPPINGS = ("/gpfs/scratch/bty414/opentarget_evidences/26.03/progression/"
            "temporal_graph_datatype_mappings.pt")


def indirect_pairs_from_paths(target, disease):
    """Every (target_ENSG, disease_EFO) pair that appears on a target--disease
    edge of the fused explanation paths for the query (target, disease),
    EXCLUDING the direct query pair itself.

    "Indirect" here means the evidence *contained in the explanation path* — all
    the intermediate target--disease edges the path routes through, including
    detours through OTHER diseases (e.g. a bridge target's mouse-model or
    genetic edge to a non-query disease). This is broader than "bridge targets
    that share the query disease": it is the full on-path evidence context, so a
    path edge like an animal_model edge to an intermediate disease is included
    rather than filtered out. Returns a list of (ENSG, EFO) tuples.
    """
    import json, torch
    try:
        P = json.load(open(PATHS_JSON))
    except FileNotFoundError:
        return []
    key = f"{target}|{disease}"
    if key not in P:
        return []
    nm = torch.load(MAPPINGS, weights_only=False)["node_mapping"]
    i2t = {v: k for k, v in dict(nm["target"]).items()}
    i2d = {v: k for k, v in dict(nm["disease"]).items()}
    pairs = []
    for p in P[key]:
        for e in p["E"]:
            # e = [edge_type_str, src_tok, dst_tok, meta]; resolve each edge to a
            # forward (target, disease) pair regardless of traversal direction.
            src_tok, dst_tok = e[1], e[2]
            toks = {tok.split("#")[0]: int(tok.split("#")[1])
                    for tok in (src_tok, dst_tok)}
            if "target" in toks and "disease" in toks:
                acc_t = i2t.get(toks["target"])
                acc_d = i2d.get(toks["disease"])
                if acc_t and acc_d and (acc_t, acc_d) != (target, disease):
                    if (acc_t, acc_d) not in pairs:
                        pairs.append((acc_t, acc_d))
    return pairs


def load_source(src, targets):
    """All dated evidence rows for the given targetIds from one datasource."""
    files = glob.glob(f"{EVDIR}/sourceId={src}/**/*.parquet", recursive=True)
    frames = []
    for f in files:
        names = pq.ParquetFile(f).schema_arrow.names
        cols = [c for c in ("targetId", "diseaseId", "score", "evidenceDate",
                            "clinicalReportId", "literature", "studyId",
                            "clinicalStage", "drugFromSource")
                if c in names]
        try:
            df = pq.read_table(f, columns=cols).to_pandas()
        except Exception:
            continue
        df = df[df.targetId.isin(targets)]
        if len(df):
            df["src"] = src
            frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def _finalize(df):
    """Common date/score coercion for an evidence frame."""
    if not len(df):
        return pd.DataFrame()
    df = df.copy()
    df["year"] = pd.to_datetime(df.evidenceDate, errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["score"] = pd.to_numeric(df.score, errors="coerce").fillna(0.0)
    return df


def evidence_for(targets, disease):
    """Dated evidence for the given targets against a SINGLE disease (used for
    the direct query pair)."""
    frames = [load_source(s, targets) for s in SOURCES]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df.diseaseId == disease].copy()
    return _finalize(df)


def evidence_for_pairs(pairs):
    """Dated evidence for an explicit set of (targetId, diseaseId) pairs — the
    on-path indirect evidence. Each edge on the explanation path contributes its
    own (target, disease) pair, so a bridge target's evidence is only counted
    against the disease it is actually linked to on the path (which may differ
    from the query disease). This is the full on-path evidence context."""
    if not pairs:
        return pd.DataFrame()
    targets = sorted({t for t, _ in pairs})
    pairset = set(pairs)
    frames = [load_source(s, targets) for s in SOURCES]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keep = [(t, d) in pairset for t, d in zip(df.targetId, df.diseaseId)]
    df = df[keep].copy()
    return _finalize(df)


def cumulative_max(df, years):
    """Cumulative max score up to each year (over the whole df)."""
    out = []
    for y in years:
        s = df[df.year <= y].score
        out.append(float(s.max()) if len(s) else 0.0)
    return out


def cumulative_max_by_source(df, years):
    """{source: [cumulative max score per year]} — one curve per datasource."""
    curves = {}
    for src in sorted(df.src.unique()):
        sub = df[df.src == src]
        curves[src] = [float(sub[sub.year <= y].score.max()) if (sub.year <= y).any() else 0.0
                       for y in years]
    return curves


# Ground the datasource -> schema-relation mapping in the repo edge schema
# (config/edge_schema_26.03.yaml), so a datasource is never mislabelled.
def _load_src_to_relation():
    import yaml
    schema_path = os.path.join(os.path.dirname(__file__), "..", "..", "config",
                               "edge_schema_26.03.yaml")
    with open(schema_path) as fh:
        schema = yaml.safe_load(fh)
    out = {}
    for src, spec in schema.items():
        specs = spec if isinstance(spec, list) else [spec]
        for s in specs:
            rel = s.get("relation_name")
            if rel and rel != "modulated_by":     # the disease-facing relation
                out[src] = rel
    # clinical_precedence's target->disease relation is bucketed (trial_outcome_bucket)
    out["clinical_precedence"] = "clinical_trial"
    return out

SRC_TO_RELATION = _load_src_to_relation()
# datasource legend label = "datasource (schema relation)"
def src_label(src):
    rel = SRC_TO_RELATION.get(src, "?")
    return f"{src} ({rel})"
SRC_LABEL = {s: src_label(s) for s in SRC_TO_RELATION}


PRESENT = 2025

def build(case_key, case):
    dec = case["decision"]
    # Indirect evidence = the full on-path evidence context: every intermediate
    # (target, disease) edge the explanation paths route through (excluding the
    # direct query pair). DERIVED from the fused paths (not hand-picked), so the
    # figure is fully reproducible and includes cross-disease detours (e.g. a
    # bridge target's animal_model edge to an intermediate disease).
    indirect_pairs = indirect_pairs_from_paths(case["target"], case["disease"])
    print(f"  {case_key}: derived on-path indirect (target,disease) pairs = "
          f"{indirect_pairs}", flush=True)
    direct = evidence_for([case["target"]], case["disease"])
    indirect = evidence_for_pairs(indirect_pairs)
    yrs = list(range(2000, PRESENT + 1))          # extend to present, not just decision
    dsrc = cumulative_max_by_source(direct, yrs)
    isrc = cumulative_max_by_source(indirect, yrs)

    # two-panel: direct (top) and indirect (bottom), one line per datasource.
    # Legend OUTSIDE each panel (right), styling matched to evaluate_advancement.py
    # (dpi 300, tight bbox, white facecolor, 8-9pt fonts).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor("white")
    cmap = plt.get_cmap("tab10")
    # one stable colour per datasource across BOTH panels, so a single shared
    # legend is unambiguous.
    all_srcs = sorted(set(dsrc) | set(isrc))
    src_color = {s: cmap(i % 10) for i, s in enumerate(all_srcs)}
    handles = {}
    for ax, curves, title in [(ax1, dsrc, f"DIRECT: {case['tname']} – {case['dname']}"),
                              (ax2, isrc, "INDIRECT: on-path intermediate target--disease edges")]:
        ax.axvspan(dec, PRESENT, color="0.92", zorder=0)
        ax.axvline(dec, ls="--", color="k", lw=1.5)
        for src, cur in sorted(curves.items(), key=lambda kv: -max(kv[1])):
            if max(cur) == 0:
                continue
            (line,) = ax.plot(yrs, cur, "-o", ms=2.5, lw=1.4, color=src_color[src],
                              label=SRC_LABEL.get(src, src))
            handles.setdefault(src, line)
        ax.set_ylabel("max evidence score (cumulative)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax1.text(dec - 0.3, 1.03, f"decision {dec}", rotation=90, va="top", ha="right",
             fontsize=9, fontweight="bold")
    ax2.set_xlabel("year", fontsize=9)
    ax2.set_xlim(2000, PRESENT)
    # single shared legend outside, on the right, ordered by datasource name
    ordered = sorted(handles.items())
    fig.legend([h for _, h in ordered], [SRC_LABEL.get(s, s) for s, _ in ordered],
               fontsize=8, loc="center left", bbox_to_anchor=(0.99, 0.5),
               frameon=False, title="datasource (schema relation)", title_fontsize=8)
    fig.suptitle(f"{case['tname']} → {case['dname']}: evidence by datasource over time",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = f"{OUT}/casestudy_temporal_{case_key}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # provenance summary: pre-decision (visible) vs post-decision (gained later)
    def stats(df):
        pre = df[df.year < dec]; post = df[df.year >= dec]
        return {
            "n_pre": int(len(pre)), "n_post": int(len(post)),
            "max_score_pre": round(float(pre.score.max()) if len(pre) else 0.0, 3),
            "max_score_now": round(float(df.score.max()) if len(df) else 0.0, 3),
            "pre_by_source": pre.groupby("src").size().to_dict(),
        }
    summary = {
        "case": case_key, "target": case["tname"], "disease": case["dname"],
        "decision": dec, "present": PRESENT,
        "direct": stats(direct),
        "indirect": stats(indirect),
        "plot": p,
    }
    # a few concrete visible records (NCT / PMID) for the write-up
    def sample_records(df):
        recs = []
        for _, r in df.sort_values("score", ascending=False).head(8).iterrows():
            recs.append({"year": int(r.year), "src": r.src, "score": round(float(r.score), 3),
                         "nct": r.get("clinicalReportId"), "pmid": r.get("literature"),
                         "stage": r.get("clinicalStage")})
        return recs
    summary["direct_top_records_pre"] = sample_records(direct[direct.year < dec])
    summary["indirect_top_records_pre"] = sample_records(indirect[indirect.year < dec])
    # record the on-path (target,disease) pairs and the intermediate diseases
    # they cover, so the write-up can name any cross-disease detour explicitly.
    summary["indirect_pairs"] = [list(p) for p in indirect_pairs]
    summary["indirect_diseases"] = sorted(indirect.diseaseId.unique().tolist()) if len(indirect) else []
    return summary


if __name__ == "__main__":
    allsum = {}
    for k, c in CASES.items():
        print(f"building {k} ...", flush=True)
        allsum[k] = build(k, c)
    with open(f"{OUT}/casestudy_temporal_summary.json", "w") as fh:
        json.dump(allsum, fh, indent=2, default=str)
    print(json.dumps(allsum, indent=2, default=str))

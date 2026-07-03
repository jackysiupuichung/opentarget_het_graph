"""Temporal evidence profile for the explainability case studies.

For a (target, disease) pair, extract every direct evidence record from the
dated Open Targets datasource parquets (evidenceDated/sourceId=*), plus the
indirect evidence reachable via named neighbour targets (e.g. the other genes
in the same multi-target trial / protein complex). Build a cumulative
max-score-vs-year line plot up to the decision year, direct vs indirect.

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

# Each case study: query pair + the indirect neighbour targets (verified) that
# carry evidence for the same disease (shared trial / complex partners).
CASES = {
    "HCM": dict(
        target="ENSG00000078814", tname="MYH7B",
        disease="EFO_0000538", dname="hypertrophic cardiomyopathy",
        decision=2016,
        # other sarcomere myosins co-registered in the mavacamten HCM programme
        indirect=["ENSG00000092054", "ENSG00000111245", "ENSG00000160808",
                  "ENSG00000197616", "ENSG00000198336", "ENSG00000106631"],
    ),
}
# (NHL/CD3 case study dropped 2026-07-03 — the paper uses MYH7B->HCM only.)


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


def evidence_for(targets, disease):
    frames = [load_source(s, targets) for s in SOURCES]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df.diseaseId == disease].copy()
    df["year"] = pd.to_datetime(df.evidenceDate, errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["score"] = pd.to_numeric(df.score, errors="coerce").fillna(0.0)
    return df


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
    schema_path = os.path.join(os.path.dirname(__file__), "..", "config",
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
    direct = evidence_for([case["target"]], case["disease"])
    indirect = evidence_for(case["indirect"], case["disease"])
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
                              (ax2, isrc, "INDIRECT: neighbour targets (shared trial / complex), same disease")]:
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
    return summary


if __name__ == "__main__":
    allsum = {}
    for k, c in CASES.items():
        print(f"building {k} ...", flush=True)
        allsum[k] = build(k, c)
    with open(f"{OUT}/casestudy_temporal_summary.json", "w") as fh:
        json.dump(allsum, fh, indent=2, default=str)
    print(json.dumps(allsum, indent=2, default=str))

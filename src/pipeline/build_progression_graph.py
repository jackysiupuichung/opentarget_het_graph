#!/usr/bin/env python

__author__ = "Cote Falaguera (mjfalagueramata@gmail.com)"
__date__ = "02 Jul 2025"

"""
timeseries.py: Assess the evolution over time of evidence supporting target-disease associations in the Open Targets Platform.

Useful GitHub links:
- https://github.com/opentargets/timeseries
- https://github.com/opentargets/issues/issues/2739
"""


import datetime
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from src.parsers.chembl_trial_expander import expand_chembl_clinical_trials



# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
EDGE_DIR = "/data/scratch/bty414/opentarget_evidences/23.06/kg_output/edges"
STATIC_EDGE_DIR = "/data/scratch/bty414/opentarget_evidences/23.06/kg_output/static_edges"
OUT_DIR = "/data/scratch/bty414/opentarget_evidences/23.06/progression_graph"
# EDGE_DIR = "/Users/pchungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/kg_output/edges"
# STATIC_EDGE_DIR = "/Users/pchungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/kg_output/static_edges"
# OUT_DIR = "/Users/pchungsiu/Documents/opentarget_het_graph/data/evidenceDated_subset/23.06/kg_output/progression_graph"
DATASOURCE_HARMONIC_FILE = f"{OUT_DIR}/datasource_harmonic.parquet"
DATATYPE_HARMONIC_FILE = f"{OUT_DIR}/datatype_harmonic.parquet"
STATIC_SUPP_FILE = f"{OUT_DIR}/static_edges.parquet"
os.makedirs(OUT_DIR, exist_ok=True)

FIRST_YEAR = 2000
LAST_YEAR = 2025
YEARS = np.arange(FIRST_YEAR, LAST_YEAR + 1)
MAX_HARMONIC = 1.644  # theoretical max sum of 1/i^2

# novelty settings
NOVELTY_SCALE = 2     # logistic steepness
NOVELTY_SHIFT = 2     # midpoint
NOVELTY_WINDOW = 10   # years after peak to decay

data_sources = [
    {
        "id": "gwas_credible_sets",
        "sectionId": "gwasCredibleSets",
        "label": "GWAS associations",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,  # needs to be a float
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gwas-associations",
    },
    {
        "id": "eva",
        "sectionId": "eva",
        "label": "ClinVar",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clinvar",
    },
    {
        "id": "gene_burden",
        "sectionId": "geneBurden",
        "label": "Gene Burden",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene-burden",
    },
    {
        "id": "genomics_england",
        "sectionId": "genomicsEngland",
        "label": "GEL PanelApp",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#genomics-england-panelapp",
    },
    {
        "id": "gene2phenotype",
        "sectionId": "gene2Phenotype",
        "label": "Gene2phenotype",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene2phenotype",
    },
    {
        "id": "uniprot_literature",
        "sectionId": "uniprotLiterature",
        "label": "UniProt literature",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#uniprot-literature",
    },
    {
        "id": "uniprot_variants",
        "sectionId": "uniprotVariants",
        "label": "UniProt curated variants",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#uniprot-variants",
    },
    {
        "id": "orphanet",
        "sectionId": "orphanet",
        "label": "Orphanet",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#orphanet",
    },
    {
        "id": "clingen",
        "sectionId": "clinGen",
        "label": "Clingen",
        "aggregation": "Genetic association",
        "aggregationId": "genetic_association",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clingen",
    },
    {
        "id": "cancer_gene_census",
        "sectionId": "cancerGeneCensus",
        "label": "Cancer Gene Census",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#cancer-gene-census",
    },
    {
        "id": "intogen",
        "sectionId": "intOgen",
        "label": "IntOGen",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#intogen",
    },
    {
        "id": "eva_somatic",
        "sectionId": "evaSomatic",
        "label": "ClinVar (somatic)",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#clinvar-somatic",
    },
    {
        "id": "cancer_biomarkers",
        "sectionId": "cancerBiomarkers",
        "label": "Cancer Biomarkers",
        "aggregation": "Somatic mutations",
        "aggregationId": "somatic_mutation",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#cancer-biomarkers",
    },
    {
        "id": "chembl",
        "sectionId": "chembl",
        "label": "ChEMBL",
        "aggregation": "Known drug",
        "aggregationId": "known_drug",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#chembl",
    },
    {
        "id": "crispr_screen",
        "sectionId": "crispr_screen",
        "label": "CRISPR Screens",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#project-score",
    },
    {
        "id": "crispr",
        "sectionId": "crispr",
        "label": "Project Score",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#project-score",
    },
    {
        "id": "slapenrich",
        "sectionId": "slapEnrich",
        "label": "SLAPenrich",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#slapenrich",
    },
    {
        "id": "progeny",
        "sectionId": "progeny",
        "label": "PROGENy",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#slapenrich",
    },
    {
        "id": "reactome",
        "sectionId": "reactome",
        "label": "Reactome",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 1.0,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#reactome",
    },
    {
        "id": "sysbio",
        "sectionId": "sysBio",
        "label": "Gene signatures",
        "aggregation": "Affected pathway",
        "aggregationId": "affected_pathway",
        "weight": 0.5,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#gene-signatures",
    },
    {
        "id": "europepmc",
        "sectionId": "europePmc",
        "label": "Europe PMC",
        "aggregation": "Literature",
        "aggregationId": "literature",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#europe-pmc",
    },
    {
        "id": "expression_atlas",
        "sectionId": "expression",
        "label": "Expression Atlas",
        "aggregation": "RNA expression",
        "aggregationId": "rna_expression",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#expression-atlas",
    },
    {
        "id": "impc",
        "sectionId": "impc",
        "label": "IMPC",
        "aggregation": "Animal model",
        "aggregationId": "animal_model",
        "weight": 0.2,
        "isPrivate": False,
        "docsLink": "https://platform-docs.opentargets.org/evidence#impc",
    },
    # {
    #     "id": "ot_crispr",
    #     "sectionId": "otCrispr",
    #     "label": "OT CRISPR",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
    # {
    #     "id": "encore",
    #     "sectionId": "encore",
    #     "label": "ENCORE",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
    # {
    #     "id": "ot_crispr_validation",
    #     "sectionId": "validationlab",
    #     "label": "OT Validation",
    #     "aggregation": "Partner-only",
    #     "aggregationId": "partner_only",
    #     "weight": 0.5,
    #     "isPrivate": True,
    #     "docsLink": "https://partner-platform.opentargets.org/projects",
    # },
]

DATA_SOURCES = {
    ds["id"]: {
        "datatype": ds["aggregationId"],
        "weight": float(ds["weight"]),
    }
    for ds in data_sources
}


# ----------------------------------------------------
# 1. LOAD DYNAMIC + STATIC EVIDENCE
# ----------------------------------------------------
def load_dynamic_evidence():
    dfs = []
    for fname in os.listdir(EDGE_DIR):
        if fname.startswith("sourceId=") and fname.endswith(".parquet"):
            df = pd.read_parquet(f"{EDGE_DIR}/{fname}")
            df["year"] = df["year"].astype(int)
            
            # only expand ChEMBL clinical trials
            datasource = df.iloc[0]['datasourceId']
            if datasource == "chembl":
                df = expand_chembl_clinical_trials(df)

            dfs.append(df[["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId", "score", "year"]])

    print(f"Loaded {len(dfs)} data sources")
    return pd.concat(dfs, ignore_index=True)

def load_static_evidence():
    dfs = []
    for fname in os.listdir(STATIC_EDGE_DIR):
        if fname.endswith(".parquet"):
            df = pd.read_parquet(f"{STATIC_EDGE_DIR}/{fname}").copy()
            df["relation_key"] = df["datasourceId"] + "::" + df["relation"]
            keep = ["sourceId", "targetId", "source_type", "target_type",
                    "relation", "datasourceId", "score", "year", "relation_key"]
            dfs.append(df[keep])

    print(f"Loaded {len(dfs)} static sources")
    return pd.concat(dfs, ignore_index=True)


# ----------------------------------------------------
# 1.2. SANITY CHECK: UNIQUE NODES + UNIQUE EDGES
# ----------------------------------------------------
def inspect_graph(evd):
    print("\n================ EVIDENCE SUMMARY ================\n")

    # ---------------------------------------------------------
    # 1. COLLECT ALL NODES BY TYPE (DYNAMIC)
    # ---------------------------------------------------------
    node_type_map = {}

    # From source side
    for node, t in evd[["sourceId", "source_type"]].drop_duplicates().itertuples(index=False):
        node_type_map.setdefault(t, set()).add(node)

    # From target side
    for node, t in evd[["targetId", "target_type"]].drop_duplicates().itertuples(index=False):
        node_type_map.setdefault(t, set()).add(node)

    # ---------------------------------------------------------
    # 2. PRINT SUMMARY OF EACH NODE TYPE
    # ---------------------------------------------------------
    total_unique_nodes = set(evd["sourceId"]) | set(evd["targetId"])

    for t, nodes in node_type_map.items():
        print(f"🟦 Node type '{t}' : {len(nodes)}")

    # Nodes that appear but have no explicit type (rare)
    typed_nodes = set().union(*node_type_map.values()) if node_type_map else set()
    untyped_nodes = total_unique_nodes - typed_nodes

    if untyped_nodes:
        print(f"⚠️ Untyped nodes found : {len(untyped_nodes)}")

    print(f"\n🌐 Total unique nodes : {len(total_unique_nodes)}")

    # ---------------------------------------------------------
    # 3. UNIQUE EDGE COUNT
    # ---------------------------------------------------------
    unique_edges = set(
        tuple(row)
        for row in evd[["sourceId", "relation", "targetId"]]
        .itertuples(index=False, name=None)
    )
    print(f"🔗 Total unique edges : {len(unique_edges)}")

    # ---------------------------------------------------------
    # 4. RELATION STATISTICS
    # ---------------------------------------------------------
    print("\n📚 Edge counts per relation:")
    print(evd["relation"].value_counts())

    # ---------------------------------------------------------
    # 5. DATASOURCE STATISTICS
    # ---------------------------------------------------------
    print("\n📦 Edge counts per datasource:")
    print(evd["datasourceId"].value_counts())

    # ---------------------------------------------------------
    # 6. YEAR RANGE
    # ---------------------------------------------------------
    print("\n📆 Year range:")
    print(f"Min year = {evd['year'].min()}, Max year = {evd['year'].max()}")



# ----------------------------------------------------
# UTILITY: harmonic sum of top-50 scores
# these are based on the implementation in https://github.com/opentargets/timeseries/blob/main/timeseries.py#L449
# ----------------------------------------------------
def harmonic_sum(scores):
    if len(scores) == 0:
        return 0.0

    s = np.sort(scores)[::-1][:50]  # top 50 descending
    idx = np.arange(1, len(s) + 1)
    return np.sum(s / (idx ** 2)) / MAX_HARMONIC

def _compute_novelty(group, score_col):
    years = group["year"].values
    scores = group[score_col].values

    diffs = np.diff(scores, prepend=0)
    peak_years = years[diffs > 0]
    peaks = diffs[diffs > 0]

    novelty_map = {}

    for py, pv in zip(peak_years, peaks):
        for t in range(py, py + NOVELTY_WINDOW + 1):
            nv = pv / (1 + np.exp(NOVELTY_SCALE * (t - py - NOVELTY_SHIFT)))
            novelty_map[t] = max(nv, novelty_map.get(t, 0))

    result = []
    for _, row in group.iterrows():
        y = row["year"]
        result.append(list(row.values) + [novelty_map.get(y, 0.0)])

    return result


# ----------------------------------------------------
# 2. DATASOURCE-LEVEL HARMONIC SCORE
# ----------------------------------------------------
def harmonic_by_datasource(evd):
    rows = []
    grouped = evd.groupby(["sourceId", "targetId", "source_type", "target_type", "relation", "datasourceId"])

    for (src, tgt, src_type, tgt_type, rel, ds), group in tqdm(grouped, desc="Datasource harmonic"):
        year_dict = group.groupby("year")["score"].apply(list).to_dict()
        collected = []
        for y in YEARS:
            if y in year_dict:
                collected.extend(year_dict[y])
            hs = harmonic_sum(collected)
            rows.append([src, tgt, src_type, tgt_type, rel, ds, y, hs])

    df = pd.DataFrame(rows, columns=[
        "sourceId", "targetId", "source_type", "target_type",
        "relation", "datasourceId", "year", "datasource_score"
    ])
    df.rename(columns={"datasource_score": "score"}, inplace=True)
    return df




# # ----------------------------------------------------
# # 3. DATASOURCE-LEVEL NOVELTY
# # ----------------------------------------------------
# def novelty_by_datasource(df):
#     rows = []
#     grouped = df.groupby(["sourceId", "targetId", "source_type",
#                           "target_type", "relation", "datasourceId"])

#     for _, group in tqdm(grouped, desc="Datasource novelty"):
#         group = group.sort_values("year")
#         rows.extend(_compute_novelty(group, "datasource_score"))

#     cols = df.columns.tolist() + ["novelty"]
#     return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------
# 4. DATATYPE-LEVEL HARMONIC SCORE
# ----------------------------------------------------
def harmonic_by_datatype(ds_df):
    rows = []

    ds_df["datatypeId"] = ds_df["datasourceId"].map(lambda x: DATA_SOURCES[x]["datatype"])
    ds_df["weight"] = ds_df["datasourceId"].map(lambda x: DATA_SOURCES[x]["weight"])
    ds_df["weighted"] = ds_df["score"] * ds_df["weight"]

    grouped = ds_df.groupby(["sourceId", "targetId", "source_type", "target_type", "relation", "datatypeId"])

    for (src, tgt, src_type, tgt_type, rel, dt), group in tqdm(grouped, desc="Datatype harmonic"):
        year_dict = group.groupby("year")["weighted"].apply(list).to_dict()
        collected = []
        for y in YEARS:
            if y in year_dict:
                collected.extend(year_dict[y])
            hs = harmonic_sum(collected)
            rows.append([src, tgt, src_type, tgt_type, rel, dt, y, hs])

    df = pd.DataFrame(rows, columns=[
        "sourceId", "targetId", "source_type", "target_type",
        "relation", "datatypeId", "year", "datatype_score"
    ])
    df.rename(columns={"datatype_score": "score"}, inplace=True)
    return df



# ----------------------------------------------------
# 5. DATATYPE-LEVEL NOVELTY
# ----------------------------------------------------
def novelty_by_datatype(df):
    rows = []
    grouped = df.groupby(["sourceId", "targetId", "datatypeId"])

    for _, group in tqdm(grouped, desc="Datatype novelty"):
        group = group.sort_values("year")
        rows.extend(_compute_novelty(group, "datatype_score"))

    cols = df.columns.tolist() + ["novelty"]
    return pd.DataFrame(rows, columns=cols)

# ----------------------------------------------------
# 6. TEMPORAL-DEDUPLICATION
# ----------------------------------------------------
def filter_temporal_edges(df):
    """
    Keep only meaningful harmonic changes.

    Automatically detects whether the input df is:
      - datasource harmonic   → expects datasourceId
      - datatype harmonic     → expects datatypeId
    """

    df = df.copy()

    # Determine grouping key automatically
    if "datasourceId" in df.columns:
        df["relation_key"] = df["datasourceId"] + "::" + df["relation"]
        group_cols = ["sourceId", "targetId", "relation_key"]
    elif "datatypeId" in df.columns:
        df["relation_key"] = df["datatypeId"] + "::" + df["relation"]
        group_cols = ["sourceId", "targetId", "relation_key"]
    else:
        raise ValueError("df must contain either 'datasourceId' or 'datatypeId'")

    # Sort properly
    df = df.sort_values(["sourceId", "targetId", "relation_key", "year"])

    # Group
    g = df.groupby(group_cols)

    # Previous year's score
    df["score_prev"] = g["score"].shift(1)

    # First appearance of evidence (first non-zero)
    cond_first_nonzero = df["score_prev"].isna() & (df["score"] > 0)

    # Keep only increases (no decreases allowed in harmonic)
    cond_increase = df["score"] > df["score_prev"]

    # Final keep mask
    keep = cond_first_nonzero | cond_increase

    filtered = df[keep].copy()
    filtered.drop(columns=["score_prev"], inplace=True)

    print(f"\n🔍 Temporal filtering")
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered)}")
    print(f"Removed rows:  {len(df) - len(filtered)}\n")

    return filtered




if __name__ == "__main__":
    print("==============================================")
    print("🚀 Loading evidence sources")
    print("==============================================")

    # -------------------------
    # Load static + dynamic
    # -------------------------
    print("Loading dynamic evidence...")
    dynamic_evd = load_dynamic_evidence()

    print("Loading static evidence...")
    static_evd = load_static_evidence()

    print("\n🔍 Inspecting dynamic evidence")
    inspect_graph(dynamic_evd)

    print("\n🔍 Inspecting static evidence")
    inspect_graph(static_evd)

    # ======================================================
    # 1. DATASOURCE-LEVEL HARMONIC  (dynamic only)
    # ======================================================
    print("\n==============================================")
    print("📊 Computing datasource-level harmonic (dynamic)")
    print("==============================================")

    ds_h = harmonic_by_datasource(dynamic_evd)   # returns column "score"
    print(f"Datasource harmonic rows: {len(ds_h)}")

    # ----------------------------------------------
    # Apply monotonic temporal filtering
    # ----------------------------------------------
    print("\n⏳ Applying temporal monotonic filtering to datasource harmonic...")
    ds_filtered = filter_temporal_edges(ds_h)  # uses score not datasource_score
    print(f"Filtered datasource harmonic rows: {len(ds_filtered)}")


    # ======================================================
    # 2. Add static edges (raw score = 1)
    # ======================================================
    print("\n==============================================")
    print("➕ Merging static edges into datasource harmonic output")
    print("==============================================")

    # Merge dynamic_filtered + static_raw
    ds_merged = pd.concat([ds_filtered, static_evd], ignore_index=True)

    # Save output
    print(f"💾 Saving datasource harmonic merged file to:\n{DATASOURCE_HARMONIC_FILE}")
    ds_merged.to_parquet(DATASOURCE_HARMONIC_FILE, index=False)


    # ======================================================
    # 3. DATATYPE-LEVEL HARMONIC (dynamic only)
    # ======================================================
    print("\n==============================================")
    print("📚 Computing datatype-level harmonic (dynamic)")
    print("==============================================")

    dt_h = harmonic_by_datatype(ds_h)
    print(f"Datatype harmonic rows: {len(dt_h)}")

    # ----------------------------------------------
    # Apply monotonic temporal filtering
    # ----------------------------------------------
    print("\n⏳ Applying temporal monotonic filtering to datatype harmonic...")
    dt_filtered = filter_temporal_edges(dt_h)
    print(f"Filtered datatype harmonic rows: {len(dt_filtered)}")

    # Save datatype harmonic
    print(f"💾 Saving datatype harmonic file to:\n{DATATYPE_HARMONIC_FILE}")
    dt_filtered.to_parquet(DATATYPE_HARMONIC_FILE, index=False)


    # ======================================================
    # COMPLETED
    # ======================================================
    print("\n🎉 COMPLETED OPEN TARGETS TEMPORAL PIPELINE")

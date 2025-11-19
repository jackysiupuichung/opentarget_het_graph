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
from datetime import timedelta

from pyspark.sql import functions as F
from pyspark.sql import SparkSession, Window
from pyspark.sql import types as T


# ----------------------------------------
# CONFIGURATION
# ----------------------------------------
EDGE_DIR = "data/kg_output/edges"

FIRST_YEAR = 1995
LAST_YEAR = 2025
MAX_HARMONIC = 1.644  # theoretical max sum of 1/i^2

# novelty settings
NOVELTY_SCALE = 2     # logistic steepness
NOVELTY_SHIFT = 2     # midpoint
NOVELTY_WINDOW = 10   # years after peak to decay

data_source = [
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

DATA_SOURCE = {
    ds["id"]: {
        "datatype": ds["aggregationId"],
        "weight": float(ds["weight"]),
    }
    for ds in data_source
}

spark = SparkSession.builder \
    .appName("OT-Lightweight-Timeseries") \
    .getOrCreate()

# ============================================================================
# 1. LOAD ALL DYNAMIC EVIDENCE (sourceId, targetId, score, year)
# ============================================================================
def load_dynamic_evidence():
    dfs = []
    for fname in os.listdir(EDGE_DIR):
        if not fname.endswith(".parquet"):
            continue

        datasourceId = fname.replace("sourceId=", "").replace(".parquet", "")

        df = (
            spark.read.parquet(os.path.join(EDGE_DIR, fname))
            .withColumn("datasourceId", F.lit(datasourceId))
            .withColumn("year", F.col("year").cast("integer"))
            .select("sourceId", "targetId", "datasourceId", "score", "year")
        )
        dfs.append(df)

    full = dfs[0].unionByName(*dfs[1:])
    return full.persist()


# ============================================================================
# 2. DATASOURCE-LEVEL HARMONIC SCORE (per year)
# ============================================================================
def harmonic_by_datasource(evd):
    # define complete year range
    years = spark.createDataFrame([(y,) for y in range(FIRST_YEAR, LAST_YEAR + 1)], ["year"])

    # Cartesian join ensures all combinations exist
    grid = (
        evd.select("datasourceId").distinct()
        .crossJoin(years)
        .crossJoin(evd.select("sourceId", "targetId").distinct())
    )

    df = (
        grid.join(evd, ["sourceId","targetId","datasourceId","year"], "left")
           .fillna(0, subset=["score"])
    )

    w = (
        Window.partitionBy("sourceId","targetId","datasourceId")
              .orderBy("year")
              .rangeBetween(Window.unboundedPreceding, 0)
    )

    out = (
        df.groupBy("sourceId","targetId","datasourceId","year")
          .agg(F.collect_list("score").alias("scores_year"))
          .withColumn("cum", F.flatten(F.collect_list("scores_year").over(w)))
          .withColumn("clean", F.expr("filter(cum, x -> x > 0)"))
          .withColumn("sorted", F.reverse(F.array_sort("clean")))
          .withColumn("top", F.expr("slice(sorted, 1, 50)"))
          .withColumn("idx", F.sequence(F.lit(1), F.size("top")))
          .withColumn("wts",
              F.expr("transform(arrays_zip(top,idx), x -> x.top / pow(x.idx,2))"))
          .withColumn("hs", F.expr("aggregate(wts, 0D, (acc,x)->acc+x)"))
          .withColumn("harmonic_score", F.col("hs") / F.lit(MAX_HARMONIC))
          .select("sourceId","targetId","datasourceId","year","harmonic_score")
    )
    return out.persist()


# ============================================================================
# 3. DATASOURCE-LEVEL NOVELTY
# ============================================================================
def novelty_by_datasource(hs_df):
    w = Window.partitionBy("sourceId","targetId","datasourceId").orderBy("year")

    # detect score increases
    peaks = (
        hs_df.fillna(0, subset=["harmonic_score"])
            .select(
                "sourceId","targetId","datasourceId",
                F.col("year").alias("peakYear"),
                (F.col("harmonic_score") -
                 F.lag("harmonic_score",1).over(w)).alias("peak")
            )
            .filter("peak > 0")
    )

    expanded = (
        peaks.select(
            "*",
            F.posexplode(
                F.sequence(F.col("peakYear"), F.col("peakYear")+F.lit(NOVELTY_WINDOW))
            ).alias("ix","year")
        ).drop("ix")
    )

    novelty = (
        expanded.groupBy("sourceId","targetId","datasourceId","year")
            .agg(F.max(
                F.col("peak") /
                (1 + F.exp(NOVELTY_SCALE*(F.col("year") - F.col("peakYear") - NOVELTY_SHIFT)))
            ).alias("novelty"))
    )

    out = hs_df.join(novelty, ["sourceId","targetId","datasourceId","year"], "left") \
               .fillna(0, subset=["novelty"])
    return out.persist()


# ============================================================================
# 4. DATATYPE-LEVEL HARMONIC + NOVELTY
# ============================================================================
def harmonic_by_datatype(ds_df):
    map_df = spark.createDataFrame(
        [(k, v["datatype"], v["weight"]) for k,v in DATA_SOURCES.items()],
        ["datasourceId","datatypeId","weight"]
    )

    df = (
        ds_df.join(map_df, "datasourceId", "left")
             .withColumn("weighted", F.col("harmonic_score") * F.col("weight"))
    )

    w = (
        Window.partitionBy("sourceId","targetId","datatypeId")
              .orderBy("year")
              .rangeBetween(Window.unboundedPreceding, 0)
    )

    out = (
        df.groupBy("sourceId","targetId","datatypeId","year")
          .agg(F.collect_list("weighted").alias("scores"))
          .withColumn("cum", F.flatten(F.collect_list("scores").over(w)))
          .withColumn("clean", F.expr("filter(cum, x -> x > 0)"))
          .withColumn("sorted", F.reverse(F.array_sort("clean")))
          .withColumn("top", F.expr("slice(sorted, 1, 50)"))
          .withColumn("idx", F.sequence(F.lit(1), F.size("top")))
          .withColumn("wts",
              F.expr("transform(arrays_zip(top,idx), x -> x.top / pow(x.idx,2))"))
          .withColumn("hs", F.expr("aggregate(wts, 0D, (acc,x)->acc+x)"))
          .withColumn("datatype_score", F.col("hs") / F.lit(MAX_HARMONIC))
          .select("sourceId","targetId","datatypeId","year","datatype_score")
    )
    return out.persist()


def novelty_by_datatype(dt_df):
    w = Window.partitionBy("sourceId","targetId","datatypeId").orderBy("year")

    peaks = (
        dt_df.fillna(0, subset=["datatype_score"])
            .select(
                "sourceId","targetId","datatypeId",
                F.col("year").alias("peakYear"),
                (F.col("datatype_score") -
                 F.lag("datatype_score",1).over(w)).alias("peak")
            )
            .filter("peak > 0")
    )

    expanded = (
        peaks.select(
            "*",
            F.posexplode(
                F.sequence(F.col("peakYear"), F.col("peakYear")+F.lit(NOVELTY_WINDOW))
            ).alias("ix","year")
        ).drop("ix")
    )

    novelty = (
        expanded.groupBy("sourceId","targetId","datatypeId","year")
            .agg(F.max(
                F.col("peak") /
                (1 + F.exp(NOVELTY_SCALE*(F.col("year") - F.col("peakYear") - NOVELTY_SHIFT)))
            ).alias("novelty"))
    )

    out = dt_df.join(novelty, ["sourceId","targetId","datatypeId","year"], "left") \
               .fillna(0, subset=["novelty"])
    return out.persist()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("🚀 Loading dynamic evidence...")
    evidence = load_dynamic_evidence()

    print("📌 Computing datasource harmonic...")
    ds_h = harmonic_by_datasource(evidence)
    ds_h.write.mode("overwrite").parquet("out/datasource_harmonic")

    print("📌 Computing datasource novelty...")
    ds_n = novelty_by_datasource(ds_h)
    ds_n.write.mode("overwrite").parquet("out/datasource_novelty")

    print("📌 Computing datatype harmonic...")
    dt_h = harmonic_by_datatype(ds_h)
    dt_h.write.mode("overwrite").parquet("out/datatype_harmonic")

    print("📌 Computing datatype novelty...")
    dt_n = novelty_by_datatype(dt_h)
    dt_n.write.mode("overwrite").parquet("out/datatype_novelty")

    print("🎉 Completed dynamic temporal metrics pipeline!")
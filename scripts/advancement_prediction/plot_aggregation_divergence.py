#!/usr/bin/env python3
"""Aggregation-divergence figure (rliable-style) for the THBKG ablation.

For each model and cutoff N in {10,50,100}, plot RS@N under THREE aggregation
methods side by side, each with a 95% CI:
  - TA-mean   : equally-weighted mean over the 13 primary TAs (macro; spike-prone)
  - TA-median : per-TA median over the 13 primary TAs (outlier-robust)
  - pooled    : all primary-TA pairs on one global list (micro; Czech Fig-3)
TA-mean / TA-median CIs are 95% percentile bootstrap over the 13 TAs; pooled CI is
the Katz log-method CI already in the pooled CSV.

Makes the mean-vs-median/pooled divergence visible: spiky conditions (e.g.
novelty-only, GATv2) show TA-mean >> TA-median; the joint EAHGT clusters high
across all three. Mirrors Agarwal et al. (NeurIPS'21) aggregate-metrics plot and
the Benchmark Lottery (Dehghani et al. 2021) aggregation-flip phenomenon.
"""
import numpy as np
import pandas as pd
import plotnine as pn

RES = "headline_results/full_ablation_eval"
OUT = "headline_results/full_ablation_eval/plots/aggregation_divergence.png"
CUTOFFS = [10, 50, 100]
N_BOOT = 2000
RNG = np.random.default_rng(0)

PRIMARY = ["cancer or benign tumor", "cardiovascular disease", "endocrine system disease",
           "gastrointestinal disease", "genetic, familial or congenital disease",
           "hematologic disease", "immune system disease", "integumentary system disease",
           "musculoskeletal or connective tissue disease", "nervous system disease",
           "reproductive system or breast disease", "respiratory or thoracic disease",
           "urinary system disease"]
DISPLAY = {"ots__all": "OTS", "rdg__no_time__positive": "RDG", "enc_hgt_ens": "HGT",
           "enc_gatv2_ens": "GATv2", "enc_rgcn_ens": "R-GCN", "enc_compgcn_ens": "CompGCN",
           "abl_score_ens": "EAHGT-s", "abl_novelty_ens": "EAHGT-n", "p3_eahgt_both": "EAHGT-s,n"}
ORDER = ["OTS", "RDG", "HGT", "GATv2", "R-GCN", "CompGCN", "EAHGT-s", "EAHGT-n", "EAHGT-s,n"]

by_ta = pd.read_csv(f"{RES}/relative_success_by_ta.csv")
by_ta = by_ta[(by_ta.stratum == "all") & (by_ta.therapeutic_area_name.isin(PRIMARY))]
pooled = pd.read_csv(f"{RES}/relative_success_by_limit_by_stratum_pooled.csv")
pooled = pooled[pooled.stratum == "all"]

rows = []
for m, disp in DISPLAY.items():
    for N in CUTOFFS:
        vals = by_ta[(by_ta.model_name == m) & (by_ta.limit == N)]["relative_success"].dropna().values
        if len(vals):
            boot_mean = [RNG.choice(vals, len(vals), replace=True).mean() for _ in range(N_BOOT)]
            boot_med = [np.median(RNG.choice(vals, len(vals), replace=True)) for _ in range(N_BOOT)]
            rows.append(dict(model=disp, cutoff=N, agg="TA-mean", rs=vals.mean(),
                             lo=np.percentile(boot_mean, 2.5), hi=np.percentile(boot_mean, 97.5)))
            rows.append(dict(model=disp, cutoff=N, agg="TA-median", rs=np.median(vals),
                             lo=np.percentile(boot_med, 2.5), hi=np.percentile(boot_med, 97.5)))
        pr = pooled[(pooled.model_name == m) & (pooled.limit == N)]
        if len(pr):
            pr = pr.iloc[0]
            rows.append(dict(model=disp, cutoff=N, agg="pooled", rs=pr.relative_success,
                             lo=pr.relative_success_low, hi=pr.relative_success_high))

d = pd.DataFrame(rows)
d["model"] = pd.Categorical(d["model"], categories=ORDER, ordered=True)
d["agg"] = pd.Categorical(d["agg"], categories=["TA-mean", "TA-median", "pooled"], ordered=True)
d["cutoff"] = pd.Categorical(d["cutoff"].map(lambda n: f"RS@{n}"),
                             categories=[f"RS@{n}" for n in CUTOFFS], ordered=True)
# Log-y for legibility (RS is a ratio; spreads @10 without compressing @50/@100).
# Floor zeros/CIs to a small value so RS=0 points still render; cap the ceiling.
FLOOR, CEIL = 0.25, 14.0
d["rs_p"] = d["rs"].clip(lower=FLOOR)
d["lo_p"] = d["lo"].clip(lower=FLOOR, upper=CEIL)
d["hi_p"] = d["hi"].clip(lower=FLOOR, upper=CEIL)
# mark the genuine RS=0 points (no advancers in top-N) so they're not read as ~0.25
d["is_zero"] = d["rs"] <= 1e-9

p = (pn.ggplot(d, pn.aes(x="model", y="rs_p", color="agg"))
     + pn.geom_hline(yintercept=1.0, linetype="dotted", color="grey", size=0.4)
     + pn.geom_errorbar(pn.aes(ymin="lo_p", ymax="hi_p"), width=0.25,
                        position=pn.position_dodge(width=0.65), size=0.45, na_rm=True)
     + pn.geom_point(pn.aes(shape="is_zero"), position=pn.position_dodge(width=0.65), size=1.9)
     + pn.facet_wrap("~cutoff", ncol=1)
     + pn.scale_y_log10(breaks=[0.25, 0.5, 1, 2, 4, 8], limits=(FLOOR, CEIL))
     + pn.scale_color_manual(values={"TA-mean": "#d95f02", "TA-median": "#1b9e77", "pooled": "#7570b3"},
                             name="aggregation")
     + pn.scale_shape_manual(values={False: "o", True: "x"}, guide=None)
     + pn.labs(x="", y="relative success (log scale, 95% CI; × = RS$=$0)")
     + pn.theme_bw()
     + pn.theme(figure_size=(8.5, 8), axis_text_x=pn.element_text(rotation=40, ha="right"),
                legend_position="top", panel_grid_minor=pn.element_blank()))
p.save(OUT, dpi=300, verbose=False)
print(f"wrote {OUT}")
print(d.pivot_table(index=["model"], columns=["cutoff", "agg"], values="rs").round(2).to_string())

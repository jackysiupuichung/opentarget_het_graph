"""
Evaluation script for advancement model predictions.

Inputs:
    --datasets_dir     directory containing evaluation_dataset.zarr
    --ta_parquet       path to therapeutic_areas.parquet
    --primary_tas_json path to primary_therapeutic_areas.json

Prediction sources:
    --only   comma-separated subset of registered run names (see DEFAULT_RUNS)
    --inject explicit list of {"path": ..., "model_name": ...} dicts (fire JSON)

By default, evaluates all runs in DEFAULT_RUNS that have predictions.

Usage:
    python evaluate_advancement.py
    python evaluate_advancement.py --only b1_hgt,p3_eahgt_both
    python evaluate_advancement.py --help
"""
import json
import logging
import os
import textwrap
import numpy as np
import pandas as pd
import xarray as xr
import plotnine as pn
import mizani.bounds
import fire
from pathlib import Path
from scipy.stats import wilcoxon, mannwhitneyu
from scipy.stats.contingency import relative_risk
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
# Run directories live outside the repo (large checkpoints/predictions). By
# default they resolve under the author's cluster scratch; set THBKG_DATA_ROOT
# to point the run registry at your own copy (e.g. the Zenodo release unpacked
# locally). The registry keys below are joined onto these roots.
_DATA_ROOT = os.environ.get(
    "THBKG_DATA_ROOT", "/gpfs/scratch/bty414/opentarget_evidences"
)
# All canonical runs live on the 26.03 graph generation, which reproduces the
# evaluation_dataset.zarr test pairs 9094/9094 and carries both advancement edge
# features (score + novelty). The 23.06 generation had a same-year clinical-trial
# edge leak and is retired — do not repoint anything at it.
_SCRATCH_2603 = f"{_DATA_ROOT}/26.03/runs"
# Canonical 26.03 graph + node mappings (overridable per call via --graph_file /
# --mappings_file, or globally via THBKG_DATA_ROOT).
_GRAPH_FILE = f"{_DATA_ROOT}/26.03/graph/hetero_graph_with_features_datatype.pt"
_MAPPINGS_FILE = (
    f"{_DATA_ROOT}/26.03/progression/temporal_graph_datatype_mappings.pt"
)


DEFAULT_RUNS: dict[str, str] = {
    # p3_lambdarank: directed graph (canonical)
    "p3_lambdarank_directed": "runs/advancement_lambdarank",
    # Study A — Model comparison (no edge features). The per-encoder baselines are
    # the 5-seed rank-fused 26.03 ensembles (same dirs as enc_*_ens below); the
    # b* aliases are kept so the headline analysis scripts that key on these slugs
    # keep resolving.
    "b1_hgt":           f"{_SCRATCH_2603}/encoder_baselines/hgt/ensemble",
    "b3_gatv2":         f"{_SCRATCH_2603}/encoder_baselines/gatv2/ensemble",
    "b6_rgcn":          f"{_SCRATCH_2603}/encoder_baselines/rgcn/ensemble",
    "b7_compgcn":       f"{_SCRATCH_2603}/encoder_baselines/compgcn/ensemble",
    # Study B — EAHGT ablation (HGT + edge feature variants; canonical HPs)
    "p1_eahgt_score":   f"{_SCRATCH_2603}/p1_eahgt_score_grouped_ap_v3",
    "p2_eahgt_novelty": f"{_SCRATCH_2603}/p2_eahgt_novelty_grouped_ap_v3",
    # Official EAHGT = grouped-allTA (group_all_tas) 5-seed percentile-rank
    # ensemble (lrgrpk100 s1/s7/s42/s123/s2024). Beats RDG at every k, sig at
    # @50/@100. Built by scripts/advancement_prediction/build_grouped_ensemble.py.
    "p3_eahgt_both":    f"{_SCRATCH_2603}/grouped_ensemble_s5",
    # Matched-recipe edge-feature ablation: 5-seed rank-fused ensembles trained
    # under the SAME recipe as p3_eahgt_both but with score-only / novelty-only
    # edge features. Built by build_matched_ablation_ensembles.py.
    "abl_score_ens":    f"{_SCRATCH_2603}/ablation_matched/score/ensemble",
    "abl_novelty_ens":  f"{_SCRATCH_2603}/ablation_matched/novelty/ensemble",
    # Encoder-family baselines: 5-seed rank-fused ensembles, each encoder at its
    # individual best params, under the headline grouped recipe (26.03 graph).
    # Built by build_encoder_baseline_ensembles.py. Same dirs as the b* aliases.
    "enc_hgt_ens":      f"{_SCRATCH_2603}/encoder_baselines/hgt/ensemble",
    "enc_gatv2_ens":    f"{_SCRATCH_2603}/encoder_baselines/gatv2/ensemble",
    "enc_rgcn_ens":     f"{_SCRATCH_2603}/encoder_baselines/rgcn/ensemble",
    "enc_compgcn_ens":  f"{_SCRATCH_2603}/encoder_baselines/compgcn/ensemble",
    # ndcgk_corr study — random-val LambdaRank, ndcg_k cutoff sweep
    "ndcgk100":         f"{_SCRATCH_2603}/ndcgk_corr/ndcgk100",
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BASE_MODEL_COLORS = {
    "RDG": "#1f77b4",
    "OTS": "#2ca02c",
}
_EXTERNAL_MODEL_COLORS = [
    "#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf",
    "#bcbd22", "#f7b6d2", "#c5b0d5", "#ffbb78", "#98df8a", "#c49c94",
    "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
_MULTIINDEX_LEVELS = ["target_id", "disease_id"]

_ADV_ETYPE = ("target", "advancement", "disease")
_DIRECT_EVIDENCE_RELATIONS = {
    "genetic_association", "somatic_mutation", "affected_pathway",
    "animal_model", "rna_expression",
}
_LITERATURE_RELATION = "literature"


def _load_dataset(path: Path) -> xr.Dataset:
    return xr.open_zarr(path).load().set_index(index=_MULTIINDEX_LEVELS)


def _load_graph_and_mappings(graph_file: Path, mappings_file: Path):
    import torch
    data = torch.load(str(graph_file), weights_only=False)
    mappings = torch.load(str(mappings_file), weights_only=False)
    return data, mappings


def _compute_target_novelty(data, mappings) -> pd.DataFrame:
    """One row per advancement edge: (target_id, disease_id, t_decision, is_pioneer).

    Pioneer iff no advancement edge on the same target with edge_time strictly
    less than the current edge_time and a different disease_id.
    """
    node_mapping = mappings["node_mapping"]
    idx_to_target  = {v: k for k, v in node_mapping["target"].items()}
    idx_to_disease = {v: k for k, v in node_mapping["disease"].items()}

    adv = data[_ADV_ETYPE]
    src = adv.edge_index[0].cpu().numpy()
    dst = adv.edge_index[1].cpu().numpy()
    t   = adv.edge_time.cpu().numpy()

    df = pd.DataFrame({
        "target_id":  [idx_to_target[i]  for i in src],
        "disease_id": [idx_to_disease[i] for i in dst],
        "t_decision": t,
    })

    is_pioneer = np.ones(len(df), dtype=bool)
    for _, grp in df.groupby("target_id"):
        years = grp["t_decision"].values
        diseases = grp["disease_id"].values
        idxs = grp.index.values
        for i, (y, d) in enumerate(zip(years, diseases)):
            if ((years < y) & (diseases != d)).any():
                is_pioneer[idxs[i]] = False
    df["is_pioneer"] = is_pioneer
    return df


def _compute_evidence_sparsity(data, mappings, novelty_df: pd.DataFrame) -> pd.DataFrame:
    """Augment novelty_df with an `evidence_stratum` column.

    For each (target_id, disease_id, t_decision), check every
    ("target", relation, "disease") edge type (excluding advancement):
      - direct_evidence: any edge in _DIRECT_EVIDENCE_RELATIONS with edge_time < t_decision
      - literature_only: no direct but any literature edge with edge_time < t_decision
      - evidence_free: neither
    """
    node_mapping = mappings["node_mapping"]
    target_to_idx  = node_mapping["target"]
    disease_to_idx = node_mapping["disease"]

    # Build per-relation: Series mapping (t_idx, d_idx) -> min edge_time
    # Existence of min_time < t_decision means a prior edge exists.
    relation_min_time: dict[str, pd.Series] = {}
    for etype in data.edge_types:
        if etype[0] != "target" or etype[2] != "disease":
            continue
        relation = etype[1]
        if relation == "advancement":
            continue
        if relation not in _DIRECT_EVIDENCE_RELATIONS and relation != _LITERATURE_RELATION:
            continue
        store = data[etype]
        n = store.edge_index.size(1)
        if n == 0:
            continue
        if hasattr(store, "edge_time") and store.edge_time is not None:
            times = store.edge_time.cpu().numpy()
        else:
            # Static edges: treat as always-present (t = -inf < any t_decision)
            times = np.full(n, -np.inf)
        src = store.edge_index[0].cpu().numpy()
        dst = store.edge_index[1].cpu().numpy()
        key_df = pd.DataFrame({"s": src, "d": dst, "t": times})
        min_time = key_df.groupby(["s", "d"])["t"].min()
        relation_min_time[relation] = min_time

    strata = []
    for row in novelty_df.itertuples(index=False):
        t_idx = target_to_idx.get(row.target_id)
        d_idx = disease_to_idx.get(row.disease_id)
        t_dec = row.t_decision

        direct_hit = False
        lit_hit = False
        if t_idx is not None and d_idx is not None:
            key = (t_idx, d_idx)
            for relation, min_time_series in relation_min_time.items():
                if key not in min_time_series.index:
                    continue
                if min_time_series.loc[key] < t_dec:
                    if relation in _DIRECT_EVIDENCE_RELATIONS:
                        direct_hit = True
                    elif relation == _LITERATURE_RELATION:
                        lit_hit = True

        if direct_hit:
            strata.append("direct_evidence")
        elif lit_hit:
            strata.append("literature_only")
        else:
            strata.append("evidence_free")

    out = novelty_df.copy()
    out["evidence_stratum"] = strata
    return out


def _collect_inject(
    only: str | None,
    inject: list[dict] | None,
) -> list[dict]:
    """Resolve predictions from DEFAULT_RUNS into a flat inject list."""
    result = list(inject or [])

    # When an explicit --inject list is given WITHOUT --only, evaluate exactly
    # those models — do NOT also auto-load every DEFAULT_RUNS entry (that mixes
    # unrelated runs onto the current zarr and can collide display slugs).
    if inject and only is None:
        return result

    runs = dict(DEFAULT_RUNS)
    if only is not None:
        if isinstance(only, (list, tuple)):
            names = [str(n).strip() for n in only if str(n).strip()]
        else:
            names = [n.strip() for n in only.split(",") if n.strip()]
        unknown = [n for n in names if n not in runs]
        if unknown:
            raise ValueError(f"Unknown run name(s): {unknown}. Available: {list(runs.keys())}")
        runs = {n: runs[n] for n in names}

    for name, output_dir in runs.items():
        pred = Path(output_dir) / "test_predictions.parquet"
        if pred.exists():
            result.append({"path": str(pred), "model_name": name})
            logger.info(f"Found predictions for '{name}'")
        else:
            logger.warning(f"Skipping '{name}' — test_predictions.parquet not found at {pred}")

    return result


def _inject_predictions(evaluation_dataset: xr.Dataset, inject: list[dict]) -> xr.Dataset:
    target_ids  = evaluation_dataset.coords["target_id"].values
    disease_ids = evaluation_dataset.coords["disease_id"].values

    for entry in inject:
        path, model_name = entry["path"], entry["model_name"]
        logger.info(f"Injecting {model_name} from {path}")
        preds  = pd.read_parquet(path)
        lookup = preds.set_index(["target_id", "disease_id"])["score"]
        scores = np.array([lookup.get((t, d), 0.0) for t, d in zip(target_ids, disease_ids)])
        n_missing = (scores == 0.0).sum()
        logger.info(f"  scores: [{scores.min():.3f}, {scores.max():.3f}], missing: {n_missing}/{len(scores)}")

        pred_values = np.stack([1 - scores, scores], axis=1)[:, np.newaxis, :]
        da = xr.DataArray(
            pred_values,
            dims=evaluation_dataset["prediction"].dims,
            coords={
                **{k: evaluation_dataset["prediction"].coords[k]
                   for k in evaluation_dataset["prediction"].coords if k != "models"},
                "models": [model_name],
            },
        )
        new_pred = xr.concat([evaluation_dataset["prediction"], da], dim="models")
        evaluation_dataset = evaluation_dataset.drop_dims("models").assign(
            {**{v: evaluation_dataset[v] for v in evaluation_dataset.data_vars if v != "prediction"},
             "prediction": new_pred}
        )
    return evaluation_dataset


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_classification_metrics(evaluation_dataset: xr.Dataset,
                                     therapeutic_areas: pd.DataFrame,
                                     model_names: list[str],
                                     pair_filter: set | None = None) -> pd.DataFrame:
    """ROC AUC, average precision, and MCC (at score-optimal threshold) per model per TA."""
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

    outcomes = evaluation_dataset.outcome.squeeze("outcomes").to_series().reset_index()
    outcomes.columns = ["target_id", "disease_id", "outcome"]
    preds = (
        evaluation_dataset.prediction.sel(classes="positive")
        .to_series().reset_index()
        .rename(columns={"prediction": "score"})
    )
    preds = preds[preds["models"].isin(model_names)]

    ta_map = therapeutic_areas[["disease_id", "therapeutic_area_name"]].drop_duplicates()
    all_ta = pd.DataFrame({"disease_id": outcomes["disease_id"].unique(), "therapeutic_area_name": "all"})
    ta_map = pd.concat([ta_map, all_ta], ignore_index=True)

    rows = []
    for model, grp in preds.groupby("models"):
        merged = grp.merge(outcomes, on=["target_id", "disease_id"]).merge(ta_map, on="disease_id")
        if pair_filter is not None:
            pf_idx = pd.MultiIndex.from_tuples(list(pair_filter)) if pair_filter else pd.MultiIndex.from_tuples([], names=["target_id", "disease_id"])
            merged = merged[merged.set_index(["target_id", "disease_id"]).index.isin(pf_idx)]
        for ta, ta_grp in merged.groupby("therapeutic_area_name"):
            y, s = ta_grp["outcome"].astype(int), ta_grp["score"]
            if y.nunique() < 2:
                continue
            # MCC at threshold that maximizes MCC across unique score values
            s_arr = s.to_numpy()
            y_arr = y.to_numpy()
            uniq = np.unique(s_arr)
            if len(uniq) > 200:
                uniq = np.quantile(s_arr, np.linspace(0, 1, 200))
            best_mcc = -1.0
            for thr in uniq:
                pred = (s_arr >= thr).astype(int)
                if pred.sum() == 0 or pred.sum() == len(pred):
                    continue
                m = matthews_corrcoef(y_arr, pred)
                if m > best_mcc:
                    best_mcc = m
            rows.append({
                "models": model,
                "therapeutic_area_name": ta,
                "roc_auc": roc_auc_score(y, s),
                "average_precision": average_precision_score(y, s),
                "mcc": best_mcc,
                "n_samples": len(y),
                "n_positives": y.sum(),
            })
    return pd.DataFrame(rows)


def _compute_relative_success_by_limit(evaluation_dataset: xr.Dataset,
                                        model_names: list[str],
                                        limits: list[int],
                                        confidence: float = 0.9,
                                        disease_filter: set | None = None,
                                        pair_filter: set | None = None) -> pd.DataFrame:
    outcomes = evaluation_dataset.outcome.squeeze("outcomes").to_series().reset_index()
    outcomes.columns = ["target_id", "disease_id", "outcome"]
    preds = (
        evaluation_dataset.prediction.sel(classes="positive")
        .to_series().reset_index()
        .rename(columns={"prediction": "score"})
    )
    preds = preds[preds["models"].isin(model_names)]

    rows = []
    for model, grp in preds.groupby("models"):
        merged = grp.merge(outcomes, on=["target_id", "disease_id"])
        if disease_filter is not None:
            merged = merged[merged["disease_id"].isin(disease_filter)]
        if pair_filter is not None:
            pf_idx = (pd.MultiIndex.from_tuples(list(pair_filter))
                      if pair_filter else
                      pd.MultiIndex.from_tuples([], names=["target_id", "disease_id"]))
            merged = merged[merged.set_index(["target_id", "disease_id"]).index.isin(pf_idx)]
        for limit in limits:
            # Threshold-mask top-N, matching Related Sciences' analysis.py
            # (Czech et al.) `_binarize_feature` for faithful Figure 3
            # reproduction: exposed = {score >= kth-largest score}, so ties at
            # the threshold are INCLUDED and exposed may exceed `limit`. This is
            # their published definition; the pooled RS curve is meant to be
            # directly analogous to their Fig 3. (Note: our per-TA RS in
            # _compute_rs_by_ta uses exact-N rank selection instead — the
            # recommender-systems convention — so the two curves are not
            # identical by construction.)
            # Ref: github.com/related-sciences/clinical_advancement_paper analysis.py
            threshold = merged["score"].nlargest(limit).min()
            exposed = merged[merged["score"] >= threshold]
            control = merged[merged["score"] < threshold]
            if len(exposed) == 0 or len(control) == 0:
                continue
            rr = relative_risk(
                exposed["outcome"].sum(), len(exposed),
                control["outcome"].sum(), len(control),
            )
            ci = rr.confidence_interval(confidence_level=confidence)
            rows.append({
                "model_name": model,
                "limit": limit,
                "relative_success": rr.relative_risk,
                "relative_success_low": ci.low,
                "relative_success_high": ci.high,
                "n_exposed": len(exposed),
                "n_exposed_true": int(exposed["outcome"].sum()),
                "n_control": len(control),
            })
    return pd.DataFrame(rows)


def _compute_rs_by_ta(evaluation_dataset: xr.Dataset,
                       therapeutic_areas: pd.DataFrame,
                       model_names: list[str],
                       limits: list[int],
                       confidence: float = 0.9,
                       pair_filter: set | None = None) -> pd.DataFrame:
    outcomes = evaluation_dataset.outcome.squeeze("outcomes").to_series().reset_index()
    outcomes.columns = ["target_id", "disease_id", "outcome"]
    preds = (
        evaluation_dataset.prediction.sel(classes="positive")
        .to_series().reset_index()
        .rename(columns={"prediction": "score"})
    )
    preds = preds[preds["models"].isin(model_names)]

    ta_map = therapeutic_areas[["disease_id", "therapeutic_area_name", "therapeutic_area_id"]].drop_duplicates()

    rows = []
    for model, grp in preds.groupby("models"):
        merged = grp.merge(outcomes, on=["target_id", "disease_id"]).merge(ta_map, on="disease_id")
        if pair_filter is not None:
            pf_idx = pd.MultiIndex.from_tuples(list(pair_filter)) if pair_filter else pd.MultiIndex.from_tuples([], names=["target_id", "disease_id"])
            merged = merged[merged.set_index(["target_id", "disease_id"]).index.isin(pf_idx)]
        for ta, ta_grp in merged.groupby("therapeutic_area_name"):
            for limit in limits:
                if len(ta_grp) <= limit:
                    continue
                top_idx = ta_grp.nlargest(limit, "score").index
                exposed = ta_grp.loc[top_idx]
                control = ta_grp.drop(top_idx)
                if len(exposed) == 0:
                    continue
                rr = relative_risk(
                    exposed["outcome"].sum(), len(exposed),
                    control["outcome"].sum(), len(control),
                )
                ci = rr.confidence_interval(confidence_level=confidence)
                rows.append({
                    "model_name": model,
                    "therapeutic_area_name": ta,
                    "limit": limit,
                    "relative_success": rr.relative_risk if not np.isinf(rr.relative_risk) else np.nan,
                    "relative_success_low": ci.low,
                    "relative_success_high": ci.high,
                    "n_exposed": len(exposed),
                    "n_exposed_true": int(exposed["outcome"].sum()),
                    "n_total": len(ta_grp),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_plot(plot: pn.ggplot, path: Path) -> None:
    logger.info(f"Saving plot to {path}")
    plot = plot + pn.theme(
        # fill white AND set the rect edge to white so no border box is drawn
        plot_background=pn.element_rect(fill="white", color="white"),
        panel_background=pn.element_rect(fill="white", color="white"),
        panel_border=pn.element_blank(),
        axis_line=pn.element_blank(),
        panel_grid_major=pn.element_line(color="#dddddd"),
        panel_grid_minor=pn.element_blank(),
    )
    plot.save(str(path), verbose=False, dpi=300)


def _build_color_map(model_names: list[str]) -> dict:
    base = {
        "rdg__no_time__positive": _BASE_MODEL_COLORS["RDG"],
        "ots__all": _BASE_MODEL_COLORS["OTS"],
    }
    colors = dict(base)
    ext_idx = 0
    for name in model_names:
        if name not in colors:
            colors[name] = _EXTERNAL_MODEL_COLORS[ext_idx % len(_EXTERNAL_MODEL_COLORS)]
            ext_idx += 1
    return colors


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate(
    datasets_dir: str = f"{Path(__file__).parent}/advancement_data/datasets",
    ta_parquet: str = f"{Path(__file__).parent}/advancement_data/features/therapeutic_areas.parquet",
    primary_tas_json: str = f"{Path(__file__).parent}/advancement_data/results/primary_therapeutic_areas.json",
    only: str | None = None,
    results_dir: str = f"{Path(__file__).parent}/advancement_data/results/external",
    inject: list[dict] | None = None,
    graph_file: str = _GRAPH_FILE,
    mappings_file: str = _MAPPINGS_FILE,
):
    """
    Evaluate model predictions against the saved evaluation dataset.

    By default, evaluates all runs in DEFAULT_RUNS that have test_predictions.parquet.

    Args:
        datasets_dir:     directory with evaluation_dataset.zarr
        ta_parquet:       path to therapeutic_areas.parquet
        primary_tas_json: path to primary_therapeutic_areas.json
        only:             comma-separated subset of DEFAULT_RUNS to evaluate
        results_dir:      output directory for CSVs and plots
        inject:           explicit list of {"path": ..., "model_name": ...} dicts
    """
    datasets_dir = Path(datasets_dir)
    results_dir  = Path(results_dir)
    plots_dir    = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    logger.info("Loading evaluation dataset...")
    evaluation_dataset = _load_dataset(datasets_dir / "evaluation_dataset.zarr")

    logger.info("Loading therapeutic areas...")
    therapeutic_areas = pd.read_parquet(ta_parquet)

    logger.info("Loading primary therapeutic areas...")
    with open(primary_tas_json) as f:
        primary_therapeutic_areas = json.load(f)
    logger.info(f"  {len(primary_therapeutic_areas)} primary TAs: {primary_therapeutic_areas}")

    # ------------------------------------------------------------------
    # Collect predictions
    # ------------------------------------------------------------------
    all_inject = _collect_inject(only, inject)
    if not all_inject:
        logger.error("No predictions found. Nothing to evaluate.")
        return
    logger.info(f"Evaluating {len(all_inject)} model(s): {[e['model_name'] for e in all_inject]}")

    evaluation_dataset = _inject_predictions(evaluation_dataset, all_inject)
    external_model_names = [e["model_name"] for e in all_inject]

    # Base models always included for comparison if present in zarr
    base_models = [m for m in ["rdg__no_time__positive", "ots__all"]
                   if m in evaluation_dataset.coords["models"].values]
    all_model_names = base_models + external_model_names

    _STATIC_DISPLAY = {
        "rdg__no_time__positive": "RDG",
        "ots__all":               "OTS",
        "p3_lambdarank_directed": "EAHGT-dir",
        "undirected_v1":          "EAHGT-undir-v1",
        "undirected_v2":          "EAHGT-undir-v2",
        "undirected_v3":          "EAHGT-undir-v3",
        "undirected_v4":          "EAHGT-undir-v4",
        "unidirected_additive":   "EAHGT-undir-add",
        # Study A — Model comparison
        "b1_hgt":                 "HGT",
        "b3_gatv2":               "GATv2",
        "b6_rgcn":                "R-GCN",
        "b7_compgcn":             "CompGCN",
        # Study B — EAHGT ablation
        "p1_eahgt_score":         "EAHGT-score",
        "p2_eahgt_novelty":       "EAHGT-novelty",
        "p3_eahgt_both":          "EAHGT",
        # Official grouped 5-seed ensemble (headline model) — display as EAHGT
        "grouped_ensemble_latest_s5": "EAHGT",
        "grouped_ensemble_s5":        "EAHGT",
        # Masking-comparison: strict (< year) vs loose (<= year) ensembles
        "grouped_ensemble_strictmask_s5": "EAHGT-strict",
        "grouped_ensemble_loose_s5":      "EAHGT-loose",
        # Matched-recipe ablation ensembles (same recipe as p3_eahgt_both)
        "abl_score_ens":          "EAHGT-score",
        "abl_novelty_ens":        "EAHGT-novelty",
        # Encoder-family baseline ensembles
        "enc_hgt_ens":            "HGT",
        "enc_gatv2_ens":          "GATv2",
        "enc_rgcn_ens":           "R-GCN",
        "enc_compgcn_ens":        "CompGCN",
        # w3 retrain (26.03 w3 labels): HGT + GATv2 edge-feature ablation.
        # Distinct slugs so they don't collide with the _ens names under the
        # 12-char truncation used for unmapped external models.
        "enc_hgt_w3":             "HGT",
        "enc_gatv2_w3":           "GATv2",
        "enc_rgcn_w3":            "R-GCN",
        "enc_compgcn_w3":         "CompGCN",
        "abl_score_w3":           "EAHGT-score",
        "abl_novelty_w3":         "EAHGT-novelty",
        "p3_eahgt_both_w3":       "EAHGT",
        "gatv2_score_w3":         "GATv2-score",
        "gatv2_novelty_w3":       "GATv2-novelty",
        "gatv2_both_w3":          "GATv2-both",
        # Bilinear-decoder variant (collapse-resistant; reported alongside MLP)
        "p3_eahgt_both_bilinear": "EAHGT-Bilinear",
        # 23.06 EAHGT variations (decoder / loss / centering / training)
        "eahgt_mlp":              "EAHGT-MLP",
        "eahgt_bilinear":         "EAHGT-Bilinear",
        "eahgt_grouped":          "EAHGT-Grouped",
        "eahgt_leakfix":          "EAHGT-leakfix",
        "eahgt_lambdarank_v2":    "EAHGT-LR-v2",
        # 26.03 random 80/20 train/val split (test stays temporal >=2016)
        "eahgt_randomsplit":          "EAHGT",
        "eahgt_randomsplit_score":    "EAHGT-score",
        "eahgt_randomsplit_novelty":  "EAHGT-novelty",
        "eahgt_center_raw":       "EAHGT-raw",
        "eahgt_center_mean":      "EAHGT-mean-ctr",
        "eahgt_center_meanstd":   "EAHGT-mean-std",
    }
    model_display = {
        **_STATIC_DISPLAY,
        **{n: n.upper().replace("__", "-")[:12]
           for n in external_model_names
           if n not in _STATIC_DISPLAY},
    }
    # Canonical legend order: our model first, then standard GNN baselines,
    # then tabular / heuristic baselines. Applied to slug_colors (insertion
    # order = legend order in plotnine) and to model_order downstream.
    _CANONICAL_SLUG_ORDER = [
        "EAHGT", "EAHGT-strict", "EAHGT-loose", "EAHGT-Bilinear", "EAHGT-score", "EAHGT-novelty",
        # 23.06 EAHGT variations
        "EAHGT-MLP", "EAHGT-Grouped", "EAHGT-leakfix", "EAHGT-LR-v2",
        "EAHGT-raw", "EAHGT-mean-ctr", "EAHGT-mean-std",
        "HGT",
        "GATv2", "GATv2-score", "GATv2-novelty", "GATv2-both",
        "R-GCN", "CompGCN",
        "RDG", "GBM", "OTS",
        "Random",
    ]
    _slug_to_name = {model_display.get(m, m): m for m in all_model_names}
    ordered_names = []
    seen = set()
    for slug in _CANONICAL_SLUG_ORDER:
        n = _slug_to_name.get(slug)
        if n is not None and n not in seen:
            ordered_names.append(n)
            seen.add(n)
    # Fall through anything not in canonical list (preserves prior order)
    for n in all_model_names:
        if n not in seen:
            ordered_names.append(n)
            seen.add(n)
    all_model_names = ordered_names
    color_map  = _build_color_map(all_model_names)
    slug_colors = {model_display.get(m, m): color_map[m] for m in all_model_names}
    # EAHGT ablation variants keep the EAHGT hue but are drawn a progressively
    # LIGHTER shade (tinted toward white), so they read as "one model family,
    # several configurations" while staying distinguishable. Colour is the sole
    # per-model channel across every plot (lines, violins/boxes, heatmaps); no
    # linetype aesthetic is used.
    import matplotlib.colors as mcolors
    def _lighten(hex_color, amount):
        """Blend `hex_color` toward white by `amount` in [0, 1]."""
        rgb = mcolors.to_rgb(hex_color)
        return tuple(c + (1.0 - c) * amount for c in rgb)

    # Per-family: base slug -> {variant slug: tint amount toward white}. Each
    # family's variants share the base hue and separate only by shade.
    _FAMILY_VARIANT_TINTS = {
        "EAHGT": {
            "EAHGT-strict":  0.30,
            "EAHGT-loose":   0.45,
            "EAHGT-score":   0.30,
            "EAHGT-novelty": 0.50,
        },
        "GATv2": {
            "GATv2-score":   0.30,
            "GATv2-novelty": 0.50,
            "GATv2-both":    0.15,
        },
    }
    # Flat slug -> (base, amount) lookup for the color_map propagation below.
    _VARIANT_TINTS = {
        v: (base, amt)
        for base, variants in _FAMILY_VARIANT_TINTS.items()
        for v, amt in variants.items()
    }
    for _base, _variants in _FAMILY_VARIANT_TINTS.items():
        if _base not in slug_colors:
            continue
        for _variant_slug, _amt in _variants.items():
            if _variant_slug in slug_colors:
                slug_colors[_variant_slug] = _lighten(slug_colors[_base], _amt)
    # Propagate the same tint to color_map (keyed by model NAME, not slug) so
    # the heatmaps — which build per-model colormaps from color_map — also
    # differentiate variants by shade.
    for _name in all_model_names:
        _tint = _VARIANT_TINTS.get(model_display.get(_name))
        if _tint is not None:
            _base, _amt = _tint
            if _base in slug_colors:
                color_map[_name] = _lighten(slug_colors[_base], _amt)
    # Helper: make model_slug an ordered categorical so plotnine renders the
    # legend in canonical order (EAHGT first, baselines last).
    _slug_categories = list(slug_colors.keys())
    def _as_ordered_slug(s):
        return pd.Categorical(s, categories=_slug_categories, ordered=True)

    # Headline line/violin plots show one representative per encoder family — the
    # family's best-scoring configuration by TA-mean RS@10 — alongside the OTS /
    # RDG reference baselines. This is COMPUTED from the eval below (see
    # `_HEADLINE_SLUGS = _select_headline_slugs(...)` after rs_by_ta_primary is
    # built), not hand-maintained, so the headline always reflects the current
    # run. This placeholder is only a fallback if that selection can't run.
    _HEADLINE_SLUGS = ["EAHGT", "EAHGT-novelty", "GATv2-both", "RDG", "OTS"]

    # ------------------------------------------------------------------
    # Test-pair stratification (target novelty × evidence sparsity)
    # ------------------------------------------------------------------
    strata_csv = results_dir / "test_pair_strata.csv"
    if strata_csv.exists():
        logger.info(f"Loading cached test pair strata from {strata_csv}")
        strata_df = pd.read_csv(strata_csv)
    else:
        logger.info("Cached strata not found — computing from graph...")
        data, mappings = _load_graph_and_mappings(Path(graph_file), Path(mappings_file))
        novelty_df = _compute_target_novelty(data, mappings)
        strata_df = _compute_evidence_sparsity(data, mappings, novelty_df)
        strata_df["novelty_stratum"] = np.where(strata_df["is_pioneer"], "pioneer", "known")
        strata_df.to_csv(strata_csv, index=False)
        logger.info(f"Wrote {strata_csv}")

    # Monotonicity check: once a target is Known at year Y, every later row for
    # that target must also be Known.
    for tid, grp in strata_df.sort_values("t_decision").groupby("target_id"):
        pioneer_flags = grp["is_pioneer"].values
        if len(pioneer_flags) > 1 and pioneer_flags[1:].any() and not pioneer_flags[:-1].all():
            # A False followed by a True within the same target would violate monotonicity
            if any(pioneer_flags[i] and not pioneer_flags[:i].all()
                   for i in range(1, len(pioneer_flags))):
                logger.warning(f"Monotonicity violation for target {tid}")

    strata_summary = (
        strata_df.groupby(["novelty_stratum", "evidence_stratum"]).size().unstack(fill_value=0)
    )
    logger.info("Strata counts (full advancement set):\n%s", strata_summary.to_string())

    # Restrict strata_df to pairs actually present in the evaluation dataset (test set)
    eval_pairs = pd.DataFrame({
        "target_id":  evaluation_dataset.coords["target_id"].values,
        "disease_id": evaluation_dataset.coords["disease_id"].values,
    })
    strata_test = strata_df.merge(eval_pairs, on=["target_id", "disease_id"], how="inner")
    logger.info(
        "Test-set strata counts:\n%s",
        strata_test.groupby(["novelty_stratum", "evidence_stratum"]).size().unstack(fill_value=0).to_string(),
    )

    n_pioneer = int(strata_test["is_pioneer"].sum())
    n_known   = int((~strata_test["is_pioneer"]).sum())
    pioneer_ratio = n_pioneer / max(len(strata_test), 1)
    if pioneer_ratio < 0.01 or pioneer_ratio > 0.20:
        logger.warning(
            f"Pioneer ratio {pioneer_ratio:.1%} is outside the expected 1%%-20%% range — "
            f"possible graph/mapping mismatch (n_pioneer={n_pioneer}, n_known={n_known})"
        )

    def _pairs(df: pd.DataFrame) -> set:
        return set(map(tuple, df[["target_id", "disease_id"]].to_numpy()))

    pair_strata: dict[str, set | None] = {
        "all":             None,
        "pioneer":         _pairs(strata_test[strata_test["is_pioneer"]]),
        "known":           _pairs(strata_test[~strata_test["is_pioneer"]]),
        "evidence_free":   _pairs(strata_test[strata_test["evidence_stratum"] == "evidence_free"]),
        "literature_only": _pairs(strata_test[strata_test["evidence_stratum"] == "literature_only"]),
        "direct_evidence": _pairs(strata_test[strata_test["evidence_stratum"] == "direct_evidence"]),
        "pioneer__evidence_free":   _pairs(strata_test[strata_test["is_pioneer"]  & (strata_test["evidence_stratum"] == "evidence_free")]),
        "pioneer__literature_only": _pairs(strata_test[strata_test["is_pioneer"]  & (strata_test["evidence_stratum"] == "literature_only")]),
        "pioneer__direct_evidence": _pairs(strata_test[strata_test["is_pioneer"]  & (strata_test["evidence_stratum"] == "direct_evidence")]),
        "known__evidence_free":     _pairs(strata_test[~strata_test["is_pioneer"] & (strata_test["evidence_stratum"] == "evidence_free")]),
        "known__literature_only":   _pairs(strata_test[~strata_test["is_pioneer"] & (strata_test["evidence_stratum"] == "literature_only")]),
        "known__direct_evidence":   _pairs(strata_test[~strata_test["is_pioneer"] & (strata_test["evidence_stratum"] == "direct_evidence")]),
    }

    # ------------------------------------------------------------------
    # 1. Predictions CSV
    # ------------------------------------------------------------------
    logger.info("Saving predictions...")
    (
        evaluation_dataset.prediction.sel(classes="positive")
        .to_series().reset_index()
        .rename(columns={"prediction": "score"})
        .pipe(lambda df: df[df["models"].isin(all_model_names)])
        .to_csv(results_dir / "predictions.csv", index=False)
    )

    # ------------------------------------------------------------------
    # 2. Classification metrics (stratified)
    # ------------------------------------------------------------------
    logger.info("Computing classification metrics across %d strata...", len(pair_strata))
    cm_frames = []
    for stratum, pair_filter in pair_strata.items():
        if pair_filter is not None and len(pair_filter) == 0:
            logger.warning(f"Stratum '{stratum}' is empty — skipping classification metrics")
            continue
        cm_s = _compute_classification_metrics(
            evaluation_dataset, therapeutic_areas, all_model_names, pair_filter=pair_filter
        )
        cm_s["stratum"] = stratum
        cm_frames.append(cm_s)
    classification_metrics = pd.concat(cm_frames, ignore_index=True)
    classification_metrics.to_csv(results_dir / "classification_metrics.csv", index=False)

    # ------------------------------------------------------------------
    # 3 & 4. Relative success by TA (primary TAs), then average for by-limit plot
    # ------------------------------------------------------------------
    _primary_ta_set = set(primary_therapeutic_areas) - {"all"}
    logger.info("Computing relative success by therapeutic area across %d strata...", len(pair_strata))
    rs_frames = []
    for stratum, pair_filter in pair_strata.items():
        if pair_filter is not None and len(pair_filter) == 0:
            continue
        rs_s = _compute_rs_by_ta(
            evaluation_dataset,
            therapeutic_areas,
            all_model_names,
            limits=sorted({*np.arange(10, 101, 10).tolist(), 75}),
            pair_filter=pair_filter,
        )
        rs_s["stratum"] = stratum
        rs_frames.append(rs_s)
    rs_by_ta_full = pd.concat(rs_frames, ignore_index=True)
    # Existing plots use the "all" subset
    rs_by_ta = rs_by_ta_full[rs_by_ta_full["stratum"] == "all"].drop(columns=["stratum"]).copy()
    rs_by_ta["model_slug"] = rs_by_ta["model_name"].map(model_display)

    # Derive rs_by_limit as mean RS across primary TAs (matching reference paper)
    logger.info("Computing relative success by limit (mean over primary TAs)...")
    rs_by_ta_primary = rs_by_ta[rs_by_ta["therapeutic_area_name"].isin(_primary_ta_set)].copy()
    rs_by_limit = (
        rs_by_ta_primary
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_success=("relative_success", "mean"))
    )

    # Headline model selection: best-scoring configuration PER ENCODER FAMILY.
    # Families are keyed by the slug prefix before the first "-" (EAHGT, GATv2,
    # HGT, R-GCN, …); within each, the config with the highest TA-mean RS@10
    # wins. We keep the top `n_families` families (by their best score) plus the
    # RDG / OTS reference baselines. Result: an honest architecture comparison
    # (one representative per encoder) rather than a hand-picked whitelist.
    _REFERENCE_SLUGS = ["RDG", "OTS"]
    def _select_headline_slugs(rs_df: pd.DataFrame, n_families: int = 2,
                               rank_limit: int = 10) -> list[str]:
        at = rs_df[rs_df["limit"] == rank_limit]
        if at.empty:
            return _HEADLINE_SLUGS  # fall back to the placeholder
        score = at.groupby("model_slug")["relative_success"].mean()
        fam = {}  # family -> (best_slug, best_score)
        for slug, s in score.items():
            if slug in _REFERENCE_SLUGS or pd.isna(slug):
                continue
            family = str(slug).split("-", 1)[0]
            if family not in fam or s > fam[family][1]:
                fam[family] = (slug, s)
        top = sorted(fam.values(), key=lambda t: t[1], reverse=True)[:n_families]
        chosen = [slug for slug, _ in top]
        chosen += [r for r in _REFERENCE_SLUGS if r in set(rs_df["model_slug"])]
        return chosen

    _HEADLINE_SLUGS = _select_headline_slugs(rs_by_limit)
    logger.info(f"Headline models (best per encoder family + refs): {_HEADLINE_SLUGS}")

    # Stratified rs_by_limit (mean over primary TAs per stratum) — kept for the
    # CSV / TA-mean consumers.
    rs_by_ta_full["model_slug"] = rs_by_ta_full["model_name"].map(model_display)
    rs_by_ta_primary_full = rs_by_ta_full[
        rs_by_ta_full["therapeutic_area_name"].isin(_primary_ta_set)
    ].copy()
    rs_by_limit_full = (
        rs_by_ta_primary_full
        .groupby(["model_name", "model_slug", "limit", "stratum"], as_index=False)
        .agg(relative_success=("relative_success", "mean"))
    )

    # Stratified POOLED rs_by_limit — pooled top-N RR within each stratum
    # (Czech-style), consistent with the headline pooled plot. The by-stratum
    # line chart uses this instead of the TA-mean so both figures report the
    # same quantity.
    logger.info("Computing pooled relative success by limit per stratum...")
    _strat_limits = sorted({*np.arange(10, 101, 10).tolist(), 75})
    # Pooled metrics are restricted to the SAME evaluation population as the
    # TA-averaged metric: pairs whose disease belongs to >=1 retained (primary)
    # therapeutic area. The TA exclusion (biological_process et al.) is applied
    # once, up front, to BOTH the pooled and TA-averaged paths -- not removed
    # post-hoc from pooling. Excluded-only diseases (notably the generic
    # biological_process bucket) otherwise dominate the single global ranking and
    # are not part of the reported evaluation set.
    _primary_ta_set_pool = set(primary_therapeutic_areas) - {"all"}
    _primary_disease_filter = set(
        therapeutic_areas.loc[
            therapeutic_areas["therapeutic_area_name"].isin(_primary_ta_set_pool),
            "disease_id",
        ].unique()
    )
    rs_pooled_strat_frames = []
    for stratum, pair_filter in pair_strata.items():
        if pair_filter is not None and len(pair_filter) == 0:
            continue
        rs_ps = _compute_relative_success_by_limit(
            evaluation_dataset, all_model_names, _strat_limits,
            confidence=0.95, pair_filter=pair_filter,
            disease_filter=_primary_disease_filter,
        )
        rs_ps["stratum"] = stratum
        rs_pooled_strat_frames.append(rs_ps)
    rs_pooled_by_limit_full = pd.concat(rs_pooled_strat_frames, ignore_index=True)
    rs_pooled_by_limit_full["model_slug"] = rs_pooled_by_limit_full["model_name"].map(model_display)
    # Saved for reference (Czech-style pooled). The by-stratum line chart now uses
    # the TA-mean (grouped) metric `rs_by_limit_full` instead — see below.
    rs_pooled_by_limit_full.to_csv(
        results_dir / "relative_success_by_limit_by_stratum_pooled.csv", index=False
    )

    # Pooled RS with 95% Katz CI — fine-grained limits for the paper-style line plot
    _n_pairs_total = int(
        evaluation_dataset.outcome.squeeze("outcomes").to_series().reset_index().shape[0]
    )
    _fine_limits = sorted(set(
        list(range(10, 50))
        + list(range(50, 200, 5))
        + list(range(200, 501, 10))
    ))
    logger.info(
        "Computing pooled RS with 95%% Katz CI over %d limits "
        "(restricted to %d primary-TA diseases)...",
        len(_fine_limits), len(_primary_disease_filter),
    )
    rs_pooled = _compute_relative_success_by_limit(
        evaluation_dataset, all_model_names, _fine_limits, confidence=0.95,
        disease_filter=_primary_disease_filter,
    )
    rs_pooled["model_slug"] = rs_pooled["model_name"].map(model_display)
    rs_pooled = rs_pooled.sort_values(["model_slug", "limit"])
    rs_pooled["relative_success_smooth"] = (
        rs_pooled.groupby("model_slug")["relative_success"]
        .transform(lambda s: s.rolling(window=9, center=True, min_periods=1).mean())
    )
    # CI bounds are NOT smoothed: each Katz interval is already computed
    # per-N, and rolling-averaging them smears adjacent CIs into each other
    # which understates per-N uncertainty. The `_smooth` columns alias the
    # raw bounds so the ribbon plotting code is unchanged.
    for _col in ("relative_success_low", "relative_success_high"):
        rs_pooled[f"{_col}_smooth"] = rs_pooled[_col]

    # Mean-of-ratios (TA-averaged) at fine limits for overlay on katz plot
    logger.info("Computing mean-of-ratios RS over fine limits for katz overlay...")
    rs_by_ta_fine_frames = []
    for stratum, pair_filter in [("all", None)]:
        rs_fine_s = _compute_rs_by_ta(
            evaluation_dataset, therapeutic_areas, all_model_names,
            limits=_fine_limits, pair_filter=pair_filter,
        )
        rs_by_ta_fine_frames.append(rs_fine_s)
    rs_by_ta_fine = pd.concat(rs_by_ta_fine_frames, ignore_index=True)
    rs_by_ta_fine["model_slug"] = rs_by_ta_fine["model_name"].map(model_display)
    _rs_by_ta_fine_primary = rs_by_ta_fine[
        rs_by_ta_fine["therapeutic_area_name"].isin(_primary_ta_set)
    ].copy()
    rs_mor = (
        _rs_by_ta_fine_primary
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_success=("relative_success", "mean"))
    )
    rs_mor = rs_mor.sort_values(["model_slug", "limit"])
    # window=5 (not 9): the TA-mean grid is coarse (~11 points), so a lighter
    # rolling mean lets the line track the dots more closely.
    rs_mor["relative_success_smooth"] = (
        rs_mor.groupby("model_slug")["relative_success"]
        .transform(lambda s: s.rolling(window=5, center=True, min_periods=1).mean())
    )

    # Bootstrap a 95% CI on the mean-of-ratios (resample over therapeutic
    # areas) so the band actually brackets the TA-averaged line plotted
    # below. This is the uncertainty of the headline metric, unlike the
    # pooled Katz CI which is the uncertainty of a different (pooled) RS.
    logger.info("Bootstrapping 95%% CI on mean-of-ratios over therapeutic areas...")
    _bootstrap_n = 2000
    _bootstrap_rng = np.random.default_rng(0)
    _mor_ci_rows = []
    for (m, slug, lim), grp in _rs_by_ta_fine_primary.groupby(
        ["model_name", "model_slug", "limit"]
    ):
        vals = grp["relative_success"].dropna().to_numpy()
        if vals.size < 2:
            continue
        idx = _bootstrap_rng.integers(0, vals.size, size=(_bootstrap_n, vals.size))
        boot_means = vals[idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        _mor_ci_rows.append(
            {"model_name": m, "model_slug": slug, "limit": lim,
             "relative_success_low": lo, "relative_success_high": hi}
        )
    rs_mor_ci = pd.DataFrame(_mor_ci_rows).sort_values(["model_slug", "limit"])
    for _col in ("relative_success_low", "relative_success_high"):
        rs_mor_ci[f"{_col}_smooth"] = (
            rs_mor_ci.groupby("model_slug")[_col]
            .transform(lambda s: s.rolling(window=9, center=True, min_periods=1).mean())
        )

    # Save CSVs: primary TAs + average row appended (stratified)
    avg_ta_rows = rs_by_limit_full.assign(therapeutic_area_name="average primary therapeutic area")
    pd.concat([rs_by_ta_primary_full, avg_ta_rows], ignore_index=True).to_csv(
        results_dir / "relative_success_by_ta.csv", index=False
    )
    rs_by_limit_full.to_csv(results_dir / "relative_success_by_limit.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots...")

    # Plot 1 (headline): pooled RS vs N with a Katz 95% CI band on the
    # proposed (EA-HGT) curve. Both line and band describe the same
    # quantity — the pooled top-N RR across the whole test set — directly
    # analogous to Figure 3 of the reference paper (Related Sciences).
    _pct1  = max(1, round(_n_pairs_total * 0.01))
    _pct2  = max(1, round(_n_pairs_total * 0.02))
    _max_limit_plot = 250
    _eahgt_slug = "EAHGT" if "EAHGT" in set(rs_pooled["model_slug"]) else model_display.get(
        "p3_lambdarank_undirected", model_display.get("p3_lambdarank_directed", "EAHGT"))
    _rs_ci_eahgt = rs_pooled[rs_pooled["model_slug"] == _eahgt_slug].copy()
    # Y-axis upper bound: 5% headroom above the highest CI upper bound across
    # all models so the entire ribbon stays in-frame. Earlier versions capped
    # this at 7.0 to match Czech et al.'s reference figure; that cap clips
    # stronger models whose RR exceeds 7 at small N.
    _rs_plot_max = float(rs_pooled["relative_success_high_smooth"].max()) * 1.05
    _nudge = 0.15

    def _build_pooled_plot(rs_df: pd.DataFrame, ci_df: pd.DataFrame) -> pn.ggplot:
        """Pooled RS-vs-N plot for the given subset of models.

        Every model is drawn as a solid line; colour alone carries the per-model
        distinction (including the EAHGT variants, which share a colour family).
        """
        _slugs = [s for s in _slug_categories if s in set(rs_df["model_slug"])]
        return (
            pn.ggplot(rs_df, pn.aes(x="limit", y="relative_success", color="model_slug", fill="model_slug", group="model_slug"))
            + pn.geom_ribbon(
                data=ci_df,
                mapping=pn.aes(x="limit", ymin="relative_success_low_smooth", ymax="relative_success_high_smooth", fill="model_slug", group="model_slug"),
                outline_type="full", alpha=0.20, color=None, inherit_aes=False,
            )
            + pn.geom_line(
                data=ci_df,
                mapping=pn.aes(x="limit", y="relative_success_low_smooth", color="model_slug", group="model_slug"),
                size=0.4, linetype="dotted", show_legend=False, inherit_aes=False,
            )
            + pn.geom_line(
                data=ci_df,
                mapping=pn.aes(x="limit", y="relative_success_high_smooth", color="model_slug", group="model_slug"),
                size=0.4, linetype="dotted", show_legend=False, inherit_aes=False,
            )
            + pn.geom_point(alpha=0.3, size=1, show_legend=False)
            + pn.geom_line(pn.aes(y="relative_success_smooth"), size=1, alpha=0.8)
            + pn.geom_hline(yintercept=1, linetype="dashed")
            + pn.annotate("text", x=_max_limit_plot, y=1 + _nudge, label="Random (RS=1)", size=8, ha="right")
            + pn.annotate("segment", x=_pct1, xend=_pct1, y=0, yend=_rs_plot_max, color="grey", linetype="dotted", size=0.6)
            + pn.annotate("text", x=_pct1, y=_rs_plot_max, label=f"Top 1%\n(n={_pct1})", size=7, ha="center", va="top", color="grey")
            + pn.annotate("segment", x=_pct2, xend=_pct2, y=0, yend=_rs_plot_max, color="grey", linetype="dotted", size=0.6)
            + pn.annotate("text", x=_pct2, y=_rs_plot_max, label=f"Top 2%\n(n={_pct2})", size=7, ha="center", va="top", color="grey")
            + pn.scale_color_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_fill_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_x_continuous(limits=(0, _max_limit_plot))
            + pn.scale_y_continuous(limits=(0, _rs_plot_max), oob=mizani.bounds.squish)
            + pn.labs(x="N top target-disease pairs", y="relative success", color="model", fill="model")
            + pn.theme_minimal()
            + pn.theme(figure_size=(10, 4))
        )

    # Headline: proposed model + two ablation variants + OTS / RDG only.
    _rs_pooled_headline = rs_pooled[rs_pooled["model_slug"].isin(_HEADLINE_SLUGS)].copy()
    _save_plot(
        _build_pooled_plot(_rs_pooled_headline, _rs_ci_eahgt),
        plots_dir / "relative_success_by_limit_pooled.png",
    )
    # Supplementary: all models (full comparison including encoder baselines).
    _save_plot(
        _build_pooled_plot(rs_pooled, _rs_ci_eahgt),
        plots_dir / "relative_success_by_limit_pooled_supp.png",
    )

    # Plot 1b (supp): TA-averaged mean-of-ratios (no CI band — the per-TA
    # average is the headline metric and bootstrapping over TAs adds noise
    # without informing the comparison). The pooled plot above already
    # shows uncertainty via the Katz CI on EA-HGT.
    # Auto-scale: 5% headroom above the highest smoothed value. Earlier
    # versions capped at 8.0 to match the reference paper; that clips
    # stronger models.
    _rs_mor_plot_max = float(rs_mor["relative_success_smooth"].max()) * 1.05 if "relative_success_smooth" in rs_mor.columns else 5.0

    def _build_mor_plot(rs_df: pd.DataFrame) -> pn.ggplot:
        _slugs = [s for s in _slug_categories if s in set(rs_df["model_slug"])]
        return (
            pn.ggplot(rs_df, pn.aes(x="limit", y="relative_success", color="model_slug", fill="model_slug", group="model_slug"))
            + pn.geom_point(alpha=0.3, size=1, show_legend=False)
            + pn.geom_line(pn.aes(y="relative_success_smooth"), size=1, alpha=0.8)
            + pn.geom_hline(yintercept=1, linetype="dashed")
            + pn.annotate("text", x=_max_limit_plot, y=1 + _nudge, label="Random (RS=1)", size=8, ha="right")
            + pn.annotate("segment", x=_pct1, xend=_pct1, y=0, yend=_rs_mor_plot_max, color="grey", linetype="dotted", size=0.6)
            + pn.annotate("text", x=_pct1, y=_rs_mor_plot_max, label=f"Top 1%\n(n={_pct1})", size=7, ha="center", va="top", color="grey")
            + pn.annotate("segment", x=_pct2, xend=_pct2, y=0, yend=_rs_mor_plot_max, color="grey", linetype="dotted", size=0.6)
            + pn.annotate("text", x=_pct2, y=_rs_mor_plot_max, label=f"Top 2%\n(n={_pct2})", size=7, ha="center", va="top", color="grey")
            + pn.scale_color_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_fill_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_x_continuous(limits=(0, _max_limit_plot))
            + pn.scale_y_continuous(limits=(0, _rs_mor_plot_max), oob=mizani.bounds.squish)
            + pn.labs(x="N top target-disease pairs (per therapeutic area)", y="mean relative success across TAs", color="model", fill="model")
            + pn.theme_minimal()
            # (a) is wider than the boxplot (b). Paired with LaTeX widths of
            # 0.62/0.38 textwidth, the ~1.6x wider aspect keeps the two panels
            # at equal rendered height (8/5 / (5/5) = 1.6 = 0.62/0.38).
            + pn.theme(figure_size=(8, 5))
        )

    _rs_mor_headline = rs_mor[rs_mor["model_slug"].isin(_HEADLINE_SLUGS)].copy()
    _save_plot(
        _build_mor_plot(_rs_mor_headline),
        plots_dir / "relative_success_by_limit_ta_averaged.png",
    )
    _save_plot(
        _build_mor_plot(rs_mor),
        plots_dir / "relative_success_by_limit_ta_averaged_supp.png",
    )

    # Plot 3: Classification metrics by primary TA (boxplot) — stratum='all' only
    cm_all = classification_metrics[classification_metrics["stratum"] == "all"].copy()
    cm_ta = cm_all[
        cm_all["therapeutic_area_name"].isin(primary_therapeutic_areas)
    ].copy()
    cm_ta["model_slug"] = cm_ta["models"].map(model_display)
    cm_ta_melt = cm_ta.melt(
        id_vars=["model_slug", "therapeutic_area_name"], value_vars=["roc_auc", "average_precision", "mcc"],
        var_name="metric", value_name="value"
    )
    def _build_cm_ta_plot(df: pd.DataFrame) -> pn.ggplot:
        _slugs = [s for s in _slug_categories if s in set(df["model_slug"])]
        return (
            pn.ggplot(df, pn.aes(x="model_slug", y="value", fill="model_slug"))
            + pn.geom_violin(alpha=0.55, scale="width", width=0.8, color="none")
            + pn.geom_boxplot(width=0.12, alpha=0.7, outlier_alpha=0.0, color="black")
            + pn.geom_jitter(width=0.12, size=1.0, alpha=0.45, color="black", show_legend=False)
            + pn.facet_wrap("~ metric", scales="free_y", ncol=1)
            + pn.scale_fill_manual(values=slug_colors, breaks=_slugs)
            + pn.labs(x="", y="", fill="model")
            + pn.theme_minimal()
            # Stacked metric facets (ncol=1) keep the panel tall and narrow so it
            # aligns in height with the wider line plot at 0.38 textwidth.
            # Model identity is carried by the fill legend, so drop the redundant
            # model names on the x-axis.
            + pn.theme(figure_size=(5, 5), legend_position="right",
                       axis_text_x=pn.element_blank(),
                       axis_ticks_major_x=pn.element_blank())
        )

    # Headline: the same best-per-family models as the RS plots; full set in supp.
    _save_plot(
        _build_cm_ta_plot(cm_ta_melt[cm_ta_melt["model_slug"].isin(_HEADLINE_SLUGS)].copy()),
        plots_dir / "classification_metrics_by_ta.png",
    )
    _save_plot(
        _build_cm_ta_plot(cm_ta_melt),
        plots_dir / "classification_metrics_by_ta_supp.png",
    )

    # Plot 3b: MCC per model (overall, stratum='all', TA='all')
    cm_overall_mcc = cm_all[cm_all["therapeutic_area_name"] == "all"].copy()
    cm_overall_mcc["model_slug"] = cm_overall_mcc["models"].map(model_display)
    _save_plot(
        pn.ggplot(cm_overall_mcc, pn.aes(x="model_slug", y="mcc", fill="model_slug"))
        + pn.geom_col(alpha=0.85)
        + pn.geom_text(pn.aes(label="mcc"), format_string="{:.3f}", va="bottom", size=8)
        + pn.scale_fill_manual(values=slug_colors, breaks=_slug_categories)
        + pn.labs(x="", y="MCC (best threshold)", fill="model",
                  title="Matthews correlation coefficient per model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 4.5), legend_position="none",
                   axis_text_x=pn.element_text(rotation=40, ha="right")),
        plots_dir / "mcc_by_model.png",
    )

    # Plot 4: Figure 12 — RS heatmap by model × TA (table style)
    # Columns: {model_slug}@{N} for selected limits; rows: TAs; cell = RS.
    _heatmap_limits  = [10, 20, 30, 40, 50, 100]
    # Show every evaluated model in the per-TA heatmap (and the delta-vs-RDG
    # heatmap below). all_model_names is already in canonical legend order
    # (EAHGT variants first, baselines last), so the heatmap columns inherit
    # that ordering and no comparison is silently dropped.
    _heatmap_models = list(all_model_names)

    # Build pivot: rows=TA, cols=(model_slug, limit)
    hm_data = rs_by_ta_primary[
        rs_by_ta_primary["model_name"].isin(_heatmap_models) &
        rs_by_ta_primary["limit"].isin(_heatmap_limits)
    ].copy()

    # Add "average primary therapeutic area" row (mean across primary TAs)
    avg_rows = (
        hm_data
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_success=("relative_success", "mean"))
        .assign(therapeutic_area_name="average primary therapeutic area")
    )
    hm_data = pd.concat([hm_data, avg_rows], ignore_index=True)

    rs_pivot = hm_data.pivot_table(
        index="therapeutic_area_name", columns=["model_name", "limit"],
        values="relative_success", aggfunc="first",
    )

    # Order columns: group by model, then limit
    col_order = [(m, lim) for m in _heatmap_models for lim in _heatmap_limits
                 if (m, lim) in rs_pivot.columns]
    rs_pivot = rs_pivot[col_order]

    # Row order: primary TAs alphabetically, then "average primary therapeutic area"
    ta_order = (
        sorted(t for t in rs_pivot.index
               if t != "average primary therapeutic area")
        + ["average primary therapeutic area"] * ("average primary therapeutic area" in rs_pivot.index)
    )
    rs_pivot = rs_pivot.loc[ta_order]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Per-model colormap (white -> line-plot color) with per-model RS normalization.
    # No legend — intensity is visual only.
    per_model_cmaps = {
        m: mcolors.LinearSegmentedColormap.from_list(
            f"cmap_{m}", ["#ffffff", color_map[m]]
        )
        for m in _heatmap_models
    }
    per_model_norms = {}
    for m in _heatmap_models:
        m_cols = [(mm, lim) for (mm, lim) in rs_pivot.columns if mm == m]
        m_vals = rs_pivot[m_cols].values if m_cols else np.array([])
        m_vals = m_vals[~np.isnan(m_vals)] if m_vals.size else m_vals
        if m_vals.size:
            vmin_m = float(np.nanmin(m_vals))
            vmax_m = float(np.nanmax(m_vals))
            if vmax_m <= vmin_m:
                vmax_m = vmin_m + 1e-6
        else:
            vmin_m, vmax_m = 0.0, 1.0
        per_model_norms[m] = mcolors.Normalize(vmin=vmin_m, vmax=vmax_m)

    def _render_stacked_heatmap(pivot, models, limits, row_order, row_labels,
                                out_path, cell_getter, cell_color, cell_text,
                                ncols_grid=3, group_seps=None):
        """Render one small (row × limit) heatmap panel per model, laid out in a
        grid (ncols_grid columns, wrapping down the page). Keeps each panel only
        `len(limits)` cells wide so the whole figure stacks vertically and fits a
        single page column instead of running off the right edge.

        cell_getter(model, row, limit) -> raw value shown as text (RS).
        cell_color(model, row, limit)  -> RGBA fill for the cell.
        cell_text(model, row, limit)   -> list of (y_frac, string, fontsize) to draw.
        """
        n_panels = len(models)
        ncol = min(ncols_grid, n_panels)
        nrow = int(np.ceil(n_panels / ncol))
        n_r, n_l = len(row_order), len(limits)
        panel_w = n_l * 0.62 + 1.9   # room for the row labels on the left column
        panel_h = n_r * 0.42 + 1.1
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(panel_w * ncol, panel_h * nrow))
        fig.patch.set_facecolor("white")
        for idx in range(nrow * ncol):
            ax = axes[idx // ncol][idx % ncol]
            if idx >= n_panels:
                ax.axis("off")
                continue
            m = models[idx]
            ax.set_aspect("auto")
            for row_i, rk in enumerate(row_order):
                for col_i, lim in enumerate(limits):
                    val = cell_getter(m, rk, lim)
                    ax.add_patch(plt.Rectangle(
                        [col_i, row_i], 1, 1, color=cell_color(m, rk, lim)))
                    if pd.notna(val):
                        for y_frac, s, fs in cell_text(m, rk, lim):
                            ax.text(col_i + 0.5, row_i + y_frac, s, ha="center",
                                    va="center", fontsize=fs, fontweight="bold")
            ax.set_xlim(0, n_l)
            ax.set_ylim(0, n_r)
            ax.set_xticks([i + 0.5 for i in range(n_l)])
            ax.set_xticklabels([f"@{lim}" for lim in limits], fontsize=8)
            # Row (TA/stratum) labels only on the left-most column of the grid.
            if idx % ncol == 0:
                ax.set_yticks([i + 0.5 for i in range(n_r)])
                ax.set_yticklabels(row_labels, fontsize=8)
            else:
                ax.set_yticks([])
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.tick_params(length=0)
            ax.set_title(model_display.get(m, m), fontsize=10, pad=14)
            for gs in (group_seps or []):
                ax.axhline(gs, color="white", linewidth=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.tight_layout()
        logger.info(f"Saving plot to {out_path}")
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _rs_at(pivot, row, model, limit):
        key = (model, limit)
        return pivot.loc[row, key] if (row in pivot.index and key in pivot.columns) else np.nan

    _avg_row_i = (ta_order.index("average primary therapeutic area")
                  if "average primary therapeutic area" in ta_order else None)
    _render_stacked_heatmap(
        rs_pivot, _heatmap_models, _heatmap_limits, ta_order, ta_order,
        plots_dir / "relative_success_by_ta_heatmap.png",
        cell_getter=lambda m, ta, lim: _rs_at(rs_pivot, ta, m, lim),
        cell_color=lambda m, ta, lim: (
            per_model_cmaps[m](per_model_norms[m](_rs_at(rs_pivot, ta, m, lim)))
            if pd.notna(_rs_at(rs_pivot, ta, m, lim)) else (0.85, 0.85, 0.85, 1)),
        cell_text=lambda m, ta, lim: [(0.5, f"{_rs_at(rs_pivot, ta, m, lim):.2f}", 8)],
        group_seps=[_avg_row_i] if _avg_row_i is not None else [],
    )

    # Plot 5d: stratum x model RS heatmap (TA-mean per stratum). Same table
    # style as the per-TA heatmap; rows = evidence/clinical-history strata,
    # cols = model x cutoff. Summarises the numbers in the by-stratum text.
    # Match the 4-stratum set used by the other stratified plots (literature_only
    # and direct_evidence are intentionally excluded — see _stratum_order below).
    _strat_hm_order = [s for s in
                       ["all", "pioneer", "known", "evidence_free"]
                       if s in set(rs_by_limit_full["stratum"])]
    _strat_hm_display = {
        "all": "all pairs",
        "evidence_free": "no prior evidence",
        "literature_only": "text-mining only",
        "direct_evidence": "prior experimental",
        "known": "reached Phase II elsewhere",
        "pioneer": "first-time Phase II",
    }
    shm = rs_by_limit_full[
        rs_by_limit_full["stratum"].isin(_strat_hm_order)
        & rs_by_limit_full["model_name"].isin(_heatmap_models)
        & rs_by_limit_full["limit"].isin(_heatmap_limits)
    ].copy()
    if not shm.empty:
        shm_pivot = shm.pivot_table(
            index="stratum", columns=["model_name", "limit"],
            values="relative_success", aggfunc="first",
        )
        s_col_order = [(m, lim) for m in _heatmap_models for lim in _heatmap_limits
                       if (m, lim) in shm_pivot.columns]
        shm_pivot = shm_pivot[s_col_order]
        s_row_order = [s for s in _strat_hm_order if s in shm_pivot.index]
        shm_pivot = shm_pivot.loc[s_row_order]

        s_norms = {}
        for m in _heatmap_models:
            m_cols = [(mm, lim) for (mm, lim) in shm_pivot.columns if mm == m]
            m_vals = shm_pivot[m_cols].values if m_cols else np.array([])
            m_vals = m_vals[~np.isnan(m_vals)] if m_vals.size else m_vals
            if m_vals.size:
                vmn, vmx = float(np.nanmin(m_vals)), float(np.nanmax(m_vals))
                if vmx <= vmn:
                    vmx = vmn + 1e-6
            else:
                vmn, vmx = 0.0, 1.0
            s_norms[m] = mcolors.Normalize(vmin=vmn, vmax=vmx)

        _render_stacked_heatmap(
            shm_pivot, _heatmap_models, _heatmap_limits, s_row_order,
            [_strat_hm_display.get(s, s) for s in s_row_order],
            plots_dir / "relative_success_by_stratum_heatmap.png",
            cell_getter=lambda m, st, lim: _rs_at(shm_pivot, st, m, lim),
            cell_color=lambda m, st, lim: (
                per_model_cmaps[m](s_norms[m](_rs_at(shm_pivot, st, m, lim)))
                if pd.notna(_rs_at(shm_pivot, st, m, lim)) else (0.85, 0.85, 0.85, 1)),
            cell_text=lambda m, st, lim: [(0.5, f"{_rs_at(shm_pivot, st, m, lim):.2f}", 8)],
        )

    # Plot 6: RS delta vs RDG heatmap (green = better than RDG, red = worse)
    _delta_models = [m for m in _heatmap_models if m != "rdg__no_time__positive"]
    if "rdg__no_time__positive" in all_model_names and _delta_models:
        rdg_rs = rs_pivot["rdg__no_time__positive"] if "rdg__no_time__positive" in rs_pivot.columns.get_level_values(0) else None

        if rdg_rs is not None:
            # Build delta pivot: RS(model) - RS(RDG) for each non-RDG model
            delta_cols = [(m, lim) for m in _delta_models for lim in _heatmap_limits
                          if (m, lim) in rs_pivot.columns]
            delta_pivot = rs_pivot[delta_cols].copy()
            for m, lim in delta_cols:
                if lim in rdg_rs.columns:
                    delta_pivot[(m, lim)] = rs_pivot[(m, lim)] - rdg_rs[lim]

            delta_vals = delta_pivot.values[~np.isnan(delta_pivot.values)]
            abs_max = np.abs(delta_vals).max() if len(delta_vals) > 0 else 1.0
            cmap_d = plt.get_cmap("RdYlGn")
            norm_d = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            def _delta_at(ta, model, limit):
                key = (model, limit)
                return (delta_pivot.loc[ta, key]
                        if (ta in delta_pivot.index and key in delta_pivot.columns)
                        else np.nan)

            def _delta_text(m, ta, lim):
                rs_val = _rs_at(rs_pivot, ta, m, lim)
                if pd.isna(rs_val):
                    return []
                dv = _delta_at(ta, m, lim)
                sign = "+" if pd.notna(dv) and dv >= 0 else ""
                out = [(0.65, f"{rs_val:.2f}", 8)]
                if pd.notna(dv):
                    out.append((0.30, f"{sign}{dv:.2f}", 7))
                return out

            _render_stacked_heatmap(
                delta_pivot, _delta_models, _heatmap_limits, ta_order, ta_order,
                plots_dir / "relative_success_delta_vs_rdg.png",
                cell_getter=lambda m, ta, lim: _rs_at(rs_pivot, ta, m, lim),
                cell_color=lambda m, ta, lim: (
                    cmap_d(norm_d(_delta_at(ta, m, lim)))
                    if pd.notna(_delta_at(ta, m, lim)) else (0.85, 0.85, 0.85, 1)),
                cell_text=_delta_text,
                group_seps=[_avg_row_i] if _avg_row_i is not None else [],
            )

    # Plot 5: RS distributions across primary TAs — violin + mean marker
    # per model, faceted by limit. The mean diamond is what reviewers should
    # read (the headline number); the violin shape shows per-TA spread.
    _dist_limits = [10, 20, 30, 40, 50, 75, 100]
    model_order = [model_display.get(m, m) for m in all_model_names]
    # EAHGT slug = the proposed model. Prefer whatever model in this run maps to
    # the "EAHGT" slug (e.g. the injected ensemble under p3_eahgt_both); fall
    # back to the legacy lambdarank keys.
    p3_slug = ("EAHGT" if "EAHGT" in model_order else
               model_display.get("p3_lambdarank_undirected",
                                  model_display.get("p3_lambdarank_directed", "EAHGT")))
    rdg_slug = model_display.get("rdg__no_time__positive", "RDG")

    rs_dist_df = rs_by_ta_primary[rs_by_ta_primary["limit"].isin(_dist_limits)].copy()
    rs_dist_df["model_slug"] = pd.Categorical(rs_dist_df["model_slug"], categories=model_order, ordered=True)

    # Facet label per limit (no p-values embedded).
    limit_label_order = [f"N={limit}" for limit in _dist_limits]

    rs_dist_df["limit_label"] = rs_dist_df["limit"].map(dict(zip(_dist_limits, limit_label_order)))
    rs_dist_df["limit_label"] = pd.Categorical(rs_dist_df["limit_label"], categories=limit_label_order, ordered=True)

    def _build_rs_violin(df: pd.DataFrame) -> pn.ggplot:
        _slugs = [s for s in _slug_categories if s in set(df["model_slug"])]
        return (
            pn.ggplot(df, pn.aes(x="model_slug", y="relative_success", fill="model_slug"))
            + pn.geom_violin(alpha=0.55, scale="width", width=0.8, color="none")
            + pn.geom_jitter(width=0.12, size=1.0, alpha=0.45, color="black", show_legend=False)
            + pn.stat_summary(fun_data="mean_cl_boot", geom="point",
                              shape="D", size=3.0, color="black", fill="white")
            + pn.geom_hline(yintercept=1, linetype="dashed", color="grey")
            + pn.facet_wrap("~ limit_label", nrow=1, scales="free_y")
            + pn.scale_fill_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_color_manual(values=slug_colors, breaks=_slugs)
            + pn.labs(x="", y="relative success (mean ◇)")
            + pn.theme_minimal()
            + pn.theme(
                figure_size=(3.5 * len(_dist_limits), 5),
                legend_position="none",
                axis_text_x=pn.element_text(rotation=35, ha="right"),
            )
        )

    # Plot 5b: same data as a box plot (median + IQR). Box outliers are
    # suppressed (outlier_alpha=0) and the per-TA points are drawn as a
    # jitter layer instead, matching Plot 5, so each TA is one visible dot.
    def _build_rs_box(df: pd.DataFrame) -> pn.ggplot:
        _slugs = [s for s in _slug_categories if s in set(df["model_slug"])]
        return (
            pn.ggplot(df, pn.aes(x="model_slug", y="relative_success", fill="model_slug"))
            + pn.geom_violin(alpha=0.55, scale="width", width=0.8, color="none")
            + pn.geom_boxplot(width=0.12, alpha=0.7, outlier_alpha=0.0, color="black")
            + pn.geom_jitter(width=0.12, size=1.0, alpha=0.45, color="black", show_legend=False)
            + pn.geom_hline(yintercept=1, linetype="dashed", color="grey")
            + pn.facet_wrap("~ limit_label", nrow=1, scales="free_y")
            + pn.scale_fill_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_color_manual(values=slug_colors, breaks=_slugs)
            + pn.labs(x="", y="relative success (per-TA median, IQR)", fill="model")
            + pn.theme_minimal()
            + pn.theme(
                figure_size=(3.5 * len(_dist_limits), 5),
                legend_position="right",
                axis_text_x=pn.element_blank(),
                axis_ticks_major_x=pn.element_blank(),
            )
        )

    # Headline: proposed model + variants + OTS / RDG; full set in the _supp twin.
    _rs_dist_headline = rs_dist_df[rs_dist_df["model_slug"].isin(_HEADLINE_SLUGS)].copy()
    _save_plot(_build_rs_violin(_rs_dist_headline), plots_dir / "rs_distributions_ta.png")
    _save_plot(_build_rs_violin(rs_dist_df),        plots_dir / "rs_distributions_ta_supp.png")
    _save_plot(_build_rs_box(_rs_dist_headline),    plots_dir / "rs_distributions_ta_box.png")
    _save_plot(_build_rs_box(rs_dist_df),           plots_dir / "rs_distributions_ta_box_supp.png")

    # ------------------------------------------------------------------
    # Stratified plots (novelty × evidence sparsity)
    # ------------------------------------------------------------------
    # literature_only ("Prior text-mining evidence only") and direct_evidence
    # ("Prior experimental evidence") are intentionally excluded from the
    # stratified plots; dropping them here removes them from every downstream
    # stratum plot/label.
    _stratum_order = [
        "all", "pioneer", "known",
        "evidence_free",
    ]
    _stratum_n = {
        s: (len(strata_test) if pair_strata[s] is None else len(pair_strata[s]))
        for s in _stratum_order
    }
    _stratum_display = {
        "all":             "All test pairs",
        "pioneer":         "Target entering Phase 2 for the first time across any disease",
        "known":           "Target has reached Phase 2 in another disease context",
        "evidence_free":   "No prior target–disease evidence",
        "literature_only": "Prior text-mining evidence only",
        "direct_evidence": "Prior experimental evidence",
    }
    _stratum_label = {
        s: "\n".join(textwrap.wrap(f"{_stratum_display[s]} (n={_stratum_n[s]})", width=30))
        for s in _stratum_order
    }
    _stratum_label_order = [_stratum_label[s] for s in _stratum_order]

    # Stratified classification metrics violin across primary TAs (stratum × TA)
    cm_strat = classification_metrics[
        classification_metrics["therapeutic_area_name"].isin(primary_therapeutic_areas)
        & classification_metrics["stratum"].isin(_stratum_order)
    ].copy()
    cm_strat["model_slug"] = cm_strat["models"].map(model_display)
    cm_strat["stratum_label"] = cm_strat["stratum"].map(_stratum_label)
    cm_strat["stratum_label"] = pd.Categorical(
        cm_strat["stratum_label"], categories=_stratum_label_order, ordered=True
    )
    cm_strat_melt = cm_strat.melt(
        id_vars=["model_slug", "stratum_label", "therapeutic_area_name"],
        value_vars=["roc_auc", "average_precision", "mcc"],
        var_name="metric", value_name="value",
    )
    _save_plot(
        pn.ggplot(cm_strat_melt, pn.aes(x="model_slug", y="value", fill="model_slug"))
        + pn.geom_violin(alpha=0.55, scale="width", width=0.8, color="none")
        + pn.geom_boxplot(width=0.12, alpha=0.7, outlier_alpha=0.0, color="black")
        + pn.geom_jitter(width=0.12, size=1.2, alpha=0.7, color="black", show_legend=False)
        + pn.facet_grid("metric ~ stratum_label", scales="free_y")
        + pn.scale_fill_manual(values=slug_colors, breaks=_slug_categories)
        + pn.labs(x="", y="", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(14, 7),
                   axis_text_x=pn.element_text(rotation=40, ha="right"),
                   legend_position="none"),
        plots_dir / "classification_metrics_by_stratum_by_ta.png",
    )

    # Stratified RS-by-limit line chart — TA-mean (grouped) RS per stratum, i.e.
    # mean over primary TAs within each stratum (NOT the pooled top-N). Uses the
    # grouped/TA-mean metric the EAHGT model is trained and reported on, matching
    # the headline ..._ta_averaged figure rather than the pooled curve.
    rs_strat = rs_by_limit_full[rs_by_limit_full["stratum"].isin(_stratum_order)].copy()
    rs_strat["stratum_label"] = rs_strat["stratum"].map(_stratum_label)
    rs_strat["stratum_label"] = pd.Categorical(
        rs_strat["stratum_label"], categories=_stratum_label_order, ordered=True
    )
    # Smooth per (model, stratum) with the same rolling mean as the headline
    # TA-mean figure (3a), so the stratum lines read the same way.
    rs_strat = rs_strat.sort_values(["stratum", "model_slug", "limit"])
    rs_strat["relative_success_smooth"] = (
        rs_strat.groupby(["stratum", "model_slug"])["relative_success"]
        .transform(lambda s: s.rolling(window=5, center=True, min_periods=1).mean())
    )
    def _build_stratum_plot(rs_df: pd.DataFrame) -> pn.ggplot:
        _slugs = [s for s in _slug_categories if s in set(rs_df["model_slug"])]
        return (
            # Match Figure 3a: faint raw points + smoothed line, dashed Random line.
            pn.ggplot(rs_df, pn.aes(x="limit", y="relative_success", color="model_slug", group="model_slug"))
            + pn.geom_point(alpha=0.3, size=1, show_legend=False)
            + pn.geom_line(pn.aes(y="relative_success_smooth"), size=1, alpha=0.8)
            + pn.geom_hline(yintercept=1, linetype="dashed")
            + pn.facet_wrap("~ stratum_label", ncol=2, scales="free_y")
            + pn.scale_color_manual(values=slug_colors, breaks=_slugs)
            + pn.scale_x_continuous(breaks=np.arange(10, 101, 20).tolist())
            + pn.scale_y_continuous(limits=(0, None))
            + pn.labs(x="N top target-disease pairs (per therapeutic area)",
                      y="mean relative success across TAs", color="model")
            + pn.theme_minimal()
            + pn.theme(figure_size=(10, 8))
        )

    # Headline: proposed model + variants + OTS / RDG (evidence-sparse recovery
    # story). Full model set relegated to the _supp twin.
    _save_plot(
        _build_stratum_plot(rs_strat[rs_strat["model_slug"].isin(_HEADLINE_SLUGS)].copy()),
        plots_dir / "relative_success_by_limit_by_stratum.png",
    )
    _save_plot(
        _build_stratum_plot(rs_strat),
        plots_dir / "relative_success_by_limit_by_stratum_supp.png",
    )

    focal_model = next(
        (m for m in ["p3_lambdarank_undirected", "p3_lambdarank_directed", *external_model_names] if m in all_model_names),
        all_model_names[-1] if all_model_names else None,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(f"\nAll results saved to {results_dir}")
    overall = cm_all[cm_all["therapeutic_area_name"] == "all"].set_index("models")[["roc_auc", "average_precision", "mcc"]].sort_values("roc_auc", ascending=False)
    print("\n=== Classification Metrics (overall) ===")
    print(overall.round(4).to_string())
    rs50 = rs_by_limit[rs_by_limit["limit"] == 50][["model_name", "relative_success"]].sort_values("relative_success", ascending=False)
    print("\n=== Relative Success @ limit=50 ===")
    print(rs50.round(3).to_string(index=False))

    if focal_model is not None:
        strat_auc = (
            classification_metrics[
                (classification_metrics["models"] == focal_model)
                & (classification_metrics["therapeutic_area_name"] == "all")
            ]
            .set_index("stratum")[["roc_auc", "average_precision", "mcc", "n_samples"]]
            .reindex(_stratum_order)
            .dropna(how="all")
        )
        print(f"\n=== ROC-AUC by stratum ({model_display.get(focal_model, focal_model)}) ===")
        print(strat_auc.round(4).to_string())


# ---------------------------------------------------------------------------
# Temporal-inflation analysis (edge-count deltas around advancement)
# ---------------------------------------------------------------------------
#
# Companion to the relative-success view in Figure 7. Splits each TD pair's
# evidence timeline at its first advancement-edge year and counts edges of
# each evidence type before vs after that split. Uses the event graph's own
# advancement edges as the split source -- no external phase table needed.
#
# Reused helpers: _load_graph_and_mappings, _save_plot, the bootstrap pattern
# at lines ~729-746 (over therapeutic areas), and the per-relation groupby
# pattern from _compute_evidence_sparsity.

_INFLATION_EVIDENCE_RELATIONS = (
    "literature",
    "genetic_association",
    "somatic_mutation",
    "affected_pathway",
    "animal_model",
    "rna_expression",
)


def _compute_advancement_split_points(data, mappings) -> pd.DataFrame:
    """Per-pair split points derived from advancement edges of the event graph.

    Columns: target_id, disease_id, t_adv_first, t_adv_last, advanced.
    advanced=1 iff any advancement edge for the pair has edge_weight==1.
    """
    node_mapping = mappings["node_mapping"]
    idx_to_target  = {v: k for k, v in node_mapping["target"].items()}
    idx_to_disease = {v: k for k, v in node_mapping["disease"].items()}

    adv = data[_ADV_ETYPE]
    src = adv.edge_index[0].cpu().numpy()
    dst = adv.edge_index[1].cpu().numpy()
    t   = adv.edge_time.cpu().numpy()
    # Outcome is stored in edge_attr[:, 0] (0 = not advanced, 1 = advanced) — not in edge_weight.
    if not (hasattr(adv, "edge_attr") and adv.edge_attr is not None):
        raise ValueError("Advancement edges have no edge_attr; cannot recover outcome label.")
    w = adv.edge_attr[:, 0].cpu().numpy()

    df = pd.DataFrame({
        "target_id":  [idx_to_target[i]  for i in src],
        "disease_id": [idx_to_disease[i] for i in dst],
        "t":          t,
        "w":          w,
    })
    grouped = df.groupby(["target_id", "disease_id"], as_index=False).agg(
        t_adv_first=("t", "min"),
        t_adv_last=("t", "max"),
        advanced=("w", lambda s: int((s >= 0.5).any())),
    )
    return grouped


def _compute_evidence_edge_counts(
    data,
    mappings,
    split_points: pd.DataFrame,
    evidence_types: tuple = _INFLATION_EVIDENCE_RELATIONS,
    split_kind: str = "first",
) -> pd.DataFrame:
    """Per-pair before/after edge counts and rates, long-form.

    One row per (pair, evidence_type). Joined against split_points so each pair
    has its own split T.

    Columns: target_id, disease_id, advanced, evidence_type, split_kind,
             T, n_before, n_after, delta, rate_before, rate_after, rate_delta,
             rate_clamped.
    """
    if split_kind not in ("first", "last"):
        raise ValueError(f"split_kind must be 'first' or 'last', got {split_kind}")
    t_col = "t_adv_first" if split_kind == "first" else "t_adv_last"

    node_mapping = mappings["node_mapping"]
    target_to_idx  = node_mapping["target"]
    disease_to_idx = node_mapping["disease"]

    # Map pair string IDs to (t_idx, d_idx) integer keys once.
    sp_df = split_points.copy()
    sp_df["t_idx"] = sp_df["target_id"].map(target_to_idx)
    sp_df["d_idx"] = sp_df["disease_id"].map(disease_to_idx)
    sp_df = sp_df.dropna(subset=["t_idx", "d_idx"]).copy()
    sp_df["t_idx"] = sp_df["t_idx"].astype(np.int64)
    sp_df["d_idx"] = sp_df["d_idx"].astype(np.int64)
    sp_df["T"] = sp_df[t_col].astype(np.int64)

    rows = []
    for relation in evidence_types:
        etype = ("target", relation, "disease")
        if etype not in data.edge_types:
            logger.warning(f"Evidence relation '{relation}' not present in graph; skipping.")
            continue
        store = data[etype]
        n_e = store.edge_index.size(1)
        if n_e == 0:
            continue
        src = store.edge_index[0].cpu().numpy()
        dst = store.edge_index[1].cpu().numpy()
        if hasattr(store, "edge_time") and store.edge_time is not None:
            times = store.edge_time.cpu().numpy()
        else:
            # Treat static edges as always-present (before any T).
            times = np.full(n_e, -10**9, dtype=np.int64)

        edges = pd.DataFrame({"t_idx": src, "d_idx": dst, "year": times})
        t_min_e = int(edges["year"].min())
        t_max_e = int(edges["year"].max())

        # Join edges to this pair-universe and count before/after via groupby.
        joined = edges.merge(sp_df[["t_idx", "d_idx", "T"]], on=["t_idx", "d_idx"], how="inner")
        if joined.empty:
            # No edges of this type touch any advancement pair.
            for _, r in sp_df.iterrows():
                rows.append({
                    "target_id": r["target_id"], "disease_id": r["disease_id"],
                    "advanced": int(r["advanced"]), "evidence_type": relation,
                    "split_kind": split_kind, "T": int(r["T"]),
                    "n_before": 0, "n_after": 0, "delta": 0,
                    "rate_before": 0.0, "rate_after": 0.0, "rate_delta": 0.0,
                    "rate_clamped": False,
                })
            continue
        joined["before"] = (joined["year"] < joined["T"]).astype(np.int64)
        joined["after"] = (joined["year"] >= joined["T"]).astype(np.int64)
        counts = joined.groupby(["t_idx", "d_idx"], as_index=False).agg(
            n_before=("before", "sum"),
            n_after=("after", "sum"),
        )
        merged = sp_df.merge(counts, on=["t_idx", "d_idx"], how="left").fillna(
            {"n_before": 0, "n_after": 0}
        )
        merged["n_before"] = merged["n_before"].astype(np.int64)
        merged["n_after"]  = merged["n_after"].astype(np.int64)
        merged["delta"] = merged["n_after"] - merged["n_before"]

        before_window = (merged["T"] - t_min_e).clip(lower=1)
        after_window  = (t_max_e - merged["T"] + 1).clip(lower=1)
        merged["rate_before"] = merged["n_before"] / before_window
        merged["rate_after"]  = merged["n_after"]  / after_window
        merged["rate_delta"]  = merged["rate_after"] - merged["rate_before"]
        merged["rate_clamped"] = (merged["T"] - t_min_e) <= 0

        for _, r in merged.iterrows():
            rows.append({
                "target_id": r["target_id"], "disease_id": r["disease_id"],
                "advanced": int(r["advanced"]), "evidence_type": relation,
                "split_kind": split_kind, "T": int(r["T"]),
                "n_before": int(r["n_before"]), "n_after": int(r["n_after"]),
                "delta": int(r["delta"]),
                "rate_before": float(r["rate_before"]),
                "rate_after":  float(r["rate_after"]),
                "rate_delta":  float(r["rate_delta"]),
                "rate_clamped": bool(r["rate_clamped"]),
            })

    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 2000,
                  seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat(values)` resampled with replacement."""
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _summarise_edge_count_groups(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Group-wise summary: mean/median/IQR with bootstrap CIs + paired/unpaired tests.

    Aggregates per (evidence_type, split_kind, advanced). Also emits an
    "advanced=both" row carrying the within-pair Wilcoxon and the MWU between
    advanced groups.
    """
    rows = []
    for (ev, sk), grp_ev in counts_df.groupby(["evidence_type", "split_kind"]):
        for adv_val, grp in grp_ev.groupby("advanced"):
            n_after  = grp["n_after"].to_numpy()
            n_before = grp["n_before"].to_numpy()
            delta    = grp["delta"].to_numpy()
            rd       = grp["rate_delta"].to_numpy()

            mean_lo, mean_hi = _bootstrap_ci(rd, np.mean)
            med_lo,  med_hi  = _bootstrap_ci(rd, lambda a, axis=None: np.median(a, axis=axis))

            # Within-group paired test on (n_after - n_before).
            if delta.size >= 5 and np.any(delta != 0):
                try:
                    wstat, wpval = wilcoxon(n_after, n_before, zero_method="wilcox", alternative="greater")
                except ValueError:
                    wstat, wpval = float("nan"), float("nan")
            else:
                wstat, wpval = float("nan"), float("nan")

            rows.append({
                "evidence_type": ev, "split_kind": sk, "advanced": int(adv_val),
                "n_pairs": int(grp.shape[0]),
                "mean_n_before": float(np.mean(n_before)),
                "mean_n_after":  float(np.mean(n_after)),
                "median_n_before": float(np.median(n_before)),
                "median_n_after":  float(np.median(n_after)),
                "iqr_delta_lo": float(np.percentile(delta, 25)),
                "iqr_delta_hi": float(np.percentile(delta, 75)),
                "mean_rate_delta": float(np.mean(rd)),
                "mean_rate_delta_ci_lo": mean_lo,
                "mean_rate_delta_ci_hi": mean_hi,
                "median_rate_delta": float(np.median(rd)),
                "median_rate_delta_ci_lo": med_lo,
                "median_rate_delta_ci_hi": med_hi,
                "wilcoxon_stat": float(wstat),
                "wilcoxon_p_greater": float(wpval),
            })

        # Between-group MWU on rate_delta (advanced=1 vs advanced=0).
        adv1 = grp_ev.loc[grp_ev["advanced"] == 1, "rate_delta"].to_numpy()
        adv0 = grp_ev.loc[grp_ev["advanced"] == 0, "rate_delta"].to_numpy()
        if adv1.size >= 5 and adv0.size >= 5:
            try:
                ustat, upval = mannwhitneyu(adv1, adv0, alternative="greater")
            except ValueError:
                ustat, upval = float("nan"), float("nan")
        else:
            ustat, upval = float("nan"), float("nan")
        rows.append({
            "evidence_type": ev, "split_kind": sk, "advanced": -1,  # marker for "between groups"
            "n_pairs": int(adv0.size + adv1.size),
            "mean_n_before": float("nan"), "mean_n_after": float("nan"),
            "median_n_before": float("nan"), "median_n_after": float("nan"),
            "iqr_delta_lo": float("nan"), "iqr_delta_hi": float("nan"),
            "mean_rate_delta": float("nan"),
            "mean_rate_delta_ci_lo": float("nan"), "mean_rate_delta_ci_hi": float("nan"),
            "median_rate_delta": float("nan"),
            "median_rate_delta_ci_lo": float("nan"), "median_rate_delta_ci_hi": float("nan"),
            "wilcoxon_stat": float("nan"), "wilcoxon_p_greater": float("nan"),
            "mwu_stat": float(ustat), "mwu_p_greater": float(upval),
        })

    return pd.DataFrame(rows)


def _build_homogeneous_adjacency(data, year: int):
    """Undirected adjacency over all nodes (target ∪ disease ∪ others) at snapshot `year`.

    Returns (csr_matrix, node_offsets, total_nodes).
    node_offsets[ntype] = starting row index for that node type's local indices.
    Edge contributions from every edge type with edge_time <= year are included
    (static edges always included).
    """
    node_types = list(data.node_types)
    sizes = {nt: int(data[nt].num_nodes) for nt in node_types}
    offsets, running = {}, 0
    for nt in node_types:
        offsets[nt] = running
        running += sizes[nt]
    total = running

    rows_idx, cols_idx = [], []
    for et in data.edge_types:
        src_nt, _, dst_nt = et
        store = data[et]
        n_e = store.edge_index.size(1)
        if n_e == 0:
            continue
        if hasattr(store, "edge_time") and store.edge_time is not None:
            t = store.edge_time.cpu().numpy()
            mask = t <= year
            if not mask.any():
                continue
            ei = store.edge_index.cpu().numpy()[:, mask]
        else:
            ei = store.edge_index.cpu().numpy()
        s = ei[0] + offsets[src_nt]
        d = ei[1] + offsets[dst_nt]
        rows_idx.append(s); cols_idx.append(d)
        # Undirected projection: add the reverse too.
        rows_idx.append(d); cols_idx.append(s)

    if not rows_idx:
        return sp.csr_matrix((total, total)), offsets, total
    rr = np.concatenate(rows_idx); cc = np.concatenate(cols_idx)
    vv = np.ones(rr.size, dtype=np.float32)
    A = sp.coo_matrix((vv, (rr, cc)), shape=(total, total)).tocsr()
    # Collapse multi-edges to 1 to avoid weighting by literature density.
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data = np.minimum(A.data, 1.0)
    return A, offsets, total


def _pagerank_sparse(A: sp.csr_matrix, damping: float = 0.85,
                     tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Power-iteration PageRank on an undirected adjacency (binary weights)."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_safe = np.where(deg > 0, deg, 1.0)
    # P[j,i] = A[i,j] / deg[i]  -> matrix-vector form: r_new = damping * (A^T diag(1/deg)) r + (1-damping)/n
    inv_deg = 1.0 / deg_safe
    # Normalised transition: scale rows of A by inv_deg, then transpose for left-mul on r.
    D_inv = sp.diags(inv_deg)
    M = (D_inv @ A).T.tocsr()  # column-stochastic transition

    r = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n
    dangling_mask = deg == 0
    for _ in range(max_iter):
        # Dangling mass: redistribute uniformly.
        dangling = damping * r[dangling_mask].sum() / n
        r_new = damping * (M @ r) + teleport + dangling
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r


def _compute_snapshot_centrality(data, years: list[int]) -> dict:
    """{year: {"target": pr_target_vec, "disease": pr_disease_vec,
                "deg_target": deg_t_vec, "deg_disease": deg_d_vec}}.

    Computes once per distinct year. PageRank/degree are over the full
    homogenised projection, then sliced back per node type via offsets.
    """
    out = {}
    for y in sorted(set(int(v) for v in years)):
        logger.info(f"  centrality snapshot year={y}")
        A, offsets, total = _build_homogeneous_adjacency(data, y)
        deg = np.asarray(A.sum(axis=1)).ravel()
        pr  = _pagerank_sparse(A)
        sl_t = slice(offsets["target"], offsets["target"] + int(data["target"].num_nodes))
        sl_d = slice(offsets["disease"], offsets["disease"] + int(data["disease"].num_nodes))
        out[y] = {
            "deg_target":  deg[sl_t],
            "deg_disease": deg[sl_d],
            "pr_target":   pr[sl_t],
            "pr_disease":  pr[sl_d],
        }
    return out


def _summarise_centrality_groups(split_points: pd.DataFrame,
                                 mappings,
                                 centrality_by_year: dict,
                                 split_kind: str = "first") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_pair_df, group_summary_df).

    per_pair_df: target_id, disease_id, advanced, T, deg_target, deg_disease,
                 pr_target, pr_disease.
    group_summary_df: per metric × advanced — median, bootstrap CI, MWU vs other group.
    """
    target_to_idx  = mappings["node_mapping"]["target"]
    disease_to_idx = mappings["node_mapping"]["disease"]
    t_col = "t_adv_first" if split_kind == "first" else "t_adv_last"

    rows = []
    for r in split_points.itertuples(index=False):
        ti = target_to_idx.get(r.target_id)
        di = disease_to_idx.get(r.disease_id)
        if ti is None or di is None:
            continue
        y = int(getattr(r, t_col))
        c = centrality_by_year.get(y)
        if c is None:
            continue
        rows.append({
            "target_id": r.target_id, "disease_id": r.disease_id,
            "advanced": int(r.advanced), "T": y,
            "deg_target":  float(c["deg_target"][ti]),
            "deg_disease": float(c["deg_disease"][di]),
            "pr_target":   float(c["pr_target"][ti]),
            "pr_disease":  float(c["pr_disease"][di]),
        })
    per_pair = pd.DataFrame(rows)
    if per_pair.empty:
        return per_pair, pd.DataFrame()

    metrics = ["deg_target", "deg_disease", "pr_target", "pr_disease"]
    sum_rows = []
    for m in metrics:
        adv1 = per_pair.loc[per_pair["advanced"] == 1, m].to_numpy()
        adv0 = per_pair.loc[per_pair["advanced"] == 0, m].to_numpy()
        for label, vec in (("advanced=1", adv1), ("advanced=0", adv0)):
            if vec.size == 0:
                continue
            med_lo, med_hi = _bootstrap_ci(vec, lambda a, axis=None: np.median(a, axis=axis))
            sum_rows.append({
                "metric": m, "group": label, "n": int(vec.size),
                "median": float(np.median(vec)),
                "median_ci_lo": med_lo, "median_ci_hi": med_hi,
            })
        if adv1.size >= 5 and adv0.size >= 5:
            try:
                ustat, upval = mannwhitneyu(adv1, adv0, alternative="greater")
            except ValueError:
                ustat, upval = float("nan"), float("nan")
        else:
            ustat, upval = float("nan"), float("nan")
        sum_rows.append({
            "metric": m, "group": "advanced=1_vs_0", "n": int(adv1.size + adv0.size),
            "median": float("nan"), "median_ci_lo": float("nan"), "median_ci_hi": float("nan"),
            "mwu_stat": float(ustat), "mwu_p_greater": float(upval),
        })
    return per_pair, pd.DataFrame(sum_rows)


def inflation_analysis(
    graph_file: str = _GRAPH_FILE,
    mappings_file: str = _MAPPINGS_FILE,
    out_dir: str = f"{Path(__file__).parent}/advancement_data/results/inflation",
    split: str = "first",
    compute_centrality: bool = True,
):
    """Edge-count temporal-inflation analysis around advancement edges.

    For each (target, disease) pair with at least one advancement edge, splits
    the pair's evidence timeline at its first (default) or last advancement
    year and counts edges of each evidence type before/after. Outputs
    per-pair counts, group summary, and (optional) node-centrality
    distributional comparison.

    Args:
        graph_file: hetero graph .pt with advancement edges
        mappings_file: node-mapping .pt
        out_dir: output directory (created if missing)
        split: 'first' (default) or 'last' — which advancement year to split at
        compute_centrality: whether to compute the PageRank/degree comparison
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading graph from {graph_file}")
    data, mappings = _load_graph_and_mappings(Path(graph_file), Path(mappings_file))

    logger.info("Computing advancement split points...")
    split_points = _compute_advancement_split_points(data, mappings)
    n_total = len(split_points)
    n_adv1  = int((split_points["advanced"] == 1).sum())
    n_adv0  = int((split_points["advanced"] == 0).sum())
    logger.info(f"  pairs with advancement edges: {n_total} (advanced=1: {n_adv1}, advanced=0: {n_adv0})")
    if n_adv0 < 50:
        logger.warning(
            f"advanced=0 group has only {n_adv0} pairs (advancement edges with edge_weight<0.5). "
            "Between-group comparisons will be underpowered."
        )

    logger.info(f"Computing edge counts before/after split (kind='{split}')...")
    counts_df = _compute_evidence_edge_counts(
        data, mappings, split_points, split_kind=split,
    )
    counts_path = out_dir / "inflation_pair_counts.parquet"
    counts_df.to_parquet(counts_path, index=False)
    logger.info(f"  wrote {counts_path} ({len(counts_df)} rows)")

    logger.info("Summarising group statistics...")
    summary_df = _summarise_edge_count_groups(counts_df)
    summary_path = out_dir / "inflation_group_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  wrote {summary_path}")

    # Print headline numbers so the verification expectations are visible.
    print("\n=== Edge-count inflation summary (per evidence type, advanced=1 group) ===")
    head_cols = [
        "evidence_type", "n_pairs", "mean_n_before", "mean_n_after",
        "mean_rate_delta", "mean_rate_delta_ci_lo", "mean_rate_delta_ci_hi",
        "wilcoxon_p_greater",
    ]
    headline = summary_df[(summary_df["advanced"] == 1) & (summary_df["split_kind"] == split)][head_cols]
    print(headline.round(4).to_string(index=False))

    # ---- Plots --------------------------------------------------------
    logger.info("Generating plots...")
    plot_df = counts_df[counts_df["split_kind"] == split].copy()
    plot_df["advanced_lbl"] = plot_df["advanced"].map({0: "not advanced", 1: "advanced"})
    long_counts = plot_df.melt(
        id_vars=["target_id", "disease_id", "evidence_type", "advanced_lbl"],
        value_vars=["n_before", "n_after"],
        var_name="window", value_name="n_edges",
    )
    long_counts["window"] = long_counts["window"].map({"n_before": "before", "n_after": "after"})

    _save_plot(
        pn.ggplot(long_counts, pn.aes(x="window", y="n_edges + 1", fill="advanced_lbl"))
        + pn.geom_boxplot(outlier_alpha=0.2)
        + pn.facet_wrap("~evidence_type", scales="free_y")
        + pn.scale_y_log10()
        + pn.labs(
            x="window relative to first advancement edge",
            y="edge count + 1 (log scale)",
            fill="pair group",
            title="Per-pair evidence edge counts before vs after advancement",
        )
        + pn.theme_minimal()
        + pn.theme(figure_size=(11, 6)),
        plots_dir / "inflation_counts_by_evidence.png",
    )

    _save_plot(
        pn.ggplot(plot_df, pn.aes(x="advanced_lbl", y="rate_delta", fill="advanced_lbl"))
        + pn.geom_violin(alpha=0.5)
        + pn.geom_boxplot(width=0.15, outlier_alpha=0.2)
        + pn.geom_hline(yintercept=0, linetype="dashed")
        + pn.facet_wrap("~evidence_type", scales="free_y")
        + pn.labs(
            x="pair group",
            y="rate(after) − rate(before)  (edges/year)",
            fill="pair group",
            title="Post-advancement edge-rate uptick by evidence type",
        )
        + pn.theme_minimal()
        + pn.theme(figure_size=(11, 6)),
        plots_dir / "inflation_rate_delta_by_evidence.png",
    )

    # ---- Centrality (optional, slower) --------------------------------
    if compute_centrality:
        logger.info("Computing snapshot centrality (degree + PageRank)...")
        years_needed = split_points["t_adv_first" if split == "first" else "t_adv_last"].astype(int).unique().tolist()
        logger.info(f"  {len(years_needed)} distinct snapshot years")
        centrality = _compute_snapshot_centrality(data, years_needed)
        per_pair_c, group_c = _summarise_centrality_groups(
            split_points, mappings, centrality, split_kind=split,
        )
        per_pair_c.to_parquet(out_dir / "inflation_centrality_pairs.parquet", index=False)
        group_c.to_csv(out_dir / "inflation_centrality.csv", index=False)
        logger.info(f"  wrote inflation_centrality.csv ({len(group_c)} rows)")

        if not per_pair_c.empty:
            cent_long = per_pair_c.melt(
                id_vars=["target_id", "disease_id", "advanced"],
                value_vars=["deg_target", "deg_disease", "pr_target", "pr_disease"],
                var_name="metric", value_name="value",
            )
            cent_long["advanced_lbl"] = cent_long["advanced"].map({0: "not advanced", 1: "advanced"})
            cent_long = cent_long[cent_long["value"] > 0]
            _save_plot(
                pn.ggplot(cent_long, pn.aes(x="advanced_lbl", y="value", fill="advanced_lbl"))
                + pn.geom_violin(alpha=0.5)
                + pn.geom_boxplot(width=0.15, outlier_alpha=0.2)
                + pn.facet_wrap("~metric", scales="free_y")
                + pn.scale_y_log10()
                + pn.labs(
                    x="pair group",
                    y="value (log scale)",
                    fill="pair group",
                    title="Node centrality at advancement year",
                )
                + pn.theme_minimal()
                + pn.theme(figure_size=(10, 6)),
                plots_dir / "inflation_centrality.png",
            )

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire({
        "evaluate": evaluate,
        "inflation_analysis": inflation_analysis,
    })

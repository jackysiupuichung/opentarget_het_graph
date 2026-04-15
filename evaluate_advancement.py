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
import numpy as np
import pandas as pd
import xarray as xr
import plotnine as pn
import fire
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats.contingency import relative_risk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------
_SCRATCH = "/gpfs/scratch/bty414/opentarget_evidences/23.06/runs"

# DEFAULT_RUNS: dict[str, str] = {
#     # Ablation baselines
#     # "b1_hgt":                f"{_SCRATCH}/b1_hgt_datatype",
#     # "b2_hgt_rte":            f"{_SCRATCH}/b2_hgt_rte_datatype",
#     # "b3_gatv2_score":        f"{_SCRATCH}/b3_gatv2_score_datatype",
#     # "b4_gatv2_novelty":      f"{_SCRATCH}/b4_gatv2_novelty_datatype",
#     # "b5_gatv2_both":         f"{_SCRATCH}/b5_gatv2_both_datatype",
#     # EA-HGT ablations
#     # "p1_eahgt_score":        f"{_SCRATCH}/p1_eahgt_score_datatype",
#     # "p2_eahgt_novelty":      f"{_SCRATCH}/p2_eahgt_novelty_datatype",
#     "p3_eahgt_both_ap@50":         f"{_SCRATCH}/p3_eahgt_both_datatype",
#     "p3_validation": f"{_SCRATCH}/p3_tuned_with_val",
#     "p3_test":       f"{_SCRATCH}/p3_tuned_with_test",
# }


DEFAULT_RUNS: dict[str, str] = {
    # Ablation baselines
    # "b1_hgt":          f"{_SCRATCH}/b1_hgt_datatype",
    # "b2_hgt_rte":      f"{_SCRATCH}/b2_hgt_rte_datatype",
    # "b3_gatv2_score":  f"{_SCRATCH}/b3_gatv2_score_datatype",
    # "b4_gatv2_novelty": f"{_SCRATCH}/b4_gatv2_novelty_datatype",
    # "b5_gatv2_both":   f"{_SCRATCH}/b5_gatv2_both_datatype",
    # # EA-HGT ablations
    # "p1_eahgt_score":   f"{_SCRATCH}/p1_eahgt_score_datatype",
    # "p2_eahgt_novelty": f"{_SCRATCH}/p2_eahgt_novelty_datatype",
    # "p3_eahgt_both":    f"{_SCRATCH}/p3_tuned_with_test",
    # LambdaRank is the active p3 representative
    "p3_lambdarank": "runs/advancement_lambdarank",
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BASE_MODEL_COLORS = {
    "RDG":   "#1f77b4",
    "RDG-T": "#aec7e8",
    "OTS":   "#2ca02c",
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

    runs = dict(DEFAULT_RUNS)
    if only is not None:
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
    """ROC AUC and average precision per model per TA."""
    from sklearn.metrics import roc_auc_score, average_precision_score

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
            rows.append({
                "models": model,
                "therapeutic_area_name": ta,
                "roc_auc": roc_auc_score(y, s),
                "average_precision": average_precision_score(y, s),
                "n_samples": len(y),
                "n_positives": y.sum(),
            })
    return pd.DataFrame(rows)


def _compute_relative_risk_by_limit(evaluation_dataset: xr.Dataset,
                                     model_names: list[str],
                                     limits: list[int],
                                     confidence: float = 0.9,
                                     disease_filter: set | None = None) -> pd.DataFrame:
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
        for limit in limits:
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
                "relative_risk": rr.relative_risk,
                "relative_risk_low": ci.low,
                "relative_risk_high": ci.high,
                "n_exposed": len(exposed),
                "n_exposed_true": int(exposed["outcome"].sum()),
                "n_control": len(control),
            })
    return pd.DataFrame(rows)


def _compute_rr_by_ta(evaluation_dataset: xr.Dataset,
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
                threshold = ta_grp["score"].nlargest(limit).min()
                exposed = ta_grp[ta_grp["score"] >= threshold]
                control = ta_grp[ta_grp["score"] < threshold]
                if len(exposed) == 0 or len(control) == 0:
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
                    "relative_risk": rr.relative_risk if not np.isinf(rr.relative_risk) else np.nan,
                    "relative_risk_low": ci.low,
                    "relative_risk_high": ci.high,
                    "n_exposed": len(exposed),
                    "n_exposed_true": int(exposed["outcome"].sum()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_plot(plot: pn.ggplot, path: Path) -> None:
    logger.info(f"Saving plot to {path}")
    plot = plot + pn.theme(plot_background=pn.element_rect(fill="white"))
    plot.save(str(path), verbose=False)


def _build_color_map(model_names: list[str]) -> dict:
    base = {
        "rdg__all__positive": _BASE_MODEL_COLORS["RDG"],
        "rdg__no_time__positive": _BASE_MODEL_COLORS["RDG-T"],
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
    graph_file: str = f"{Path(__file__).parent}/output/graph/hetero_graph_with_advancement.pt",
    mappings_file: str = f"{Path(__file__).parent}/output/graph/temporal_graph_mappings.pt",
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
    base_models = [m for m in ["rdg__all__positive", "rdg__no_time__positive", "ots__all"]
                   if m in evaluation_dataset.coords["models"].values]
    all_model_names = base_models + external_model_names

    model_display = {
        "rdg__all__positive":     "RDG-T",
        "rdg__no_time__positive": "RDG",
        "ots__all":               "OTS",
        **{n: n.upper().replace("__", "-")[:12] for n in external_model_names},
    }
    color_map  = _build_color_map(all_model_names)
    slug_colors = {model_display.get(m, m): color_map[m] for m in all_model_names}

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
    # 3 & 4. Relative risk by TA (primary TAs), then average for by-limit plot
    # ------------------------------------------------------------------
    _primary_ta_set = set(primary_therapeutic_areas) - {"all"}
    logger.info("Computing relative risk by therapeutic area across %d strata...", len(pair_strata))
    rr_frames = []
    for stratum, pair_filter in pair_strata.items():
        if pair_filter is not None and len(pair_filter) == 0:
            continue
        rr_s = _compute_rr_by_ta(
            evaluation_dataset,
            therapeutic_areas,
            all_model_names,
            limits=np.arange(10, 101, 10).tolist(),
            pair_filter=pair_filter,
        )
        rr_s["stratum"] = stratum
        rr_frames.append(rr_s)
    rr_by_ta_full = pd.concat(rr_frames, ignore_index=True)
    # Existing plots use the "all" subset
    rr_by_ta = rr_by_ta_full[rr_by_ta_full["stratum"] == "all"].drop(columns=["stratum"]).copy()
    rr_by_ta["model_slug"] = rr_by_ta["model_name"].map(model_display)

    # Derive rr_by_limit as mean RR across primary TAs (matching reference paper)
    logger.info("Computing relative risk by limit (mean over primary TAs)...")
    rr_by_ta_primary = rr_by_ta[rr_by_ta["therapeutic_area_name"].isin(_primary_ta_set)].copy()
    rr_by_limit = (
        rr_by_ta_primary
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_risk=("relative_risk", "mean"))
    )

    # Stratified rr_by_limit (mean over primary TAs per stratum)
    rr_by_ta_full["model_slug"] = rr_by_ta_full["model_name"].map(model_display)
    rr_by_ta_primary_full = rr_by_ta_full[
        rr_by_ta_full["therapeutic_area_name"].isin(_primary_ta_set)
    ].copy()
    rr_by_limit_full = (
        rr_by_ta_primary_full
        .groupby(["model_name", "model_slug", "limit", "stratum"], as_index=False)
        .agg(relative_risk=("relative_risk", "mean"))
    )

    # Save CSVs: primary TAs + average row appended (stratified)
    avg_ta_rows = rr_by_limit_full.assign(therapeutic_area_name="average primary therapeutic area")
    pd.concat([rr_by_ta_primary_full, avg_ta_rows], ignore_index=True).to_csv(
        results_dir / "relative_risk_by_ta.csv", index=False
    )
    rr_by_limit_full.to_csv(results_dir / "relative_risk_by_limit.csv", index=False)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots...")

    # Plot 1: RR vs limit (mean over primary TAs, matching reference paper)
    _rr_max = int(np.ceil(rr_by_limit["relative_risk"].max()))
    _save_plot(
        pn.ggplot(rr_by_limit, pn.aes(x="limit", y="relative_risk", color="model_slug", group="model_slug"))
        + pn.geom_line(size=1, alpha=0.8)
        + pn.geom_hline(yintercept=1, linetype="dashed")
        + pn.scale_color_manual(values=slug_colors)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.scale_x_continuous(breaks=np.arange(10, 101, 10).tolist())
        + pn.scale_y_continuous(breaks=np.arange(1, _rr_max + 1, 1).tolist(),
                                limits=(1, _rr_max))
        + pn.labs(x="N top target-disease pairs", y="relative risk", color="model", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 4)),
        plots_dir / "relative_risk_by_limit.png",
    )

    # Plot 2: Classification metrics bar chart (stratum='all' only for legacy plot)
    cm_all = classification_metrics[classification_metrics["stratum"] == "all"].copy()
    cm_overall = cm_all[cm_all["therapeutic_area_name"] == "all"].copy()
    cm_overall["model_slug"] = cm_overall["models"].map(model_display)
    cm_melt = cm_overall.melt(
        id_vars=["model_slug"], value_vars=["roc_auc", "average_precision"],
        var_name="metric", value_name="value"
    )
    _save_plot(
        pn.ggplot(cm_melt, pn.aes(x="model_slug", y="value", fill="model_slug"))
        + pn.geom_col(position="dodge")
        + pn.facet_wrap("~ metric", scales="free_y", ncol=1)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="", fill="model")
        + pn.coord_flip()
        + pn.theme_minimal()
        + pn.theme(figure_size=(7, 4), legend_position="none"),
        plots_dir / "classification_metrics_overall.png",
    )

    # Plot 3: Classification metrics by primary TA (boxplot) — stratum='all' only
    cm_ta = cm_all[
        cm_all["therapeutic_area_name"].isin(primary_therapeutic_areas)
    ].copy()
    cm_ta["model_slug"] = cm_ta["models"].map(model_display)
    cm_ta_melt = cm_ta.melt(
        id_vars=["model_slug", "therapeutic_area_name"], value_vars=["roc_auc", "average_precision"],
        var_name="metric", value_name="value"
    )
    _save_plot(
        pn.ggplot(cm_ta_melt, pn.aes(x="model_slug", y="value", fill="model_slug"))
        + pn.geom_boxplot(outlier_size=1, alpha=0.6)
        + pn.geom_jitter(width=0.15, size=1.5, alpha=0.7)
        + pn.facet_wrap("~ metric", scales="free_y", ncol=1)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="", fill="model")
        + pn.coord_flip()
        + pn.theme_minimal()
        + pn.theme(figure_size=(7, 5), legend_position="none"),
        plots_dir / "classification_metrics_by_ta.png",
    )

    # Plot 4: Figure 12 — RR heatmap by model × TA (table style)
    # Columns: {model_slug}@{N} for selected limits; rows: TAs; cell = RR (n_pairs)
    _heatmap_limits  = [10, 50, 100]
    _heatmap_models  = [
        m for m in ["rdg__all__positive", "ots__all", "p3_lambdarank", "rdg__no_time__positive"]
        if m in all_model_names
    ]

    # Build pivot: rows=TA, cols=(model_slug, limit)
    hm_data = rr_by_ta_primary[
        rr_by_ta_primary["model_name"].isin(_heatmap_models) &
        rr_by_ta_primary["limit"].isin(_heatmap_limits)
    ].copy()

    # Add "average primary therapeutic area" row (mean across primary TAs)
    avg_rows = (
        hm_data
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_risk=("relative_risk", "mean"), n_exposed=("n_exposed", "mean"))
        .assign(therapeutic_area_name="average primary therapeutic area")
    )
    hm_data = pd.concat([hm_data, avg_rows], ignore_index=True)

    # Pivot RR and n_exposed separately
    rr_pivot = hm_data.pivot_table(
        index="therapeutic_area_name", columns=["model_name", "limit"],
        values="relative_risk", aggfunc="first",
    )
    np_pivot = hm_data.pivot_table(
        index="therapeutic_area_name", columns=["model_name", "limit"],
        values="n_exposed", aggfunc="first",
    )

    # Order columns: group by model, then limit
    col_order = [(m, lim) for m in _heatmap_models for lim in _heatmap_limits
                 if (m, lim) in rr_pivot.columns]
    rr_pivot = rr_pivot[col_order]
    np_pivot = np_pivot[col_order]

    # Row order: primary TAs alphabetically, then "average primary therapeutic area"
    ta_order = (
        sorted(t for t in rr_pivot.index
               if t != "average primary therapeutic area")
        + ["average primary therapeutic area"] * ("average primary therapeutic area" in rr_pivot.index)
    )
    rr_pivot = rr_pivot.loc[ta_order]
    np_pivot = np_pivot.loc[ta_order]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_rows, n_cols = rr_pivot.shape
    fig_h = max(5, n_rows * 0.45 + 1.5)
    fig_w = max(6, n_cols * 1.2 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("auto")

    cmap  = plt.get_cmap("RdYlGn")
    vmin, vmax = 0.5, rr_pivot.values[~np.isnan(rr_pivot.values)].max() if rr_pivot.notna().any().any() else 3.0
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for row_i, ta in enumerate(ta_order):
        for col_i, (model_name, limit) in enumerate(col_order):
            rr_val = rr_pivot.loc[ta, (model_name, limit)] if (model_name, limit) in rr_pivot.columns else np.nan
            np_val = np_pivot.loc[ta, (model_name, limit)] if (model_name, limit) in np_pivot.columns else np.nan
            color  = cmap(norm(rr_val)) if pd.notna(rr_val) else (0.85, 0.85, 0.85, 1)
            rect   = plt.Rectangle([col_i, row_i], 1, 1, color=color)
            ax.add_patch(rect)
            if pd.notna(rr_val):
                ax.text(col_i + 0.5, row_i + 0.62, f"{rr_val:.2f}",
                        ha="center", va="center", fontsize=8, fontweight="bold")
                if pd.notna(np_val):
                    ax.text(col_i + 0.5, row_i + 0.28, f"n={int(np_val)}",
                            ha="center", va="center", fontsize=6.5, color="#333333")

    # Axis labels
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(
        [f"{model_display.get(m, m)}@{lim}" for m, lim in col_order],
        rotation=35, ha="right", fontsize=9,
    )
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(ta_order, fontsize=9)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)

    # Separator lines between model groups
    for sep in range(len(_heatmap_limits), n_cols, len(_heatmap_limits)):
        ax.axvline(sep, color="white", linewidth=2)
    # Separator before "average primary therapeutic area" row
    if "average primary therapeutic area" in ta_order:
        sep_row = ta_order.index("average primary therapeutic area")
        ax.axhline(sep_row, color="white", linewidth=2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Relative risk")
    ax.set_title("Relative risk by method and therapeutic area", pad=30, fontsize=11)
    fig.tight_layout()

    out_path = plots_dir / "relative_risk_by_ta_heatmap.png"
    logger.info(f"Saving plot to {out_path}")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Plot 6: RR delta vs RDG heatmap (green = better than RDG, red = worse)
    _delta_models = [m for m in _heatmap_models if m != "rdg__no_time__positive"]
    if "rdg__no_time__positive" in all_model_names and _delta_models:
        rdg_rr = rr_pivot["rdg__no_time__positive"] if "rdg__no_time__positive" in rr_pivot.columns.get_level_values(0) else None

        if rdg_rr is not None:
            # Build delta pivot: RR(model) - RR(RDG) for each non-RDG model
            delta_cols = [(m, lim) for m in _delta_models for lim in _heatmap_limits
                          if (m, lim) in rr_pivot.columns]
            delta_pivot = rr_pivot[delta_cols].copy()
            for m, lim in delta_cols:
                if lim in rdg_rr.columns:
                    delta_pivot[(m, lim)] = rr_pivot[(m, lim)] - rdg_rr[lim]

            n_rows_d, n_cols_d = delta_pivot.shape
            fig_h_d = max(5, n_rows_d * 0.45 + 1.5)
            fig_w_d = max(6, n_cols_d * 1.2 + 2.5)
            fig_d, ax_d = plt.subplots(figsize=(fig_w_d, fig_h_d))
            ax_d.set_aspect("auto")

            delta_vals = delta_pivot.values[~np.isnan(delta_pivot.values)]
            abs_max = np.abs(delta_vals).max() if len(delta_vals) > 0 else 1.0
            cmap_d = plt.get_cmap("RdYlGn")
            norm_d = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

            for row_i, ta in enumerate(ta_order):
                for col_i, (model_name, limit) in enumerate(delta_cols):
                    rr_val    = rr_pivot.loc[ta, (model_name, limit)] if (model_name, limit) in rr_pivot.columns else np.nan
                    delta_val = delta_pivot.loc[ta, (model_name, limit)] if (model_name, limit) in delta_pivot.columns else np.nan
                    color = cmap_d(norm_d(delta_val)) if pd.notna(delta_val) else (0.85, 0.85, 0.85, 1)
                    rect  = plt.Rectangle([col_i, row_i], 1, 1, color=color)
                    ax_d.add_patch(rect)
                    if pd.notna(rr_val):
                        sign = "+" if pd.notna(delta_val) and delta_val >= 0 else ""
                        delta_str = f"{sign}{delta_val:.2f}" if pd.notna(delta_val) else ""
                        ax_d.text(col_i + 0.5, row_i + 0.65, f"{rr_val:.2f}",
                                  ha="center", va="center", fontsize=8, fontweight="bold")
                        ax_d.text(col_i + 0.5, row_i + 0.30, delta_str,
                                  ha="center", va="center", fontsize=7, color="#111111")

            ax_d.set_xlim(0, n_cols_d)
            ax_d.set_ylim(0, n_rows_d)
            ax_d.set_xticks([i + 0.5 for i in range(n_cols_d)])
            ax_d.set_xticklabels(
                [f"{model_display.get(m, m)}@{lim}" for m, lim in delta_cols],
                rotation=35, ha="right", fontsize=9,
            )
            ax_d.set_yticks([i + 0.5 for i in range(n_rows_d)])
            ax_d.set_yticklabels(ta_order, fontsize=9)
            ax_d.invert_yaxis()
            ax_d.xaxis.tick_top()
            ax_d.xaxis.set_label_position("top")
            ax_d.tick_params(length=0)

            for sep in range(len(_heatmap_limits), n_cols_d, len(_heatmap_limits)):
                ax_d.axvline(sep, color="white", linewidth=2)
            if "average primary therapeutic area" in ta_order:
                sep_row = ta_order.index("average primary therapeutic area")
                ax_d.axhline(sep_row, color="white", linewidth=2)

            sm_d = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
            sm_d.set_array([])
            plt.colorbar(sm_d, ax=ax_d, shrink=0.6, label="RR delta vs RDG")
            ax_d.set_title("Relative risk vs RDG (green = better, red = worse)", pad=30, fontsize=11)
            fig_d.tight_layout()

            out_path_d = plots_dir / "relative_risk_delta_vs_rdg.png"
            logger.info(f"Saving plot to {out_path_d}")
            fig_d.savefig(str(out_path_d), dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig_d)

    # Plot 5: Figure 11 — one subplot per RR@N, dots+boxplot per model, p-value per subplot
    _dist_limits = [10, 20, 50, 100]
    rdg_slug = model_display.get("rdg__no_time__positive", "RDG")
    ots_slug = model_display.get("ots__all", "OTS")
    model_order = [model_display.get(m, m) for m in all_model_names]
    rng = np.random.default_rng(42)

    n_subplots = len(_dist_limits)
    fig, axes = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 5), sharey=True)
    if n_subplots == 1:
        axes = [axes]

    for ax, limit in zip(axes, _dist_limits):
        grp = rr_by_ta_primary[rr_by_ta_primary["limit"] == limit].copy()
        valid_slugs = [s for s in model_order if s in grp["model_slug"].values]

        box_data = [grp[grp["model_slug"] == s]["relative_risk"].dropna().values for s in valid_slugs]
        bp = ax.boxplot(box_data, positions=range(len(valid_slugs)), widths=0.5,
                        patch_artist=True, zorder=2, showfliers=False)
        for patch, slug in zip(bp["boxes"], valid_slugs):
            patch.set_facecolor(slug_colors.get(slug, "#aaaaaa"))
            patch.set_alpha(0.4)
        for element in ["whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(0.8)

        for i, slug in enumerate(valid_slugs):
            vals = grp[grp["model_slug"] == slug]["relative_risk"].dropna().values
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(i + jitter, vals, color=slug_colors.get(slug, "#aaaaaa"),
                       s=25, zorder=3, alpha=0.85)

        ax.axhline(1, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xticks(range(len(valid_slugs)))
        ax.set_xticklabels(valid_slugs, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"RR@{limit}", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Relative risk")

        # Wilcoxon p-value (p3 > RDG) for this limit
        p3_slug = model_display.get("p3_lambdarank", "p3_lambdarank")
        p3_v  = grp[grp["model_slug"] == p3_slug].set_index("therapeutic_area_name")["relative_risk"]
        rdg_v = grp[grp["model_slug"] == rdg_slug].set_index("therapeutic_area_name")["relative_risk"]
        paired = pd.concat([p3_v, rdg_v], axis=1, keys=["p3", "rdg"]).dropna()
        if len(paired) >= 3 and (paired["p3"] - paired["rdg"]).abs().sum() > 0:
            _, pval = wilcoxon(paired["p3"], paired["rdg"], alternative="greater")
            pval_str = f"p={pval:.3f}" if pval >= 0.001 else f"p={pval:.2e}"
            ax.text(0.5, -0.28, f"Wilcoxon {pval_str}\n(p3 > RDG)",
                    ha="center", va="top", transform=ax.transAxes, fontsize=7.5, color="#333333")

    fig.suptitle("Relative risk distributions across select therapeutic areas", fontsize=11, y=1.02)
    fig.tight_layout()

    out_path = plots_dir / "rr_distributions_ta.png"
    logger.info(f"Saving plot to {out_path}")
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Stratified plots (novelty × evidence sparsity)
    # ------------------------------------------------------------------
    _stratum_order = [
        "all", "pioneer", "known",
        "evidence_free", "literature_only", "direct_evidence",
        "pioneer__evidence_free", "pioneer__literature_only", "pioneer__direct_evidence",
        "known__evidence_free",   "known__literature_only",   "known__direct_evidence",
    ]

    # Stratified classification metrics bar chart (TA == 'all' rows only)
    cm_strat = classification_metrics[classification_metrics["therapeutic_area_name"] == "all"].copy()
    cm_strat["model_slug"] = cm_strat["models"].map(model_display)
    cm_strat["stratum"] = pd.Categorical(cm_strat["stratum"], categories=_stratum_order, ordered=True)
    cm_strat_melt = cm_strat.melt(
        id_vars=["model_slug", "stratum"], value_vars=["roc_auc", "average_precision"],
        var_name="metric", value_name="value",
    )
    _save_plot(
        pn.ggplot(cm_strat_melt, pn.aes(x="stratum", y="value", fill="model_slug"))
        + pn.geom_col(position="dodge")
        + pn.facet_wrap("~ metric", ncol=1, scales="free_y")
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(10, 6), axis_text_x=pn.element_text(rotation=40, ha="right")),
        plots_dir / "classification_metrics_by_stratum.png",
    )

    # Stratified RR-by-limit line chart (mean over primary TAs per stratum)
    rr_strat = rr_by_limit_full.copy()
    rr_strat["stratum"] = pd.Categorical(rr_strat["stratum"], categories=_stratum_order, ordered=True)
    _save_plot(
        pn.ggplot(rr_strat, pn.aes(x="limit", y="relative_risk", color="model_slug", group="model_slug"))
        + pn.geom_line(size=0.9, alpha=0.85)
        + pn.geom_hline(yintercept=1, linetype="dashed")
        + pn.facet_wrap("~ stratum", ncol=3, scales="free_y")
        + pn.scale_color_manual(values=slug_colors)
        + pn.scale_x_continuous(breaks=np.arange(10, 101, 20).tolist())
        + pn.labs(x="N top target-disease pairs", y="relative risk", color="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(13, 10)),
        plots_dir / "relative_risk_by_limit_by_stratum.png",
    )

    # 2×3 risk matrix heatmap: ROC-AUC for a focal model (prefer p3_lambdarank)
    focal_model = next(
        (m for m in ["p3_lambdarank", *external_model_names] if m in all_model_names),
        all_model_names[-1] if all_model_names else None,
    )
    if focal_model is not None:
        cell_rows = ["pioneer", "known"]
        cell_cols = ["evidence_free", "literature_only", "direct_evidence"]
        auc_mat = np.full((2, 3), np.nan)
        n_mat   = np.zeros((2, 3), dtype=int)

        # n per cell from strata_test
        for i, nov in enumerate(cell_rows):
            for j, ev in enumerate(cell_cols):
                sub = strata_test[
                    (strata_test["novelty_stratum"] == nov)
                    & (strata_test["evidence_stratum"] == ev)
                ]
                n_mat[i, j] = len(sub)

        # ROC-AUC per cell from classification_metrics (TA=='all', focal model)
        cm_focal = classification_metrics[
            (classification_metrics["models"] == focal_model)
            & (classification_metrics["therapeutic_area_name"] == "all")
        ].set_index("stratum")
        for i, nov in enumerate(cell_rows):
            for j, ev in enumerate(cell_cols):
                key = f"{nov}__{ev}"
                if key in cm_focal.index:
                    auc_mat[i, j] = cm_focal.loc[key, "roc_auc"]

        fig_rm, ax_rm = plt.subplots(figsize=(7, 3.5))
        cmap_rm = plt.get_cmap("RdYlGn")
        finite = auc_mat[np.isfinite(auc_mat)]
        vmin = float(finite.min()) if finite.size else 0.5
        vmax = float(finite.max()) if finite.size else 1.0
        if vmin == vmax:
            vmin, vmax = vmin - 0.01, vmax + 0.01
        norm_rm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(2):
            for j in range(3):
                auc_val = auc_mat[i, j]
                color = cmap_rm(norm_rm(auc_val)) if np.isfinite(auc_val) else (0.85, 0.85, 0.85, 1)
                ax_rm.add_patch(plt.Rectangle([j, i], 1, 1, color=color))
                label = f"AUC={auc_val:.3f}" if np.isfinite(auc_val) else "—"
                ax_rm.text(j + 0.5, i + 0.62, label, ha="center", va="center",
                           fontsize=10, fontweight="bold")
                ax_rm.text(j + 0.5, i + 0.30, f"n={n_mat[i, j]}", ha="center", va="center",
                           fontsize=8.5, color="#222222")
        ax_rm.set_xlim(0, 3)
        ax_rm.set_ylim(0, 2)
        ax_rm.set_xticks([0.5, 1.5, 2.5])
        ax_rm.set_xticklabels(cell_cols, fontsize=10)
        ax_rm.set_yticks([0.5, 1.5])
        ax_rm.set_yticklabels(cell_rows, fontsize=10)
        ax_rm.invert_yaxis()
        ax_rm.xaxis.tick_top()
        ax_rm.tick_params(length=0)
        ax_rm.set_title(f"Risk matrix — ROC-AUC ({model_display.get(focal_model, focal_model)})",
                        pad=25, fontsize=11)
        fig_rm.tight_layout()
        rm_path = plots_dir / "risk_matrix_heatmap.png"
        logger.info(f"Saving plot to {rm_path}")
        fig_rm.savefig(str(rm_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig_rm)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(f"\nAll results saved to {results_dir}")
    overall = cm_overall.set_index("models")[["roc_auc", "average_precision"]].sort_values("roc_auc", ascending=False)
    print("\n=== Classification Metrics (overall) ===")
    print(overall.round(4).to_string())
    rr50 = rr_by_limit[rr_by_limit["limit"] == 50][["model_name", "relative_risk"]].sort_values("relative_risk", ascending=False)
    print("\n=== Relative Risk @ limit=50 ===")
    print(rr50.round(3).to_string(index=False))

    if focal_model is not None:
        strat_auc = (
            classification_metrics[
                (classification_metrics["models"] == focal_model)
                & (classification_metrics["therapeutic_area_name"] == "all")
            ]
            .set_index("stratum")[["roc_auc", "average_precision", "n_samples"]]
            .reindex(_stratum_order)
            .dropna(how="all")
        )
        print(f"\n=== ROC-AUC by stratum ({model_display.get(focal_model, focal_model)}) ===")
        print(strat_auc.round(4).to_string())

        risk_df = pd.DataFrame(auc_mat, index=cell_rows, columns=cell_cols)
        print(f"\n=== Risk matrix (ROC-AUC, {model_display.get(focal_model, focal_model)}) ===")
        print(risk_df.round(4).to_string())


if __name__ == "__main__":
    fire.Fire(evaluate)

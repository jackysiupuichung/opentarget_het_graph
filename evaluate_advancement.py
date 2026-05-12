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
import textwrap
import numpy as np
import pandas as pd
import xarray as xr
import plotnine as pn
import mizani.bounds
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


DEFAULT_RUNS: dict[str, str] = {
    # p3_lambdarank: directed graph (canonical)
    "p3_lambdarank_directed": "runs/advancement_lambdarank",
    # Undirected experiments (same base HP as v1, varying dropout and early-stop/ndcg_k)
    "undirected_v1": f"{_SCRATCH}/advancement_lambdarank_undirected_v1",  # dropout=0.2, ndcg_k=50,  metric=ndcg@10
    "unidirected_additive": f"{_SCRATCH}/advancement_lambdarank_undirected_v1_additive",  # dropout=0.2, ndcg_k=50,  metric=ndcg@10
    "undirected_v2": f"{_SCRATCH}/advancement_lambdarank_undirected_v2",  # dropout=0.3, ndcg_k=50,  metric=ndcg@50
    "undirected_v3": f"{_SCRATCH}/advancement_lambdarank_undirected_v3",  # dropout=0.3, ndcg_k=100, metric=ndcg@100
    "undirected_v4": f"{_SCRATCH}/advancement_lambdarank_undirected_v4",  # dropout=0.4, ndcg_k=50,  metric=ndcg@50
    # Study A — Model comparison (no edge features; per-encoder tuned)
    "b1_hgt":           f"{_SCRATCH}/b1_hgt_lambdarank_v2",
    "b3_gatv2":         f"{_SCRATCH}/b3_gatv2_lambdarank_v2",
    "b6_rgcn":          f"{_SCRATCH}/b6_rgcn_lambdarank_v2",
    "b7_compgcn":       f"{_SCRATCH}/b7_compgcn_lambdarank_v2",
    # Study B — EAHGT ablation (HGT + edge feature variants; canonical HPs)
    "p1_eahgt_score":   f"{_SCRATCH}/p1_eahgt_score_lambdarank_v2",
    "p2_eahgt_novelty": f"{_SCRATCH}/p2_eahgt_novelty_lambdarank_v2",
    "p3_eahgt_both":    f"{_SCRATCH}/p3_eahgt_both_lambdarank_v2",
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
                    "relative_risk": rr.relative_risk if not np.isinf(rr.relative_risk) else np.nan,
                    "relative_risk_low": ci.low,
                    "relative_risk_high": ci.high,
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
        plot_background=pn.element_rect(fill="white"),
        panel_background=pn.element_rect(fill="white"),
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
    }
    model_display = {
        **_STATIC_DISPLAY,
        **{n: n.upper().replace("__", "-")[:12]
           for n in external_model_names
           if n not in _STATIC_DISPLAY},
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

    # Pooled RR with 95% Katz CI — fine-grained limits for the paper-style line plot
    _n_pairs_total = int(
        evaluation_dataset.outcome.squeeze("outcomes").to_series().reset_index().shape[0]
    )
    _fine_limits = sorted(set(
        list(range(10, 50))
        + list(range(50, 200, 5))
        + list(range(200, 501, 10))
    ))
    logger.info("Computing pooled RR with 95%% Katz CI over %d limits...", len(_fine_limits))
    rr_pooled = _compute_relative_risk_by_limit(
        evaluation_dataset, all_model_names, _fine_limits, confidence=0.95
    )
    rr_pooled["model_slug"] = rr_pooled["model_name"].map(model_display)
    rr_pooled = rr_pooled.sort_values(["model_slug", "limit"])
    rr_pooled["relative_risk_smooth"] = (
        rr_pooled.groupby("model_slug")["relative_risk"]
        .transform(lambda s: s.rolling(window=9, center=True, min_periods=1).mean())
    )

    # Mean-of-ratios (TA-averaged) at fine limits for overlay on katz plot
    logger.info("Computing mean-of-ratios RR over fine limits for katz overlay...")
    rr_by_ta_fine_frames = []
    for stratum, pair_filter in [("all", None)]:
        rr_fine_s = _compute_rr_by_ta(
            evaluation_dataset, therapeutic_areas, all_model_names,
            limits=_fine_limits, pair_filter=pair_filter,
        )
        rr_by_ta_fine_frames.append(rr_fine_s)
    rr_by_ta_fine = pd.concat(rr_by_ta_fine_frames, ignore_index=True)
    rr_by_ta_fine["model_slug"] = rr_by_ta_fine["model_name"].map(model_display)
    rr_mor = (
        rr_by_ta_fine[rr_by_ta_fine["therapeutic_area_name"].isin(_primary_ta_set)]
        .groupby(["model_name", "model_slug", "limit"], as_index=False)
        .agg(relative_risk=("relative_risk", "mean"))
    )
    rr_mor = rr_mor.sort_values(["model_slug", "limit"])
    rr_mor["relative_risk_smooth"] = (
        rr_mor.groupby("model_slug")["relative_risk"]
        .transform(lambda s: s.rolling(window=9, center=True, min_periods=1).mean())
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

    # Plot 1: Mean-of-ratios line+points with 95% Katz CI ribbon
    # rr_pooled provides the CI band only; rr_mor drives the line and points.
    _pct1  = max(1, round(_n_pairs_total * 0.01))
    _pct2  = max(1, round(_n_pairs_total * 0.02))
    _max_limit_plot = 250
    _eahgt_slug = "EAHGT" if "EAHGT" in set(rr_mor["model_slug"]) else model_display.get(
        "p3_lambdarank_undirected", model_display.get("p3_lambdarank_directed", "EAHGT"))
    _rr_mor_katz = rr_mor[rr_mor["model_slug"] == _eahgt_slug].copy()
    _rr_pooled_katz = rr_pooled[rr_pooled["model_slug"] == _eahgt_slug].copy()
    _rr_plot_max = min(6.0, float(_rr_pooled_katz["relative_risk_high"].max()))
    _nudge = 0.15
    _save_plot(
        pn.ggplot(rr_mor, pn.aes(x="limit", y="relative_risk", color="model_slug", fill="model_slug", group="model_slug"))
        + pn.geom_ribbon(
            data=_rr_pooled_katz,
            mapping=pn.aes(x="limit", ymin="relative_risk_low", ymax="relative_risk_high", fill="model_slug", group="model_slug"),
            outline_type="full", alpha=0.22, color=None,
        )
        + pn.geom_line(
            data=_rr_pooled_katz,
            mapping=pn.aes(x="limit", y="relative_risk_low", color="model_slug", group="model_slug"),
            size=0.4, linetype="dotted", show_legend=False,
        )
        + pn.geom_line(
            data=_rr_pooled_katz,
            mapping=pn.aes(x="limit", y="relative_risk_high", color="model_slug", group="model_slug"),
            size=0.4, linetype="dotted", show_legend=False,
        )
        + pn.geom_point(alpha=0.3, size=1)
        + pn.geom_line(pn.aes(y="relative_risk_smooth"), size=1, alpha=0.8)
        + pn.geom_hline(yintercept=1, linetype="dashed")
        + pn.annotate("text", x=_max_limit_plot, y=1 + _nudge, label="Random (RR=1)", size=8, ha="right")
        + pn.annotate("segment", x=_pct1, xend=_pct1, y=0, yend=_rr_plot_max, color="grey", linetype="dotted", size=0.6)
        + pn.annotate("text", x=_pct1, y=_rr_plot_max, label=f"Top 1%\n(n={_pct1})", size=7, ha="center", va="top", color="grey")
        + pn.annotate("segment", x=_pct2, xend=_pct2, y=0, yend=_rr_plot_max, color="grey", linetype="dotted", size=0.6)
        + pn.annotate("text", x=_pct2, y=_rr_plot_max, label=f"Top 2%\n(n={_pct2})", size=7, ha="center", va="top", color="grey")
        + pn.scale_color_manual(values=slug_colors)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.scale_x_continuous(limits=(0, _max_limit_plot))
        + pn.scale_y_continuous(limits=(0, _rr_plot_max), oob=mizani.bounds.squish)
        + pn.labs(x="N top target-disease pairs", y="relative risk", color="model", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(10, 4)),
        plots_dir / "relative_risk_by_limit_katz95.png",
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
    _save_plot(
        pn.ggplot(cm_ta_melt, pn.aes(x="model_slug", y="value", fill="model_slug"))
        + pn.geom_boxplot(outlier_size=1, alpha=0.6)
        + pn.geom_jitter(width=0.15, size=1.5, alpha=0.7)
        + pn.facet_wrap("~ metric", scales="free_y", ncol=3)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(10, 5), legend_position="none",
                   axis_text_x=pn.element_text(rotation=40, ha="right")),
        plots_dir / "classification_metrics_by_ta.png",
    )

    # Plot 3b: MCC per model (overall, stratum='all', TA='all')
    cm_overall_mcc = cm_all[cm_all["therapeutic_area_name"] == "all"].copy()
    cm_overall_mcc["model_slug"] = cm_overall_mcc["models"].map(model_display)
    _save_plot(
        pn.ggplot(cm_overall_mcc, pn.aes(x="model_slug", y="mcc", fill="model_slug"))
        + pn.geom_col(alpha=0.85)
        + pn.geom_text(pn.aes(label="mcc"), format_string="{:.3f}", va="bottom", size=8)
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="MCC (best threshold)", fill="model",
                  title="Matthews correlation coefficient per model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(8, 4.5), legend_position="none",
                   axis_text_x=pn.element_text(rotation=40, ha="right")),
        plots_dir / "mcc_by_model.png",
    )

    # Plot 4: Figure 12 — RR heatmap by model × TA (table style)
    # Columns: {model_slug}@{N} for selected limits; rows: TAs; cell = RR.
    _heatmap_limits  = [10, 20, 30, 40, 50, 100]
    _heatmap_models  = [
        m for m in ["ots__all", "rdg__no_time__positive",
                    "p3_lambdarank_directed",
                    "undirected_v1", "undirected_v2",
                    "undirected_v3", "undirected_v4"]
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
        .agg(relative_risk=("relative_risk", "mean"))
        .assign(therapeutic_area_name="average primary therapeutic area")
    )
    hm_data = pd.concat([hm_data, avg_rows], ignore_index=True)

    rr_pivot = hm_data.pivot_table(
        index="therapeutic_area_name", columns=["model_name", "limit"],
        values="relative_risk", aggfunc="first",
    )

    # Order columns: group by model, then limit
    col_order = [(m, lim) for m in _heatmap_models for lim in _heatmap_limits
                 if (m, lim) in rr_pivot.columns]
    rr_pivot = rr_pivot[col_order]

    # Row order: primary TAs alphabetically, then "average primary therapeutic area"
    ta_order = (
        sorted(t for t in rr_pivot.index
               if t != "average primary therapeutic area")
        + ["average primary therapeutic area"] * ("average primary therapeutic area" in rr_pivot.index)
    )
    rr_pivot = rr_pivot.loc[ta_order]

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_rows, n_cols = rr_pivot.shape
    fig_h = max(5, n_rows * 0.45 + 1.5)
    fig_w = max(6, n_cols * 1.2 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_aspect("auto")

    # Per-model colormap (white -> line-plot color) with per-model RR normalization.
    # No legend — intensity is visual only.
    per_model_cmaps = {
        m: mcolors.LinearSegmentedColormap.from_list(
            f"cmap_{m}", ["#ffffff", color_map[m]]
        )
        for m in _heatmap_models
    }
    per_model_norms = {}
    for m in _heatmap_models:
        m_cols = [(mm, lim) for (mm, lim) in rr_pivot.columns if mm == m]
        m_vals = rr_pivot[m_cols].values if m_cols else np.array([])
        m_vals = m_vals[~np.isnan(m_vals)] if m_vals.size else m_vals
        if m_vals.size:
            vmin_m = float(np.nanmin(m_vals))
            vmax_m = float(np.nanmax(m_vals))
            if vmax_m <= vmin_m:
                vmax_m = vmin_m + 1e-6
        else:
            vmin_m, vmax_m = 0.0, 1.0
        per_model_norms[m] = mcolors.Normalize(vmin=vmin_m, vmax=vmax_m)

    for row_i, ta in enumerate(ta_order):
        for col_i, (model_name, limit) in enumerate(col_order):
            rr_val = rr_pivot.loc[ta, (model_name, limit)] if (model_name, limit) in rr_pivot.columns else np.nan
            cmap_m = per_model_cmaps[model_name]
            norm_m = per_model_norms[model_name]
            color  = cmap_m(norm_m(rr_val)) if pd.notna(rr_val) else (0.85, 0.85, 0.85, 1)
            rect   = plt.Rectangle([col_i, row_i], 1, 1, color=color)
            ax.add_patch(rect)
            if pd.notna(rr_val):
                ax.text(col_i + 0.5, row_i + 0.5, f"{rr_val:.2f}",
                        ha="center", va="center", fontsize=8, fontweight="bold")

    # Axis labels
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([i + 0.5 for i in range(n_cols)])
    ax.set_xticklabels(
        [f"{model_display.get(m, m)}@{lim}" for m, lim in col_order],
        rotation=90, ha="center", fontsize=9,
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

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("", pad=0)
    fig.tight_layout()

    out_path = plots_dir / "relative_risk_by_ta_heatmap.png"
    logger.info(f"Saving plot to {out_path}")
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
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
            fig_d.patch.set_facecolor("white")
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
                rotation=90, ha="center", fontsize=9,
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

            for spine in ax_d.spines.values():
                spine.set_visible(False)

            sm_d = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm_d)
            sm_d.set_array([])
            plt.colorbar(sm_d, ax=ax_d, shrink=0.6, label="RR delta vs RDG")
            ax_d.set_title("", pad=0)
            fig_d.tight_layout()

            out_path_d = plots_dir / "relative_risk_delta_vs_rdg.png"
            logger.info(f"Saving plot to {out_path_d}")
            fig_d.savefig(str(out_path_d), dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig_d)

    # Plot 5: RR distributions across primary TAs — boxplot + jitter per model, faceted by limit
    _dist_limits = [10, 20, 50, 100]
    p3_slug  = model_display.get("p3_lambdarank_undirected", model_display.get("p3_lambdarank_directed", "EAHGT"))
    rdg_slug = model_display.get("rdg__no_time__positive", "RDG")
    model_order = [model_display.get(m, m) for m in all_model_names]

    rr_dist_df = rr_by_ta_primary[rr_by_ta_primary["limit"].isin(_dist_limits)].copy()
    rr_dist_df["model_slug"] = pd.Categorical(rr_dist_df["model_slug"], categories=model_order, ordered=True)

    # Compute Wilcoxon p-values per limit — embed in facet label
    limit_label_order = []
    for limit in _dist_limits:
        grp = rr_by_ta_primary[rr_by_ta_primary["limit"] == limit]
        p3_v  = grp[grp["model_slug"] == p3_slug].set_index("therapeutic_area_name")["relative_risk"]
        rdg_v = grp[grp["model_slug"] == rdg_slug].set_index("therapeutic_area_name")["relative_risk"]
        paired = pd.concat([p3_v, rdg_v], axis=1, keys=["p3", "rdg"]).dropna()
        if len(paired) >= 3 and (paired["p3"] - paired["rdg"]).abs().sum() > 0:
            _, pval = wilcoxon(paired["p3"], paired["rdg"], alternative="greater")
            pval_str = f"p={pval:.3f}" if pval >= 0.001 else f"p={pval:.2e}"
            label = f"N={limit} (Wilcoxon {pval_str})"
        else:
            label = f"N={limit}"
        limit_label_order.append(label)

    rr_dist_df["limit_label"] = rr_dist_df["limit"].map(dict(zip(_dist_limits, limit_label_order)))
    rr_dist_df["limit_label"] = pd.Categorical(rr_dist_df["limit_label"], categories=limit_label_order, ordered=True)

    _save_plot(
        pn.ggplot(rr_dist_df, pn.aes(x="model_slug", y="relative_risk", fill="model_slug"))
        + pn.geom_boxplot(outlier_size=0, alpha=0.5, width=0.5)
        + pn.geom_jitter(pn.aes(color="model_slug"), width=0.15, size=1.5, alpha=0.85)
        + pn.geom_hline(yintercept=1, linetype="dashed", color="grey")
        + pn.facet_wrap("~ limit_label", nrow=1, scales="free_y")
        + pn.scale_fill_manual(values=slug_colors)
        + pn.scale_color_manual(values=slug_colors)
        + pn.labs(x="", y="relative risk")
        + pn.theme_minimal()
        + pn.theme(
            figure_size=(4 * len(_dist_limits), 5),
            legend_position="none",
            axis_text_x=pn.element_text(rotation=35, ha="right"),
        ),
        plots_dir / "rr_distributions_ta.png",
    )

    # ------------------------------------------------------------------
    # Stratified plots (novelty × evidence sparsity)
    # ------------------------------------------------------------------
    _stratum_order = [
        "all", "pioneer", "known",
        "evidence_free", "literature_only", "direct_evidence",
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

    # Stratified classification metrics boxplot across primary TAs (stratum × TA)
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
        + pn.geom_boxplot(outlier_size=1, alpha=0.6)
        + pn.geom_jitter(width=0.15, size=1.2, alpha=0.7)
        + pn.facet_grid("metric ~ stratum_label", scales="free_y")
        + pn.scale_fill_manual(values=slug_colors)
        + pn.labs(x="", y="", fill="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(14, 7),
                   axis_text_x=pn.element_text(rotation=40, ha="right"),
                   legend_position="none"),
        plots_dir / "classification_metrics_by_stratum_by_ta.png",
    )

    # Stratified RR-by-limit line chart
    rr_strat = rr_by_limit_full[rr_by_limit_full["stratum"].isin(_stratum_order)].copy()
    rr_strat["stratum_label"] = rr_strat["stratum"].map(_stratum_label)
    rr_strat["stratum_label"] = pd.Categorical(
        rr_strat["stratum_label"], categories=_stratum_label_order, ordered=True
    )
    _save_plot(
        pn.ggplot(rr_strat, pn.aes(x="limit", y="relative_risk", color="model_slug", group="model_slug"))
        + pn.geom_line(size=0.9, alpha=0.85)
        + pn.geom_point(size=1.5, alpha=0.85)
        + pn.geom_hline(yintercept=1, linetype="dashed")
        + pn.facet_wrap("~ stratum_label", ncol=3, scales="free_y")
        + pn.scale_color_manual(values=slug_colors)
        + pn.scale_x_continuous(breaks=np.arange(10, 101, 20).tolist())
        + pn.scale_y_continuous(limits=(0, None))
        + pn.labs(x="N top target-disease pairs", y="relative risk", color="model")
        + pn.theme_minimal()
        + pn.theme(figure_size=(13, 7)),
        plots_dir / "relative_risk_by_limit_by_stratum.png",
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
    rr50 = rr_by_limit[rr_by_limit["limit"] == 50][["model_name", "relative_risk"]].sort_values("relative_risk", ascending=False)
    print("\n=== Relative Risk @ limit=50 ===")
    print(rr50.round(3).to_string(index=False))

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


if __name__ == "__main__":
    fire.Fire(evaluate)

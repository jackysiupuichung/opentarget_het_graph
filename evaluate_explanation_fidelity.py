"""Faithfulness metrics for the advancement explainer (SCAFFOLD).

Measures whether the edges an attribution method ranks highly actually control
the model's advancement prediction, via perturb-and-re-run fidelity metrics
(fid+ / fid- / characterization score / unfaithfulness), swept over sparsity.

See note/explain_fidelity.md for the rationale and method.

STATUS: scaffold. The metric/masking functions below define the intended
signatures and document the algorithm; the model/subgraph plumbing is meant to
REUSE explain_advancement.py (build_model + checkpoint load + LinkNeighborLoader)
rather than be re-implemented here. Marked TODO where the wiring is deferred
pending review. Nothing here runs the model yet.

Intended invocation (GPU, via sbatch):
    uv run python evaluate_explanation_fidelity.py \
        --config runs/<exp>/config.yaml \
        --checkpoint runs/<exp>/best_model.pt \
        --edges-parquet runs/<exp>/explanations/per_pair_edges.parquet \
        --methods ig abs_ig attention random \
        --sparsities 0.05 0.1 0.2 0.5 \
        --out-dir runs/<exp>/explanations/fidelity
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EdgeType = Tuple[str, str, str]

# Attribution methods we score. Each maps a per-edge attribution frame to a
# ranking key (higher = more important for the explanation).
METHODS = {
    "ig": "ig_total",            # signed IG (positive contributors)
    "abs_ig": "ig_total",        # |IG| (importance regardless of sign)
    "attention": "attention",    # heuristic; expected to score worse — that's the point
    "random": None,              # baseline: random edge ranking
}


def _rank_edges(edges: pd.DataFrame, method: str, seed: int) -> np.ndarray:
    """Return edge-row indices ordered most- to least-important under ``method``."""
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(len(edges))
    col = METHODS[method]
    vals = pd.to_numeric(edges[col], errors="coerce").fillna(0.0).to_numpy()
    if method == "abs_ig":
        vals = np.abs(vals)
    return np.argsort(-vals)  # descending


def mask_edge_index_dict(
    edge_index_dict: Dict[EdgeType, "object"],
    keep_mask: Dict[EdgeType, np.ndarray],
) -> Dict[EdgeType, "object"]:
    """Drop the columns where ``keep_mask`` is False, per edge type.

    The parallel ``edge_time_dict`` / ``edge_feat_dict`` must be masked with the
    SAME per-type boolean masks so the tensors stay column-aligned. The query
    (advancement) edge is never in these dicts (it is the edge_label_index), so
    it is never masked.

    TODO(impl): operate on torch tensors; return masked dicts for all three
    parallel dicts. Kept signature-only in the scaffold.
    """
    raise NotImplementedError("scaffold: tensor masking deferred (see note)")


def fidelity_for_pair(
    model,
    batch,
    edges: pd.DataFrame,
    method: str,
    k_frac: float,
    seed: int,
) -> Dict[str, float]:
    """fid+ / fid- / unfaithfulness for one pair, one method, one sparsity.

    Algorithm:
      base_p   = sigmoid(model(full subgraph))                      # query-edge prob
      order    = _rank_edges(edges, method, seed)
      top      = first ceil(k_frac * E) edges in ``order``
      # fid+ (necessity): remove ``top`` -> p_remove; fid+ = |base_p - p_remove|
      # fid- (sufficiency): keep ONLY ``top`` -> p_keep; fid- = |base_p - p_keep|
      # unfaithfulness: KL between full and kept-only prediction distributions
    Build masked edge_index_dict/edge_time_dict/edge_feat_dict via
    ``mask_edge_index_dict`` and re-run ``model.forward(...)`` exactly as
    explain_advancement.py calls it (src_type='target', dst_type='disease', etc.).

    TODO(impl): the two forward re-runs once mask_edge_index_dict lands.
    """
    raise NotImplementedError("scaffold: needs mask_edge_index_dict + model forward")


def characterization_score(fid_pos: float, fid_neg: float,
                           w_pos: float = 0.5, w_neg: float = 0.5) -> float:
    """Weighted harmonic mean of fid+ and (1 - fid-) (GraphFramEx, PyG).

    High only when the explanation is BOTH necessary (high fid+) and sufficient
    (low fid-). Returns 0 if either component is degenerate.
    """
    pos = max(0.0, min(1.0, fid_pos))
    neg_inv = max(0.0, min(1.0, 1.0 - fid_neg))
    if pos <= 0.0 or neg_inv <= 0.0:
        return 0.0
    return (w_pos + w_neg) / (w_pos / pos + w_neg / neg_inv)


def _load_model_and_loader(args):
    """Reuse explain_advancement.py's model build + checkpoint load + subgraph
    LinkNeighborLoader. Factor those out of explain_advancement.main into a
    shared helper (e.g. src/explain/runtime.py) and import here, rather than
    duplicating the (graph-loading, sampler) code.

    TODO(impl): extract + import. Deferred so this branch stays a scaffold and
    does not trigger a graph/model load (per the no-local-pipeline-runs rule;
    this is a GPU sbatch job when implemented).
    """
    raise NotImplementedError("scaffold: share runtime with explain_advancement.py")


def main(args: argparse.Namespace) -> None:
    # model, loader, context = _load_model_and_loader(args)   # TODO
    edges_all = pd.read_parquet(args.edges_parquet)
    pairs = edges_all[["target_id", "disease_id"]].drop_duplicates()
    print(f"[fidelity] {len(pairs)} pair(s); methods={args.methods}; "
          f"sparsities={args.sparsities}", flush=True)

    rows: List[dict] = []
    for _, pr in pairs.iterrows():
        t, d = pr["target_id"], pr["disease_id"]
        edges = edges_all[(edges_all["target_id"] == t)
                          & (edges_all["disease_id"] == d)]
        # batch = next subgraph from loader matching (t, d)               # TODO
        for method in args.methods:
            for k in args.sparsities:
                # m = fidelity_for_pair(model, batch, edges, method, k, args.seed)
                # m["characterization"] = characterization_score(m["fid_pos"], m["fid_neg"])
                # rows.append({"target_id": t, "disease_id": d, "method": method,
                #              "sparsity": k, **m})
                pass

    out = pd.DataFrame(rows)
    # out.to_parquet(...); summary table by (method, sparsity); fid-vs-sparsity plot
    print(f"[fidelity] SCAFFOLD: wiring deferred (see note/explain_fidelity.md). "
          f"Would write {args.out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Faithfulness (fidelity+/-/characterization/unfaithfulness) "
                    "of the advancement explainer's edge attributions.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--edges-parquet", required=True,
                   help="per_pair_edges.parquet with per-edge ig_total/attention.")
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   choices=list(METHODS))
    p.add_argument("--sparsities", nargs="+", type=float,
                   default=[0.05, 0.1, 0.2, 0.5],
                   help="Top-k edge fraction(s) to mask in/out.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

"""Faithfulness metrics for the advancement explainer.

Measures whether the edges an attribution method ranks highly actually control
the model's advancement prediction, via perturb-and-re-run fidelity metrics,
swept over sparsity, against a random baseline. See note/explain_fidelity.md.

For each explained (target, disease) pair we:
  1. sample its subgraph (shared ExplainRuntime; identical to explain_advancement),
  2. compute per-edge attributions in-batch (IG via integrated_gradients_for_pair;
     attention via capture_attention) so edges are aligned to the subgraph's own
     edge_index_dict — no cross-file index matching,
  3. for each method (ig / abs_ig / attention / random) and each top-k fraction:
       fid+ : drop the top-k edges      -> |p_base - p_drop|   (necessity)
       fid- : keep ONLY the top-k edges -> |p_base - p_keep|   (sufficiency)
       unfaithfulness : KL(p_base || p_keep) over the {neg,pos} class dist,
  4. characterization = harmonic mean of fid+ and (1 - fid-).

The query (advancement) edge is the prediction target and never lives in
edge_index_dict, so it is never masked. Masking drops the SAME columns from
edge_index_dict and edge_feat_dict so the two stay aligned.

GPU job via sbatch (see scripts). One eval process.

Invocation:
    uv run python evaluate_explanation_fidelity.py \
        --config <run>/config.yaml --checkpoint <run>/best_model.pt \
        --pairs-csv explain_pairs_evfree_diverse.csv \
        --methods ig abs_ig attention random \
        --sparsities 0.05 0.1 0.2 0.5 \
        --n-steps 32 --out-dir <run>/explanations/fidelity
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import ADV_ETYPE, build_edge_time_dict
from src.explain.runtime import ExplainRuntime, build_edge_feat_dict
from src.explain.attention_extractor import capture_attention, read_attention
from src.explain.captum_edge_explainer import integrated_gradients_for_pair

EdgeType = Tuple[str, str, str]
METHODS = ("ig", "abs_ig", "attention", "random")


# ----------------------------------------------------------------------------
# Per-edge attribution, flattened across edge types with a stable order.
# ----------------------------------------------------------------------------
def _flat_attributions(
    rt: ExplainRuntime, batch, edge_time_dict, n_steps: int, seed: int,
) -> Tuple[List[Tuple[EdgeType, int]], Dict[str, np.ndarray]]:
    """Return (edge_keys, {method: score[]}) where edge_keys[i] = (edge_type,
    local_col) identifies edge i, and score arrays are aligned to edge_keys.

    Higher score = more important under that method. IG/attention are computed
    on THIS batch so they index the batch's own edge_index_dict columns.
    """
    # IG: signed per-edge contribution = sum over feature dims.
    ig = integrated_gradients_for_pair(
        model=rt.model, batch=batch, edge_feat_cols=rt.edge_feat_cols,
        edge_time_dict=edge_time_dict, n_steps=n_steps,
    )
    ig_by_et = {et: a.sum(dim=1).numpy() for et, a in ig.edge_feat_attr.items()}

    # Attention: per-edge scalar (mean over heads/layers), keyed by edge type.
    with capture_attention(rt.model) as convs:
        with torch.no_grad():
            rt.predict_logit(batch, edge_time_dict=edge_time_dict)
        attn = read_attention(convs)
    attn_by_et = {et: a.cpu().numpy() for et, a in attn.items()}

    # Flat edge list over the edge types IG attributed (non-empty featured types
    # — exactly the ones we can mask meaningfully).
    edge_keys: List[Tuple[EdgeType, int]] = []
    ig_flat, absig_flat, attn_flat = [], [], []
    for et in ig_by_et:
        n = ig_by_et[et].shape[0]
        a = attn_by_et.get(et)
        for j in range(n):
            edge_keys.append((et, j))
            ig_flat.append(float(ig_by_et[et][j]))
            absig_flat.append(abs(float(ig_by_et[et][j])))
            attn_flat.append(float(a[j]) if a is not None and j < len(a) else 0.0)

    rng = np.random.default_rng(seed)
    scores = {
        "ig": np.asarray(ig_flat),
        "abs_ig": np.asarray(absig_flat),
        "attention": np.asarray(attn_flat),
        "random": rng.random(len(edge_keys)),
    }
    return edge_keys, scores


# ----------------------------------------------------------------------------
# Masking: keep / drop a set of flat edge indices.
# ----------------------------------------------------------------------------
def _keep_mask_from_topk(
    edge_keys: List[Tuple[EdgeType, int]], order: np.ndarray, k: int, mode: str,
    type_sizes: Dict[EdgeType, int],
) -> Dict[EdgeType, np.ndarray]:
    """Boolean keep-mask per edge type. ``order`` is flat-index order (most->least
    important). mode='drop_top' removes the top-k (fid+); mode='keep_top' keeps
    only the top-k (fid-). Edge types never ranked are kept whole."""
    top = set(int(i) for i in order[:k])
    keep = {et: np.ones(n, dtype=bool) for et, n in type_sizes.items()}
    ranked_types = {et for et, _ in edge_keys}
    if mode == "keep_top":
        # Only ranked types get pruned to their top-k; others pass through.
        for et in ranked_types:
            keep[et][:] = False
    for flat_i, (et, col) in enumerate(edge_keys):
        is_top = flat_i in top
        keep[et][col] = (not is_top) if mode == "drop_top" else is_top
    return keep


def _masked_dicts(
    batch, keep_cols: Dict[EdgeType, np.ndarray], edge_feat_cols: List[int],
) -> Tuple[Dict[EdgeType, torch.Tensor], Dict[EdgeType, torch.Tensor]]:
    """Build (edge_index_dict, edge_feat_dict) keeping only ``keep_cols`` per
    edge type. The SAME boolean mask is applied to edge_index and edge_feat so
    they stay column-aligned."""
    ei_out: Dict[EdgeType, torch.Tensor] = {}
    ef_out: Dict[EdgeType, torch.Tensor] = {}
    full_feat = build_edge_feat_dict(batch, edge_feat_cols)
    for et in batch.edge_types:
        ei = batch[et].edge_index
        mask = torch.as_tensor(keep_cols[et], dtype=torch.bool, device=ei.device)
        ei_out[et] = ei[:, mask]
        if et in full_feat:
            ef_out[et] = full_feat[et][mask]
    return ei_out, ef_out


# ----------------------------------------------------------------------------
# Metrics.
# ----------------------------------------------------------------------------
def _prob(logit: torch.Tensor) -> float:
    return float(torch.sigmoid(logit).item())


def _kl_bernoulli(p: float, q: float, eps: float = 1e-6) -> float:
    """KL( Bernoulli(p) || Bernoulli(q) ) over {neg, pos} — PyG-style
    unfaithfulness between full and masked prediction distributions."""
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def characterization_score(fid_pos: float, fid_neg: float) -> float:
    """Harmonic mean of fid+ and (1 - fid-): high only when the explanation is
    both necessary (high fid+) and sufficient (low fid-)."""
    pos = min(max(fid_pos, 0.0), 1.0)
    neg_inv = min(max(1.0 - fid_neg, 0.0), 1.0)
    if pos <= 0.0 or neg_inv <= 0.0:
        return 0.0
    return 2.0 / (1.0 / pos + 1.0 / neg_inv)


# ----------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    rt = ExplainRuntime.from_config(args.config, args.checkpoint)
    print(f"[fidelity] device={rt.device}; methods={args.methods}; "
          f"sparsities={args.sparsities}", flush=True)

    pair_idx = rt.select_pairs_from_csv(args.pairs_csv)
    print(f"[fidelity] {len(pair_idx)} pairs from {args.pairs_csv}", flush=True)
    if len(pair_idx) == 0:
        raise SystemExit("[fidelity] no pairs resolved to the test split")
    loader = rt.pair_loader(pair_idx)

    rows: List[dict] = []
    for bi, batch in enumerate(loader):
        batch = batch.to(rt.device)
        etd = build_edge_time_dict(batch, ADV_ETYPE)
        t_id, d_id = rt.pair_ids(batch)

        with torch.no_grad():
            base_p = _prob(rt.predict_logit(batch, edge_time_dict=etd))

        edge_keys, scores = _flat_attributions(
            rt, batch, etd, args.n_steps, args.seed + bi)
        E = len(edge_keys)
        type_sizes = {et: batch[et].edge_index.size(1) for et in batch.edge_types}
        if E == 0:
            print(f"[fidelity] {t_id}->{d_id}: no attributable edges, skip", flush=True)
            continue

        for method in args.methods:
            order = np.argsort(-scores[method])  # most->least important
            for frac in args.sparsities:
                k = max(1, int(math.ceil(frac * E)))

                keep_drop = _keep_mask_from_topk(edge_keys, order, k, "drop_top", type_sizes)
                ei_d, ef_d = _masked_dicts(batch, keep_drop, rt.edge_feat_cols)
                with torch.no_grad():
                    p_drop = _prob(rt.predict_logit(batch, edge_index_dict=ei_d,
                                                    edge_feat_dict=ef_d, edge_time_dict=etd))

                keep_keep = _keep_mask_from_topk(edge_keys, order, k, "keep_top", type_sizes)
                ei_k, ef_k = _masked_dicts(batch, keep_keep, rt.edge_feat_cols)
                with torch.no_grad():
                    p_keep = _prob(rt.predict_logit(batch, edge_index_dict=ei_k,
                                                    edge_feat_dict=ef_k, edge_time_dict=etd))

                fid_pos = abs(base_p - p_drop)
                fid_neg = abs(base_p - p_keep)
                rows.append({
                    "target_id": t_id, "disease_id": d_id, "method": method,
                    "sparsity": frac, "k": k, "n_edges": E, "base_p": base_p,
                    "p_drop_top": p_drop, "p_keep_top": p_keep,
                    "fid_pos": fid_pos, "fid_neg": fid_neg,
                    "characterization": characterization_score(fid_pos, fid_neg),
                    "unfaithfulness": _kl_bernoulli(base_p, p_keep),
                })
        print(f"[fidelity] {bi+1}/{len(pair_idx)} {t_id}->{d_id}: "
              f"E={E} base_p={base_p:.3f}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "fidelity_per_pair.parquet", index=False)

    if not df.empty:
        summary = (df.groupby(["method", "sparsity"])
                   .agg(fid_pos=("fid_pos", "mean"), fid_neg=("fid_neg", "mean"),
                        characterization=("characterization", "mean"),
                        unfaithfulness=("unfaithfulness", "mean"),
                        n=("fid_pos", "size"))
                   .reset_index()
                   .sort_values(["method", "sparsity"]))
        summary.to_parquet(out_dir / "fidelity_summary.parquet", index=False)
        print("\n[fidelity] summary (mean over pairs):", flush=True)
        print(summary.to_string(index=False), flush=True)
    print(f"\n[fidelity] wrote {len(df)} rows -> {out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Faithfulness (fid+/fid-/characterization/unfaithfulness) of "
                    "the advancement explainer's edge attributions.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-csv", required=True,
                   help="target_id,disease_id pairs to evaluate.")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--sparsities", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5])
    p.add_argument("--n-steps", type=int, default=32, help="IntegratedGradients steps.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())

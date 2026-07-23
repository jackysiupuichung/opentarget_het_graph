"""PaGE-Link path explanations for the EAHGT advancement predictor.

Reimplements PaGE-Link (Zhang et al., WWW 2023, arXiv:2302.12465) against our
PyG EAHGT model. Two stages per (target, disease) pair:

  STAGE 1 (mask learning): learn a soft edge mask m_e in [0,1] over the pair's
    subgraph that, when scaled into HGTConv message passing, preserves the
    model's advancement logit (BCE to the unmasked prediction), with size +
    entropy regularisers driving the mask sparse and decisive. Model frozen.

  STAGE 2 (path enforcement): turn the soft mask into connected target->disease
    PATHS. Edge cost = -log(m_e); the k lowest-cost simple paths (Yen, via
    networkx) are the explanation — connection-interpretable chains through the
    relation types, the ChronoMedKG-style decomposition.

Outputs reuse the per_pair_edges.parquet schema (mask weight in the ig_total
slot) so export_pair_evidence_json.py / present_pair_evidence.py render
PaGE-Link explanations with OT evidence unchanged, plus per_pair_paths.parquet.

GPU job via sbatch (mask learning is per-pair gradient descent). See
note/explain_pagelink.md.

Invocation:
    uv run python pagelink_explain.py \
        --config <run>/config.yaml --checkpoint <run>/best_model.pt \
        --pairs-csv explain_pairs_evfree_diverse.csv \
        --mask-epochs 200 --lr 0.01 --size-coeff 5e-3 --entropy-coeff 1e-1 \
        --num-paths 5 --out-dir <run>/explanations/pagelink
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # repo root (explain/cli/ -> repo)
sys.path.insert(0, str(ROOT))

from src.data.temporal_loader import ADV_ETYPE, build_edge_time_dict
from src.explain.runtime import ExplainRuntime, build_edge_feat_dict
from src.explain.edge_mask import EdgeMask, apply_edge_mask

EdgeType = Tuple[str, str, str]


# ----------------------------------------------------------------------------
# Stage 1 — learn the soft edge mask.
# ----------------------------------------------------------------------------
def learn_edge_mask(rt: ExplainRuntime, batch, edge_time_dict, args) -> EdgeMask:
    """Optimise an EdgeMask so the masked subgraph reproduces the model's logit.

    Loss = BCE(masked_logit, sigmoid(base_logit)) + size/entropy reg. The model
    is frozen; only mask logits are optimised (Adam). The mask is built over the
    SAME edge_index_dict the conv iterates, so flat() aligns with message order.
    """
    for p in rt.model.parameters():
        p.requires_grad_(False)

    edge_order = list(batch.edge_index_dict.keys())
    edge_counts = {et: batch[et].edge_index.size(1) for et in edge_order}
    # init<0 => sigmoid<0.5 => mask starts SPARSE, so an edge is kept only if the
    # BCE term actively pushes it up. (init>0 started near-all-on and the weak
    # reg never pruned it; mask mean rose to ~0.9.)
    mask = EdgeMask(edge_counts, edge_order, init=args.init, device=rt.device)

    with torch.no_grad():
        base_logit = rt.predict_logit(batch, edge_time_dict=edge_time_dict).detach()
    base_p = torch.sigmoid(base_logit)

    opt = torch.optim.Adam(mask.parameters(), lr=args.lr)
    for ep in range(args.mask_epochs):
        opt.zero_grad()
        with apply_edge_mask(rt.model, mask):
            logit = rt.predict_logit(batch, edge_time_dict=edge_time_dict)
        # Keep the masked prediction matched to the original (PaGE-Link fidelity
        # objective): BCE against the model's own probability, plus mask reg.
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, base_p)
        loss = bce + mask.regularisation(args.size_coeff, args.entropy_coeff)
        loss.backward()
        opt.step()
        if args.verbose and (ep % max(1, args.mask_epochs // 5) == 0 or ep == args.mask_epochs - 1):
            with torch.no_grad():
                m = mask.flat()
                print(f"[pagelink]     ep{ep:>3} loss={loss.item():.4f} "
                      f"bce={bce.item():.4f} mask[mean={m.mean():.3f} "
                      f">0.5={float((m > 0.5).float().mean()):.3f}]", flush=True)
    return mask


# ----------------------------------------------------------------------------
# Stage 2 — enforce target->disease paths from the soft mask.
# ----------------------------------------------------------------------------
def _global_node(batch, ntype: str, local_idx: int) -> Tuple[str, int]:
    return (ntype, int(batch[ntype].n_id[local_idx].item()))


def _is_excluded(rel: str, exclude_prefixes: Tuple[str, ...]) -> bool:
    """True if ``rel`` (with any rev_ stripped) starts with an excluded prefix."""
    base = rel[4:] if rel.startswith("rev_") else rel
    return any(base.startswith(p) for p in exclude_prefixes)


def enforce_paths(rt: ExplainRuntime, batch, mask: EdgeMask, num_paths: int,
                  min_mask: float, exclude_prefixes: Tuple[str, ...],
                  candidate_mult: int = 6, max_hops: int = 4
                  ) -> Tuple[List[dict], Dict[Tuple, float]]:
    """k best simple target->disease paths from the learned mask.

    Refinements over a plain shortest-path (so paths aren't dominated by
    trivial 1-hop clinical-trial edges):
      (a) EXCLUDE edge types whose (rev_-stripped) relation starts with any of
          ``exclude_prefixes`` (e.g. 'clinical_trial') from the path graph. The
          query edge is itself an advancement/trial edge, so trial hops are
          near-tautological explanations; dropping them forces paths to route
          through genetic/literature/pathway/animal-model evidence.
      (b) RECEPTIVE-FIELD CAP: drop any path longer than ``max_hops``. The model
          is a 2-layer message-passing net (num_neighbors [20,10]), so the target
          embedding aggregates <=2 hops and the disease embedding <=2 hops; a
          target->disease path within the model's actual receptive field is at
          most 2+2 = 4 hops. Longer connectivity chains traverse nodes the model
          never incorporated when scoring the query, so they are not faithful
          explanations and are excluded.
      (c) LENGTH-NORMALISE: enumerate candidates by summed cost (Yen), then
          RE-RANK by MEAN edge cost (total/n_hops) so a strong multi-hop path
          isn't beaten purely for having more hops.

    cost(edge) = -log(m_e). Returns (paths, edge_mask_by_key).
    """
    import networkx as nx

    mvals = {et: v.detach().cpu().numpy() for et, v in mask.values().items()}
    G = nx.DiGraph()
    edge_mask_by_key: Dict[Tuple, float] = {}

    for et in batch.edge_index_dict:
        st, rel, dt = et
        if _is_excluded(rel, exclude_prefixes):     # (a) drop excluded relations
            continue
        ei = batch[et].edge_index.cpu().numpy()
        m = mvals.get(et)
        if m is None:
            continue
        for j in range(ei.shape[1]):
            su, dv = int(ei[0, j]), int(ei[1, j])
            me = float(m[j])
            if me < min_mask:
                continue
            gn_s = _global_node(batch, st, su)
            gn_d = _global_node(batch, dt, dv)
            cost = -math.log(max(me, 1e-6))
            key = (et, gn_s[1], gn_d[1])
            edge_mask_by_key[key] = me
            if G.has_edge(gn_s, gn_d):
                if cost < G[gn_s][gn_d]["cost"]:
                    G[gn_s][gn_d].update(cost=cost, et=et, me=me)
            else:
                G.add_edge(gn_s, gn_d, cost=cost, et=et, me=me)

    # Endpoints: the query edge in global node space.
    ls = int(batch[ADV_ETYPE].edge_label_index[0, 0].item())
    ld = int(batch[ADV_ETYPE].edge_label_index[1, 0].item())
    src = ("target", int(batch["target"].n_id[ls].item()))
    dst = ("disease", int(batch["disease"].n_id[ld].item()))

    candidates: List[dict] = []
    if src in G and dst in G:
        try:
            gen = nx.shortest_simple_paths(G, src, dst, weight="cost")
            # shortest_simple_paths yields by increasing cost, not length, so a
            # too-long path can appear before a valid short one: skip over-length
            # paths (don't count them against the pool) and keep enumerating,
            # with a hard iteration bound so a pathological graph can't loop.
            seen = 0
            for nodes in gen:
                seen += 1
                if seen > num_paths * candidate_mult * 20:
                    break
                n = len(nodes) - 1
                if n > max_hops:                       # (b) receptive-field cap
                    continue
                if len(candidates) >= num_paths * candidate_mult:
                    break
                edges, total = [], 0.0
                for a, b in zip(nodes[:-1], nodes[1:]):
                    d = G[a][b]
                    edges.append({"edge_type": "::".join(d["et"]),
                                  "src_type": a[0], "src_global": a[1],
                                  "dst_type": b[0], "dst_global": b[1],
                                  "m_e": d["me"]})
                    total += d["cost"]
                candidates.append({"n_hops": n, "total_cost": total,
                                   "mean_cost": total / max(n, 1), "edges": edges})
        except nx.NetworkXNoPath:
            pass

    # Rank candidates by TOTAL edge cost (PaGE-Link's concise-path scoring).
    # (An earlier mean-cost re-rank favoured long chains of high-mask edges,
    # producing 7-hop artifact paths; total cost keeps explanations concise as
    # in the published method.)
    candidates.sort(key=lambda p: p["total_cost"])
    paths = []
    for rank, p in enumerate(candidates[:num_paths]):
        p["rank"] = rank
        paths.append(p)
    return paths, edge_mask_by_key


# ----------------------------------------------------------------------------
# Output.
# ----------------------------------------------------------------------------
def _edge_rows(rt: ExplainRuntime, batch, mask: EdgeMask, t_id: str, d_id: str) -> List[dict]:
    """Per-edge rows in the per_pair_edges.parquet schema, mask weight in the
    ig_total slot so the existing decomposition tooling consumes it unchanged."""
    mvals = {et: v.detach().cpu().numpy() for et, v in mask.values().items()}
    rows = []
    for et in batch.edge_index_dict:
        ei = batch[et].edge_index.cpu().numpy()
        m = mvals.get(et)
        if m is None:
            continue
        st, rel, dt = et
        for j in range(ei.shape[1]):
            su, dv = int(ei[0, j]), int(ei[1, j])
            rows.append({
                "target_id": t_id, "disease_id": d_id,
                "edge_type": "::".join(et),
                "src": int(batch[st].n_id[su].item()),
                "dst": int(batch[dt].n_id[dv].item()),
                "ig_total": float(m[j]),      # mask weight as the attribution
                "mask_weight": float(m[j]),
            })
    return rows


def main(args: argparse.Namespace) -> None:
    rt = ExplainRuntime.from_config(args.config, args.checkpoint)
    print(f"[pagelink] device={rt.device}; mask_epochs={args.mask_epochs}; "
          f"num_paths={args.num_paths}", flush=True)

    pair_idx = rt.select_pairs_from_csv(args.pairs_csv)
    print(f"[pagelink] {len(pair_idx)} pairs from {args.pairs_csv}", flush=True)
    if len(pair_idx) == 0:
        raise SystemExit("[pagelink] no pairs resolved to the test split")
    loader = rt.pair_loader(pair_idx)

    all_edge_rows: List[dict] = []
    all_path_rows: List[dict] = []
    for bi, batch in enumerate(loader):
        batch = batch.to(rt.device)
        # Align the explainer's edge universe with the model's post-collapse
        # messages when latest_edge_only is set (no-op otherwise).
        batch = rt.collapse_batch(batch)
        etd = build_edge_time_dict(batch, ADV_ETYPE)
        t_id, d_id = rt.pair_ids(batch)
        print(f"[pagelink] {bi+1}/{len(pair_idx)} {t_id}->{d_id}: learning mask...",
              flush=True)

        mask = learn_edge_mask(rt, batch, etd, args)
        all_edge_rows.extend(_edge_rows(rt, batch, mask, t_id, d_id))

        exclude = tuple(x for x in args.exclude_relations.split(",") if x)
        paths, _ = enforce_paths(rt, batch, mask, args.num_paths, args.min_mask,
                                 exclude_prefixes=exclude, max_hops=args.max_hops)
        for p in paths:
            e0 = p["edges"][0]
            # index chain (stable id) + human-readable accession+name chain
            chain = " -> ".join(
                [f"{e0['src_type']}#{e0['src_global']}"]
                + [f"[{e['edge_type']}] {e['dst_type']}#{e['dst_global']}" for e in p["edges"]]
            )
            named = " -> ".join(
                [rt.node_label(e0["src_type"], e0["src_global"])]
                + [f"[{e['edge_type']}] {rt.node_label(e['dst_type'], e['dst_global'])}"
                   for e in p["edges"]]
            )
            all_path_rows.append({
                "target_id": t_id, "disease_id": d_id, "rank": p["rank"],
                "n_hops": p["n_hops"], "total_cost": p["total_cost"],
                "mean_cost": p["mean_cost"],
                "min_m_e": min((e["m_e"] for e in p["edges"]), default=float("nan")),
                "path": chain, "path_named": named,
            })
        print(f"[pagelink]   -> {len(paths)} path(s); "
              f"{len([r for r in all_edge_rows if r['target_id']==t_id and r['disease_id']==d_id])} edges",
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_edge_rows).to_parquet(out_dir / "per_pair_edges.parquet", index=False)
    paths_df = pd.DataFrame(all_path_rows)
    paths_df.to_parquet(out_dir / "per_pair_paths.parquet", index=False)
    if not paths_df.empty:
        print("\n[pagelink] best path per pair (named):", flush=True)
        for _, r in paths_df[paths_df["rank"] == 0].iterrows():
            print(f"  [{r['target_id']} -> {r['disease_id']}] "
                  f"hops={r['n_hops']} mean_cost={r['mean_cost']:.3f}", flush=True)
            print(f"    {r['path_named']}", flush=True)
    print(f"\n[pagelink] wrote {len(all_edge_rows)} edge rows, "
          f"{len(all_path_rows)} path rows -> {out_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PaGE-Link path explanations for EAHGT advancement.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-csv", required=True,
                   help="target_id,disease_id pairs to explain.")
    p.add_argument("--mask-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--size-coeff", type=float, default=5e-3)
    p.add_argument("--entropy-coeff", type=float, default=1e-1)
    p.add_argument("--init", type=float, default=-1.0,
                   help="EdgeMask logit init. <0 => sigmoid<0.5 => mask starts "
                        "sparse so edges are kept only if BCE pushes them up "
                        "(default -1.0 ~ m_e 0.27).")
    p.add_argument("--max-hops", type=int, default=4,
                   help="Max target->disease path length. Default 4 = the "
                        "2-layer model's receptive field (2 hops per endpoint); "
                        "longer paths traverse nodes the model never saw when "
                        "scoring the query and are unfaithful.")
    p.add_argument("--num-paths", type=int, default=5,
                   help="k best (mean-cost) target->disease paths per pair.")
    p.add_argument("--min-mask", type=float, default=0.1,
                   help="Drop edges with m_e below this from path search.")
    p.add_argument("--exclude-relations", type=str, default="clinical_trial",
                   help="Comma-separated relation PREFIXES (rev_ ignored) to drop "
                        "from stage-2 path search. Default 'clinical_trial' so "
                        "paths explain via non-trial evidence (trial hops are "
                        "near-tautological for an advancement prediction). Pass "
                        "'' to keep all relations.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    main(args)

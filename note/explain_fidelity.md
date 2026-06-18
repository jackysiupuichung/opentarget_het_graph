# Faithfulness metrics for the advancement explainer (#1)

## Why

The current explainer (`explain_advancement.py`) emits two per-edge signals —
Integrated-Gradients attribution (`ig_total`) over edge features (score,
novelty), and aggregated post-softmax HGTConv `attention`. Neither has been
**fidelity-tested** against the model. Case studies built on them make a
*structural* claim ("these edges drove the advancement call") from
*feature-level / heuristic* attributions.

Two literature caveats motivate this branch:

- Raw attention is not a faithful explanation (Jain & Wallace, arXiv:1902.10186;
  Wiegreffe & Pinter, arXiv:1908.04626). HGT attention is also computed
  per-relation then aggregated, so cross-relation magnitudes aren't directly
  comparable.
- IG attributes *feature values*, not edge *existence* / message-passing
  structure — so it does not answer "which subgraph drives this".

So before (or alongside) any structural method, we measure whether the
edges an attribution ranks highly actually *control the model's prediction*.

## What we compute

Standard GNN-explanation faithfulness metrics (as in `torch_geometric.explain.metric`
and GraphFramEx, arXiv:2206.09677), per explained pair, per attribution method
(IG, |IG|, attention, and a random baseline):

- **Fidelity+ (necessity):** prediction change when the top-k explanation edges
  are **removed** from the subgraph. Higher = those edges were necessary.
- **Fidelity- (sufficiency):** prediction change when **only** the top-k edges
  are kept. Lower = they were sufficient.
- **Characterization score:** weighted harmonic mean of fid+ and (1 - fid-) —
  the single number that can't be gamed by trading one fidelity for the other.
- **Unfaithfulness:** KL divergence between full-subgraph and masked-subgraph
  prediction distributions (PyG's `unfaithfulness`).
- Swept over **sparsity** (k as a fraction of edges) to get fid-vs-sparsity curves.

The model's prediction here is the advancement logit on the query
(target, disease) edge; "masking" an edge = dropping its column from
`edge_index_dict` (and the parallel `edge_time_dict` / `edge_feat_dict`), then
re-running `model.forward(...)` on the same query edge. The query edge itself is
never masked.

## Approach (perturb-and-re-run; no PyG Explainer rewrite needed)

We reuse the existing subgraph extraction and per-edge attributions. We do NOT
need to route through `torch_geometric.explain` — the model forward already
takes `edge_index_dict`, so masking is a dict-column drop. (A later step *can*
adopt `CaptumExplainer(edge_mask_type='object')` / `GNNExplainer` to get a
structural mask; this branch's job is to *measure* whatever attribution we have.)

`evaluate_explanation_fidelity.py` (scaffold in this branch):
1. Load model + checkpoint exactly as `explain_advancement.py` does (reuse its
   `build_model` + checkpoint load + `LinkNeighborLoader` subgraph extraction —
   factor those into a shared helper rather than duplicate).
2. For each pair, load its per-edge attribution from `per_pair_edges.parquet`
   (or recompute), align edges to the subgraph's `edge_index_dict` columns.
3. For each method and each sparsity k: build the masked `edge_index_dict`
   (remove top-k for fid+, keep-only-top-k for fid-), re-run forward, record the
   logit/probability.
4. Aggregate fid+/fid-/characterization/unfaithfulness, write
   `fidelity_<method>.parquet` + a summary table; plot fid-vs-sparsity.

GPU job via `sbatch` (one task, `--gres=gpu:1`); single eval process.

## Status

Scaffold only (`evaluate_explanation_fidelity.py`) — the masking/aggregation
functions are stubbed with the intended signatures and docstrings; the
subgraph-extraction reuse and the actual sbatch run are deferred pending review.

## References
- PyG metrics: `torch_geometric.explain.metric` (fidelity, characterization_score,
  fidelity_curve_auc, unfaithfulness).
- GraphFramEx (Amara et al., arXiv:2206.09677) — model vs phenomenon explanations,
  characterization score. We want `explanation_type='model'`.
- Sundararajan et al., Integrated Gradients, arXiv:1703.01365.

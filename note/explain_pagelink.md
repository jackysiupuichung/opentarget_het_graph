# PaGE-Link path explanations for EAHGT advancement (#4)

## Why

For a *publication-grade* answer to "which subgraph structure drives this
advancement prediction", the on-target method is **PaGE-Link** (Zhang et al.,
WWW 2023, arXiv:2302.12465): path-based GNN explanation built specifically for
**heterogeneous link prediction**. It returns human-readable **paths** between
the predicted source and target — for us, target -> ... -> disease paths through
the 20 relation types — which is exactly the ChronoMedKG-style decomposition the
case studies aim for (a prediction told as a grounded relational chain).

This complements branch #1 (fidelity): #1 *measures* whatever attribution we
have; #4 *produces* a faithful, connected, path-structured attribution.

## What PaGE-Link does (two stages)

1. **Mask learning.** Learn a soft edge mask m_e in [0,1] over the (k-hop)
   subgraph around the query (s, t) link by maximising the masked subgraph's
   ability to reproduce the model's prediction, with sparsity + entropy
   regularisers (GNNExplainer-style objective, but for the *link* logit).
2. **Path enforcement / pruning.** Convert the soft mask into connected
   **paths** between s and t: edge weights w_e = -log(m_e) define a cost; the
   explanation is the set of shortest / low-cost paths under those weights,
   filtered for heterogeneity (sensible relation sequences). This is the part
   that distinguishes PaGE-Link from a plain edge mask — it guarantees the
   output is path-structured and connection-interpretable.

Output per pair: a ranked list of paths (sequence of typed edges with
attribution weight) from target to disease, plus the underlying edge mask.

## Adaptation to this repo

Reference implementation is **DGL-based**; EAHGT here is **PyG** (HGT over a
PyG `HeteroData`, `model.forward(x_dict, edge_index_dict, edge_label_index, ...)`).
No official PyG port exists -> we reimplement against our model. Plan:

- **Subgraph + model reuse.** Use the same k-hop subgraph extraction and model
  forward as `explain_advancement.py` (factor into a shared runtime helper —
  same helper branch #1 needs). The query edge is `edge_label_index`.
- **Differentiable edge mask.** Multiply m_e into message passing. Our HGTConv
  already multiplies a per-edge `edge_attr` scalar into attention
  (src/models/hgt_conv_rte.py); the cleanest hook is to inject an extra
  per-edge multiplicative mask alongside `edge_feat_dict` so we don't fork the
  conv. Stage-1 optimises m over a frozen model on the single link logit.
- **Path enforcement.** Build a weighted graph from the learned mask over the
  subgraph's edges, run shortest-path (k shortest paths) from target to disease
  with cost -log(m_e); keep paths whose relation sequence is admissible. Reuse
  the rev_/forward canonicalisation already in join_pair_evidence.py so paths
  read in the forward relation orientation.
- **Output -> existing decomposition.** Emit per-edge attribution into the same
  `per_pair_edges.parquet` schema (mask weight in the `ig_total` slot) so
  export_pair_evidence_json.py / present_pair_evidence.py render PaGE-Link paths
  with their OT evidence unchanged. Additionally emit `per_pair_paths.parquet`
  (one row per path: ordered edges + total cost) for the path view.
- **Validate with branch #1.** Report fid+/fid-/characterization for PaGE-Link
  paths vs IG vs attention vs random — the head-to-head that justifies the
  method choice in the paper.

## Files (scaffold in this branch)

- `pagelink_explain.py` — driver + the two-stage algorithm, stubbed with the
  intended signatures (mask learning, path enforcement, output writers).
- `src/explain/edge_mask.py` — the differentiable per-edge mask module + the
  injection hook into HGTConv message passing (stub).

## Status

Scaffold only — algorithm + signatures + adaptation plan documented; the
differentiable mask, the optimisation loop, and the path search are stubbed
(NotImplementedError) pending review. No training/graph load is triggered.
Implementation is a GPU sbatch job (mask learning is gradient descent per pair).

## References
- PaGE-Link, arXiv:2302.12465 (DGL reference impl).
- GNNExplainer mask objective, arXiv:1903.03894.
- Fidelity metrics: note/explain_fidelity.md (branch #1) — shared evaluation.

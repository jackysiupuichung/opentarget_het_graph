# Explainability tooling: PaGE-Link and GraphFramEx — what they are, and how ours is implemented

**Date:** 2026-07-02
**Scope:** The two published methods the paper's explainability section (§3.4) leans on —
PaGE-Link (path explanations) and GraphFramEx (faithfulness evaluation) — plus a precise
accounting of which parts are off-the-shelf packages versus code we wrote in this repo.
Written to answer: *are these packages or raw code we created?*

---

## TL;DR

| Component | Published method (whose) | In our repo | Package or our code? |
|---|---|---|---|
| Edge attribution (Integrated Gradients) | Sundararajan et al. 2017 | `src/explain/captum_edge_explainer.py` | **`captum` package** (pip dep) wrapped by our thin adapter |
| Attention read-out | HGT (Hu et al. 2020) | `src/explain/attention_extractor.py` | **our code** (forward hooks on PyG `HGTConv`) |
| Faithfulness evaluation (fidelity+/−, characterisation) | GraphFramEx (Amara et al. 2022) | `evaluate_explanation_fidelity.py` | **our code**, implementing GraphFramEx's *protocol* (NOT their package) |
| Path explanation (soft mask → simple paths) | PaGE-Link (Zhang et al. 2023) | driven via `src/explain/runtime.py` hooks | **our code**, re-implementing PaGE-Link's *idea* (NOT their package) |

**Bottom line:** only **captum** (for Integrated Gradients) is an installed third-party
package. **GraphFramEx and PaGE-Link are not packages we install** — we re-implemented their
*methods/protocols* in our own code against our HGT model and PyG `edge_index_dict` forward
API. So the reputable *methods* are prior published work; the *implementations* are ours.

---

## 1. GraphFramEx (Amara et al., 2022) — faithfulness evaluation framework

### What it is (published work)
- **Amara, Ying, et al., "GraphFramEx: Towards Systematic Evaluation of Explainability
  Methods for Graph Neural Networks", Learning on Graphs (LoG) 2022**, arXiv:2206.09677.
- It is the field's *systematic evaluation framework* for GNN explanations. It defines the
  now-standard metric triple:
  - **Fidelity+ (necessity):** prediction change when the top-k explanation edges are
    **removed**. Higher ⇒ those edges were necessary.
  - **Fidelity− (sufficiency):** prediction change when **only** the top-k edges are kept.
    Lower ⇒ they were sufficient.
  - **Characterisation score:** (weighted) harmonic mean of fid+ and (1 − fid−) — a single
    number that can't be gamed by trading one fidelity against the other.
  - Distinguishes *model* vs *phenomenon* explanations; we use `explanation_type='model'`.
- Bib key in paper: `Amara2022GraphFramEx`. Cited at §3.4.1 (`\subsubsection{Faithfulness of
  edge attributions.}`).

### How WE implemented it (our code, not their package)
- File: **`evaluate_explanation_fidelity.py`** (top-level, ~285 lines). Documented in
  `note/explain_fidelity.md`.
- **Not** installed from GraphFramEx. We compute the GraphFramEx-style metrics ourselves,
  because our model's forward already accepts an `edge_index_dict`, so "masking an edge" is
  just dropping its column from the dict (and the parallel `edge_time_dict` /
  `edge_feat_dict`) and re-running `model.forward(...)`. No PyG `Explainer` rewrite needed.
- Key implementation facts (verified in source):
  - `METHODS = ("ig", "abs_ig", "attention", "random")` — the four rankings compared.
  - `characterization_score(fid_pos, fid_neg)` = `2 / (1/pos + 1/neg_inv)` where
    `neg_inv = 1 - fid_neg` — plain harmonic mean, our own function.
  - Unfaithfulness / divergence = `_kl_bernoulli(p, q)`, KL between full-subgraph and
    masked-subgraph Bernoulli prediction distributions (PyG-style `unfaithfulness`), our own
    function.
  - Swept over sparsity (top-k as a fraction of edges); the paper's Table~\ref{tab:fidelity}
    reports **each row at the sparsity maximising its own characterisation** (noted in the
    table foot — this is the "not a single fixed budget" caveat).
  - The query (target, disease) edge is never masked.
- Related note: `note/explain_fidelity.md` cross-references `torch_geometric.explain.metric`
  (PyG *does* ship fidelity/characterisation/unfaithfulness helpers) — we chose the
  perturb-and-re-run route over PyG's `Explainer` wrapper, so the metric math is ours.

### Reputability
The **metrics and their names are GraphFramEx's** (and PyG mirrors them). Ours is a faithful
re-implementation of that protocol, not a bespoke metric we invented. This is the systematic
method that already grounds the *edge-level* evaluation.

---

## 2. PaGE-Link (Zhang et al., 2023) — path explanations

### What it is (published work)
- **Zhang, Liu, et al., "PaGE-Link: Path-based Graph Neural Network Explanation for
  Heterogeneous Link Prediction", TheWebConf (WWW) 2023.**
- Method: learns a **sparse soft mask** over subgraph edges, constrained to preserve the
  model's link-prediction score, then extracts the **lowest-cost simple paths** between the
  two endpoints under edge costs derived from the mask. Purpose-built for *heterogeneous link
  prediction* explanation — exactly our setting (target→disease link).
- Bib key in paper: `Zhang2023PaGELink`. Cited at §3.4.2 (`\subsubsection{Path-level
  explanations.}`).

### How WE implemented it (our code, not their package)
- **PaGE-Link is not installed as a package.** We re-implement its idea against our model.
- The infrastructure hooks live in **`src/explain/runtime.py`** (`ExplainRuntime`):
  - `predict_logit(...)` is deliberately **NOT** wrapped in `no_grad` "so PaGE-Link can
    backprop to a mask" (verbatim from the docstring) — i.e. our forward supports the
    gradient flow a learned soft mask needs.
  - The docstring states the fidelity harness and "the PaGE-Link explainer" build the SAME
    model on the SAME context graph, so masking is consistent across both.
- Our adaptations to PaGE-Link for the advancement task (verified against §3.4.2 prose and
  the runtime hooks):
  - Because the prediction target is itself a clinical-trial edge, the only *direct*
    target–disease relations are trial edges. We **exclude trial relations from the path
    search** so an explanation can't just restate the label.
  - We rank candidate paths by **mean** rather than total edge cost, forcing routes through
    genetic / expression / pathway / literature evidence.
- **Status caveat (important, from memory + note):** PaGE-Link's mask **does not sparsify
  well** in our setup (size-coefficient too weak), so the paper presents paths as a *worked
  demonstration* (the IL17F → psoriatic-arthritis case), and explicitly defers systematic
  path-plausibility evaluation to future work. The path section is the one **ad hoc** piece;
  the edge section is systematic (GraphFramEx).

### Reputability
The **method is PaGE-Link's** (published, WWW 2023). Our implementation is a re-derivation
of that method wired to our HGT model, with two task-specific modifications (trial-edge
exclusion; mean-cost ranking). We do not claim a new path-explanation algorithm.

---

## 3. Integrated Gradients — the one genuine package

- **`captum` (pip dependency, `captum>=0.7.0`, resolved 0.9.0 in `uv.lock`).** This is the
  only installed third-party explainability package.
- Wrapped by **`src/explain/captum_edge_explainer.py`** (`integrated_gradients_for_pair`,
  `PairAttribution`), which adapts our heterogeneous `edge_attr` / `edge_index_dict` forward
  into the `[B, 2]` logit shape captum's `IntegratedGradients` expects. The IG *algorithm* is
  captum's; the adapter is ours.
- Method reference: Sundararajan et al. 2017 (`Sundararajan2017Axiomatic`), arXiv:1703.01365.

---

## 4. Supporting code we wrote (all ours)

| File | Role |
|---|---|
| `src/explain/runtime.py` | `ExplainRuntime`: shared model+checkpoint+subgraph loader used by BOTH the fidelity harness and PaGE-Link; the grad-enabled `predict_logit` |
| `src/explain/captum_edge_explainer.py` | captum IG adapter for heterogeneous edge features |
| `src/explain/attention_extractor.py` | `capture_attention` / `read_attention` — forward hooks capturing post-softmax HGTConv attention |
| `src/explain/aggregate.py` | `per_pair_edges_df` / `per_pair_nodes_df` — assemble per-pair explanation tables |
| `src/explain/viz.py` | `plot_pair_subgraph` — subgraph visualisation |
| `evaluate_explanation_fidelity.py` | GraphFramEx-protocol fidelity harness (driver) |
| `explain_advancement.py` | Top-level explainer: IG + attention + evidence lookup + naming |
| `note/explain_fidelity.md` | Design note for the fidelity harness |

---

## 5. Implication for the paper (path-level section, §3.4.2)

- The **edge-attribution** evaluation is already systematic — it runs the **GraphFramEx**
  fidelity protocol (our implementation of it). Reputable, quantitative, done.
- The **path** evaluation (PaGE-Link) is the **only ad hoc** piece: one hand-picked case, no
  systematic scoring, because the learned mask won't sparsify.
- Both underlying methods are **published prior work** (GraphFramEx LoG'22; PaGE-Link WWW'23)
  — the reputability the user wanted is real; only the *path evaluation* is anecdotal.
- Cleanest honest options (no new methodology invented): (a) score the PaGE-Link path mask
  under the SAME GraphFramEx fidelity protocol already used for edges (needs a compute run,
  and the weak-sparsification caveat means it may not tell a clean story), or (b) reframe the
  path section to state the mask is *evaluable* under GraphFramEx fidelity and that
  ground-truth-motif plausibility is what THBKG lacks — keeping the IL17F case as
  *illustrative*, not evidential.

---

## References (methods, not packages)
- Amara, Ying, et al. GraphFramEx. LoG 2022. arXiv:2206.09677. (bib `Amara2022GraphFramEx`)
- Zhang, Liu, et al. PaGE-Link. WWW 2023. (bib `Zhang2023PaGELink`)
- Sundararajan, Taly, Yan. Integrated Gradients. ICML 2017. arXiv:1703.01365.
  (bib `Sundararajan2017Axiomatic`)
- Jain & Wallace, "Attention is not Explanation", NAACL 2019. arXiv:1902.10186.
  (bib `Jain2019Attention`)

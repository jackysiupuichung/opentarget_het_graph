# Methodological provenance: multi-seed + rank-fusion + robust reporting under val/test decorrelation

**Question (from the user):** the THBKG paper combines four practices to handle a sharp
validation/test decorrelation under temporal shift in a low-base-rate ranking task —
(1) multi-seed ensembling with validation-selected checkpoints, (2) percentile-rank
(rank-based) fusion rather than score averaging, (3) reporting an outlier-robust per-group
median + breadth (count of TAs beating baseline) instead of the spike-prone mean,
(4) acknowledging that a single validation-selected epoch is unstable / the peak epoch is
unselectable. Is there prior work that implements all/most of these *together* to solve
such an issue?

**Short answer.** No single paper assembles all four — and certainly none in the biomedical-KG
/ clinical-trial-ranking domain. Each practice is individually well-grounded in a *separate*
methods literature (domain-generalization model selection, rank fusion, RL/benchmark evaluation
statistics, OOD model-selection). The THBKG contribution is the **combination**, applied to
decision-aligned clinical-advancement ranking. This is a defensible, honest positioning: cite
the four provenances, then state that combining them for temporally-masked low-base-rate KG
ranking is, to our knowledge, new.

> Search date: 2026-06. Databases: web (arXiv / Semantic Scholar / proceedings) + PubMed.
> This is a *provenance* search, not a PRISMA systematic review. Citations below were
> metadata-checked; a few search hits with implausible future arXiv IDs were discarded as
> likely hallucinations and are NOT included.

---

## Practice-by-practice provenance (each exists; none combined)

### (1) Multi-seed ensembling + validation-selected checkpoints under shift
- **Arpit et al., "Ensemble of Averages" (EoA), NeurIPS 2022** (arXiv:2110.10832). The closest
  single anchor. Trains independent seeds, observes that per-seed performance on shifted test
  domains is *chaotic*, and ensembles to **improve the rank correlation between in-domain
  validation and out-domain test** — i.e. exactly the val/test-decorrelation framing, used to
  make early stopping / checkpoint selection reliable. Differs from us: vision domain
  generalization, accuracy not RS@N, weight-space averaging (their twist) vs our prediction-space
  rank fusion.
- **Maddox/Izmailov/Wilson, MultiSWAG / MultiSWA** (via "Beyond Deep Ensembles…", arXiv:2306.12306
  survey, and the original SWAG line). Multi-seed averaging beats plain deep ensembles
  *specifically when few models are used and corruption/shift is high* — supports the "ensemble
  to absorb single-run instability under shift" rationale.
- **Domain (biomedical KG): Gilmer-adjacent, "Ensembles of KG embedding models improve
  predictions for drug discovery," Bioinformatics 2022** (PMC9677479). Shows two KGE models on
  the *same* biomedical graph give nearly **disjoint top-predicted triples**, and ensembling
  stabilises them. This is the in-domain precedent for *why ensemble at all* — but it is **not**
  motivated by temporal/distribution shift, does **not** use rank fusion for decorrelation, and
  does **not** report robust per-group statistics.

### (2) Rank-based fusion > score averaging
- **Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion" (RRF), SIGIR 2009.** Canonical result
  that a parameter-free *rank*-based combiner beats trained score-combination / learning-to-rank
  fusers and "makes better use of ranker diversity." Direct provenance for choosing
  percentile-rank fusion over score averaging because RS@N depends only on ordering and seed
  scores live on different scales.
- Supporting IR/recsys fusion line (CombSUM/CombMNZ; UMRE monotonic ranking ensemble,
  arXiv:2508.07613) — same theme: normalise to a scale-free rank space before combining.
- Note the nuance from the IR literature: with careful per-system score normalisation, score
  averaging can match rank fusion — so our justification should be the *scale-free, training-free,
  ordering-is-all-RS@N-needs* argument, which RRF makes cleanly.

### (3) Robust per-group reporting (median / breadth) instead of the spike-prone mean
- **Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice,"
  NeurIPS 2021 (Outstanding Paper)** (arXiv:2108.13264). *The* anchor for practice #3. Argues the
  task-**mean is dominated by a few outlier tasks**, recommends **interquartile mean (IQM) /
  median**, **stratified-bootstrap interval estimates**, and **performance profiles** (tail
  distribution of scores) — robust to a handful of high-variance tasks and reliable with few
  runs. This is almost exactly our "novelty's RS@10 mean is a 2-TA spike; report median + #TA>RDG
  breadth + Katz CI" move, transplanted from RL benchmarking to per-therapeutic-area RS@N.
- **DomainBed / subpopulation-shift line — Gulrajani & Lopez-Paz, "In Search of Lost Domain
  Generalization," ICLR 2021; Yang et al., "Change is Hard," ICML 2023** (arXiv:2302.12254).
  Emphasise **worst-group** and group-stratified reporting over the average; selecting by highest
  *average* validation accuracy can drop worst-group test by >20%. Provenance for "report breadth /
  worst areas, not just the mean."

### (4) Single val-selected epoch is unstable / the peak epoch is unselectable
- **DomainBed (Gulrajani & Lopez-Paz, ICLR 2021)** and the OOD-rank literature
  (e.g. "Understanding and Testing Generalization… on OOD Data," arXiv:2111.09190): the
  **in-distribution rank of models does not represent the OOD rank** — so a validation maximum
  need not coincide with the test maximum, and the test-peak checkpoint is not selectable from
  validation. Direct support for "the peak epoch is unobservable; select a deployable checkpoint
  by a robust val signal and ensemble away the residual variance."
- **WILDS (Koh et al., ICML 2021)** (arXiv:2012.07421): standardises the finding that
  in-distribution validation is a weak/again-unreliable selector under real-world shift.
- **Gap in the *temporal-KG* literature (negative result, useful for positioning):** searches over
  temporal-KG evaluation (TGB 2.0 arXiv:2406.09639; "On the Evaluation of Methods for Temporal KG
  Forecasting"; temporal-split tutorials) return careful **leakage-avoiding chronological splits**
  but **no discussion of peak-epoch / checkpoint-selection instability or val/test decorrelation**.
  So the *problem we name* is essentially unaddressed in time-split KG evaluation — the practices
  we import come from the DG/RL-eval literatures, not from KG papers.

---

## What is genuinely novel here (positioning)

The four practices are individually standard *somewhere*, but:

1. **No biomedical-KG / clinical-trial-ranking paper combines them.** The clinical-advancement
   line (Czech et al.; BenevolentAI R2E; Aliper et al.) reports pooled AUROC/AP and single runs;
   the KG-ensemble paper (PMC9677479) ensembles for variance, not for temporal-shift decorrelation,
   and doesn't do rank fusion or robust per-group reporting.
2. **The transplant is the contribution:** importing (a) EoA-style multi-seed val-selected
   ensembling, (b) RRF-style scale-free rank fusion, and (c) Statistical-Precipice-style
   median+breadth+interval reporting, to handle (d) the val/test decorrelation that *per-instance
   temporal masking* induces in low-base-rate Phase-II→III ranking. That specific decorrelation
   under decision-aligned masking is, per the negative result above, not treated in KG eval work.

**Honest framing for the paper (suggested sentence):**
> "Each component of our evaluation protocol has precedent — multi-seed validation-selected
> ensembling improves in-/out-domain rank correlation under distribution shift [EoA, NeurIPS'22];
> rank-based fusion is scale-free and beats score combination [RRF, SIGIR'09]; and per-group
> medians with interval estimates are robust to a few outlier groups where the mean is not
> [Agarwal et al., NeurIPS'21]. We are not aware of prior work that combines them to address the
> validation/test decorrelation that decision-aligned temporal masking induces in low-base-rate
> clinical-advancement ranking."

## Candidate citations to add to reference.bib (verify DOIs before final)
- Arpit et al. 2022, *Ensemble of Averages*, NeurIPS — arXiv:2110.10832
- Cormack, Clarke, Büttcher 2009, *Reciprocal Rank Fusion*, SIGIR — doi:10.1145/1571941.1572114
- Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice*, NeurIPS — arXiv:2108.13264
- Gulrajani & Lopez-Paz 2021, *In Search of Lost Domain Generalization*, ICLR — arXiv:2007.01434
- Koh et al. 2021, *WILDS*, ICML — arXiv:2012.07421
- (domain) *Ensembles of KG embedding models improve predictions for drug discovery*, Bioinformatics 2022 — PMC9677479

## Caveats
- Provenance scan, not exhaustive. arXiv full-text and Semantic Scholar citation-graph chaining
  from EoA / Statistical-Precipice would likely surface a few more domain-generalization
  model-selection papers, but is unlikely to overturn the headline (no single combined paper).
- DOIs/IDs above are from search metadata and should be run through `verify_citations.py` (or a
  manual CrossRef check) before they go into reference.bib.

---

## Diagnosis: pooled RS@10=0 for EAHGT is a `biological_process` artifact (2026-06-18)

While adding encoder baselines, the **pooled** RS@N (all pairs on one global list, Czech
Fig-3 method) showed the rank-fused EAHGT ensembles (score/novelty/both) with **0 advancers
in the global top-10/20/30**, while encoder baselines + OTS caught advancers. Investigated
before writing — root cause found:

- The global pooled top-30 is **100% `biological process`** (30/30) for both rank-fusion AND
  score-averaging — so it is **not** a percentile-fusion artifact.
- **`biological process` is NOT one of the 13 primary TAs** (it's in the paper's exclusion
  list). The EAHGT models score its 211 generic pairs near percentile ~1.0 globally; that TA
  has 62/211 advancers but the very top is non-advancers, so it monopolises and zeros the
  global head.
- **Restricting the pool to the 13 primary TAs** (what the paper's metric does) gives
  EAHGT-both pooled RS@10=3.90, @20=6.57, @30=5.71, @50=5.32, @100=4.61 — strong and sensible.

**Conclusion:** non-issue for the paper (it never evaluates on `biological_process`); the
full-pool eval CSV just doesn't pre-filter to primary TAs before pooling. No fusion/code change
needed (score-avg has the same artifact). Worth ONE caveat sentence in the paper: EAHGT pushes
the excluded, generic `biological_process` associations to the global top, which is itself a
reason the evaluation is TA-stratified rather than a single global ranking.

**Encoder-family ablation (26.03, all 5-seed rank-fused, TA-mean RS):** HGT 3.76/2.72/2.39,
GATv2 4.67/3.08/3.08, R-GCN 5.10/1.84/1.82, CompGCN 1.26/1.92/2.00 (@10/@50/@100) vs EAHGT-both
5.03/3.38/2.68. Encoders competitive on MEAN (GATv2@100 mean 3.08>both), but their wins are
TA-concentrated; on per-TA MEDIAN, EAHGT-both leads every cutoff (3.72/3.13/2.72). Results in
`headline_results/full_ablation_eval/`.

### Refined mechanism (why biological_process tops the global list) — 2026-06-18

Checked whether it's a "not in the grouped loss" effect (user hypothesis). It is NOT excluded
from the loss: with `group_all_tas: true`, the grouping whitelist is EVERY real TA except the
synthetic "all" (train_advancement_lambdarank.py:219-227), so `biological_process` DOES get
slates + gradients. The actual mechanism is a *failure to learn* this generic bucket:

- Within `biological_process`, the model ranks **advancers BELOW non-advancers** (mean fused
  percentile: advancers 0.388 vs non-advancers 0.668) — it is actively mis-ordered.
- The 33 bp pairs at the global top (fused>0.99) are **0 advancers / 33 non-advancers**.
- bp mean percentile 0.586 ≫ median 0.416 → a subset of bp is shoved to the extreme top while
  most sit low.

Why: `biological_process` is an ontology bucket, not a clinical area (211 pairs, 29% base rate,
generic disease embeddings). The grouped LambdaRank loss only enforces *within-TA* ordering and
even that fails here; the cross-TA percentile scale then surfaces its mis-ranked non-advancers
globally. This is exactly why the paper EXCLUDES `biological_process` from evaluation — the
pooled-RS@10=0 artifact validates that exclusion rather than indicating a model defect on the
populations actually scored. Harmless to all reported (primary-TA) numbers.

### Fix applied (2026-06-18): pooled RS restricted to primary TAs, uniformly

Patched evaluate_advancement.py: both pooled paths (`rs_pooled` Katz-CI curve and the
`_by_stratum_pooled.csv`) now pass `disease_filter = {diseases in >=1 primary TA}`, applied
BEFORE computation — same population as the TA-averaged metric. Not "drop a group from pooling":
the TA exclusion is applied once, up front, to both metrics. (biological_process is 650/657
exclusive to excluded TAs, so the filter is clean.) Re-ran eval (job 12850379, COMPLETED).

Corrected pooled RS@N (primary TAs), all 5-seed rank-fused ensembles:
| model | @10 | @20 | @30 | @50 | @100 |
| OTS   |7.80 |4.55 |3.47 |2.87 |2.23 |
| RDG   |3.88 |4.55 |3.91 |3.14 |2.23 |
| HGT   |0.00 |1.94 |2.16 |2.08 |2.50 |
| GATv2 |9.12 |5.21 |3.47 |2.87 |2.50 |
| RGCN  |7.80 |3.90 |2.60 |3.14 |2.63 |
| CompGCN|0.00|0.64 |1.72 |2.34 |1.96 |
| E-score|3.88|3.90 |3.91 |3.94 |2.91 |
| E-nov |0.00 |0.00 |0.00 |0.00 |0.77 |  <- weak pooled, confirms TA-mean was an outlier mirage
| E-both|3.88 |6.54 |5.69 |5.30 |4.58 |  <- best pooled @20-100

NOTE: this also shifts the headline RS-by-limit figure's Katz-CI band (uses rs_pooled), now
primary-TA-restricted — correct + consistent. @10 is near-degenerate (10 of ~7850 pairs).

---

## Display precedents for the mean-vs-robust / aggregation-flip phenomenon (2026-06-18)

User asked for papers that DESCRIBE this phenomenon and how they DISPLAY it.

- **The Benchmark Lottery — Dehghani et al. 2021 (arXiv:2107.07002).** Direct match: model
  rankings flip with task selection + aggregation. Displays: (a) rank-correlation HEATMAP of
  aggregate-vs-individual-task (Kendall τ; some tasks ≈0.60 or NEGATIVE → best-on-average ≠
  best-per-task); (b) "TOP-3 models computed multiple ways" TABLE (rows = aggregation schemes,
  cols = Rank-1/2/3; winner changes per row); (c) #unique-rankings vs #tasks-selected curve.
- **rliable / Statistical Precipice — Agarwal et al. 2021 (already in bib).** "Mean dominated by a
  few high-variance tasks; median unaffected even if half the tasks are 0." Displays:
  (a) AGGREGATE-METRICS bar plot — Median / IQM / Mean side-by-side per model with 95% bootstrap
  CIs (mean≫median = spiky); (b) PERFORMANCE PROFILES — x=threshold τ, y=fraction of tasks with
  score>τ, CI bands; curve crossings show where rankings flip; (c) probability-of-improvement matrix.
- **Macro vs micro averaging** (textbook framing): pooled≈micro, TA-mean≈macro; they disagree /
  reverse under group imbalance. Our pooled-RS vs TA-mean divergence is exactly this.

**Chosen display for THBKG ablation (reuse existing infra, cite precedents):**
1. Table 4 compact: RS@{10,50,100} mean + median@50 + #TA>RDG@50 + AUROC/AP (mean-vs-median cols
   = rliable aggregate-metrics idea in table form; exposes the spikes).
2. Promote existing rs_distributions_ta boxplot (per-TA median+IQR) as the performance-profile
   analog — E-nov/GATv2 spiky, E-both tight/high.
3. Prose sentence (Benchmark-Lottery move): "the top model changes with aggregation — GATv2 by
   pooled@10, EAHGT-novelty by TA-mean@10, EAHGT-both by per-TA median at every cutoff" — cite
   Dehghani + Agarwal so the divergence reads as a known phenomenon, not special pleading.

Citation to ADD: Dehghani et al. 2021, The Benchmark Lottery, arXiv:2107.07002.

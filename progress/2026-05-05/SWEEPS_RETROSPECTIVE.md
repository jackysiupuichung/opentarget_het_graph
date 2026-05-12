# Sweeps Retrospective — early-stop, LR, loss

Three tuning sweeps were run on p3_eahgt_both during this period to
chase the post-peak-collapse / brittle-selection pattern. All three
are now **retired** (configs + scripts + scratch run dirs removed);
this doc records what each found and what was decided. The
individual write-ups [EARLY_STOP_SWEEP.md](EARLY_STOP_SWEEP.md),
[LR_SWEEP.md](LR_SWEEP.md), [LOSS_SWEEP.md](LOSS_SWEEP.md) are kept
for detail.

## 1. Early-stop metric sweep

Tested `ndcg@10`, `ndcg@50`, `ndcg_ta_mean@10`, `ndcg_ta_mean@50`
as early-stop metrics.

**Finding:** flat `ndcg@10` is degenerate (zero 7/8 epochs);
`ndcg@50` is a more stable *signal* (non-zero 6/8 epochs) though it
selected the same epoch in that run. TA-grouped NDCG variants were
well-behaved as signals but **anti-correlated with test RR** — the
val/test temporal mismatch makes per-TA val metrics poor estimators
of per-TA test metrics.

**Decision (shipped):** `early_stopping.metric = ndcg@50` is now the
project default (hard-coded in the trainers; configs also set it
explicitly). Do **not** early-stop on TA-grouped metrics.

## 2. LR sweep

Halved (2.17e-4) and quartered (1.09e-4) the canonical LR, ran 50
epochs.

**Finding:** lowering LR shifts the test peak by 0–2 epochs but
**doesn't raise the peak height and doesn't soften the collapse**.
All LRs collapse from ~6–7 down to ~1.5 by epoch ~30. Best-epoch
selection also breaks at long horizons (epoch 44 or epoch 1
selected under low LR). The pattern is **distribution shift** (train
≤2010, test ≥2019), not classical overfitting.

**Decision:** keep canonical LR (4.35e-4). Don't lower it. The real
lever is the val/test temporal boundary — deferred experiment.

## 3. Loss-implementation sweep

Compared three LambdaRank implementations: in-house flat,
allRank flat (`[1, B]` slate), allRank grouped (one slate per
primary TA).

**Initial finding (single seed):** `allrank_grouped` selected epoch
3 with `rr_ta_mean@50 = 7.71` — +24% over the in-house canonical
6.19. Looked like a clean SOTA improvement.

**Follow-up (3-seed re-run):** the 7.71 did **not** replicate.
Three fresh seeds gave `rr_ta_mean@50 ∈ {5.46, 1.72, 4.30}` — mean
≈ 3.8, std ≈ 1.9 — and a re-run of the *same* seed 42 landed at
1.72 (CUDA non-determinism alone moved it from 7.71 to 1.72). The
grouped loss is **on average no better than, and possibly worse
than, the flat variants**. The variance comes from the same
selection-brittleness the LR sweep diagnosed: tiny float
perturbations flip which epoch gets selected, and the selected
epoch swings between near-the-peak and deep-in-the-collapse.

**Decisions (shipped):**
1. **Retired the in-house `lambdarank_loss`** (`src/losses/lambdarank.py`
   deleted). It was a less-tested implementation of the same flat-
   slate math allRank provides.
2. **`impl: allrank` (flat) is the project default** in all 8
   experiment configs (p1/p2/p3, b1/b3/b6/b7, undirected_v1). The
   `allrank_grouped` impl is kept available as opt-in but not
   recommended given its seed brittleness.
3. **`ndcg_k = 50`** for the loss truncation (matches the early-stop
   metric and the reported operating point).

## Net state after this period

- Loss: allRank `lambdaLoss` flat, `lambdaRank_scheme`, `ndcg_k=50`.
- Early stop: `ndcg@50`, `patience=5`.
- LR: canonical 4.35e-4 (unchanged).
- The headline MODEL_COMPARISON.md numbers were generated with the
  *retired* in-house loss; they should be regenerated with `allrank`
  before publishing — but no single-seed lift is expected over the
  in-house numbers (the two flat losses are mathematically very
  close).

## Still open — the real problem

None of the three sweeps improved on the canonical recipe because
all three are downstream of the **val/test temporal mismatch**
(val 2011–2018, test 2019+). The model overfits the 2010-era graph
structure; no LR, loss, or early-stop change fixes that. The next
experiment that could actually move the needle is narrowing the val
window (or moving the train cutoff) so the val signal is predictive
of test performance. That's deferred.

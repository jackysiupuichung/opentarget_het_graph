# LR Sweep — p3_eahgt_both

Goal: test whether the train-loss-down-test-metric-up-then-collapse
pattern from [EARLY_STOP_SWEEP.md](EARLY_STOP_SWEEP.md) is driven by
overfitting from too-large LR. Halve and quarter the canonical LR;
run **50 epochs with early stopping disabled** so we see the full
trajectory.

Recipe is otherwise canonical p3_eahgt_both. Output dirs:
`/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/lr_sweep/`.

## Configurations

| # | Slug | LR | vs canonical | num_epochs | early_stop |
| --- | --- | --- | --- | --- | --- |
| ref | (canonical p3) | 4.347e-4 | 1× | 50 | enabled (ndcg@10) |
| 1 | `p3_lr_half` | 2.173e-4 | ½× | 50 | enabled (ndcg@50) |
| 2 | `p3_lr_quarter` | 1.087e-4 | ¼× | 50 | enabled (ndcg@50) |

(`early_stop.enabled: false` was the original plan; configs ended
up with patience=5 and metric=ndcg@50 active. With non-zero
`val_ndcg@50` showing up regularly under both LRs, patience never
triggered before epoch 50, so we got the full trajectories
anyway. Side benefit: best_epoch is the model that actually got
saved, comparable to MODEL_COMPARISON.md numbers.)

## Selected-checkpoint test metrics

TA-grouped (apples-to-apples with [MODEL_COMPARISON.md](MODEL_COMPARISON.md)):

| Slug | best_epoch | rr_ta_mean@10 | @50 | @100 | ndcg_ta_mean@50 | AUC | AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical (early-stop sweep) | 3 | **6.02** | **6.19** | **4.82** | **0.36** | 0.59 | **0.19** |
| p3_lr_half | 44 | 3.77 | 1.62 | 1.70 | 0.14 | 0.62 | 0.12 |
| p3_lr_quarter | 1 | 3.63 | 4.34 | 5.13 | 0.22 | 0.62 | 0.14 |

The half-LR best_epoch is **44**, but the test peak was at epoch 2.
The selected checkpoint is therefore deep in the overfit regime.
The quarter-LR best_epoch is 1, with a degenerate `val_ndcg@50 =
0.548` driven by tied scores at initialisation — selection is also
broken, but in a different way.

## Per-epoch trajectory — test_rr_ta_mean@50

| epoch | half (lr=2.17e-4) | quarter (lr=1.09e-4) | reference: canonical (lr=4.35e-4) |
| --- | --- | --- | --- |
| 1 | 5.02 | 4.34 | 6.28 |
| 2 | **6.53** | 4.07 | **7.52** |
| 3 | 5.04 | 6.17 | 6.19 (selected) |
| 4 | 5.49 | **6.42** | 6.19 |
| 5 | 3.89 | 5.68 | 5.91 |
| 6 | 3.77 | 5.17 | 5.41 |
| 7 | 3.11 | 5.20 | 1.72 |
| 10 | 3.85 | 5.24 | — |
| 20 | ~3 | ~5 | — |
| 44 | 1.62 (selected) | ~1.5 | — |
| 50 | 1.65 | 1.48 | — |

Three findings:

1. **Lower LR delays the peak slightly, but doesn't move it much.**
   - Canonical: peak @ epoch 2 (rr_ta_mean@50 = 7.52).
   - Half-LR: peak @ epoch 2 (6.53).
   - Quarter-LR: peak @ epoch 4 (6.42).

   The peak shifts by ~2 epochs but the **peak value is no higher**;
   if anything it's slightly lower. So lowering LR doesn't unlock
   better test performance — it just makes overfitting take a few
   more steps to manifest.

2. **Lower LR doesn't soften the collapse.** All three runs
   eventually crash from ~6 down to ~1.5–1.7 by epoch ~30+.
   Half-LR and quarter-LR both end up at virtually identical
   degraded test metrics by epoch 50. The collapse is gentler in
   timing but identical in magnitude.

3. **Best-epoch selection is broken under low LR + 50 epochs.**
   - Half-LR: `val_ndcg@10` finally fires meaningfully at epoch ~40
     (training catches up to the val signal threshold), so
     selection picks epoch 44 — a checkpoint that has rr_ta_mean@50
     ≈ 1.6 (4× worse than the actual best).
   - Quarter-LR: epoch 1 produces a degenerate `val_ndcg@50 =
     0.548` from tied scores at init, then val NDCG goes to 0 for
     ~40 epochs. Best-epoch = 1 by default; checkpoint is
     essentially random initialisation, with `test_rr@K = 0.000`
     for flat metrics. The TA-grouped numbers look OK (4.34 / 5.13)
     but only because TA-grouped RR is more robust to the tied-at-
     init pathology.

## Implication

**LR is not the lever.** Halving and quartering both:
- shift the peak by 0–2 epochs
- preserve the peak height
- preserve the post-peak collapse magnitude
- additionally break the selection rule

The pattern is consistent with **distribution shift** between train
(≤2010) and test (≥2019), not with classical overfitting. The
model isn't memorising train edges — it's learning the 2010-era
graph structure increasingly precisely, and that structure is not
exactly what's true in 2019+. Lowering LR slows that adaptation
slightly without changing where it's going.

What this rules out:
- "We're overshooting; smaller LR will fix it." — No.
- "Training too long destroys the model." — Yes, but lower LR
  doesn't change that.
- "Multi-seed at canonical LR will recover bigger numbers." — Most
  likely no; the seed noise band is roughly ±1 on rr_ta_mean@50,
  and we can already see 7.5 from canonical's epoch-2 peak. The
  question is which checkpoint to keep, not whether better
  checkpoints exist.

What this is consistent with:
- The "stop at epoch 2–3" outcome being roughly optimal for this
  recipe + this val/test split.
- Increased patience or 50-epoch runs **hurt** test metrics because
  they push past the test peak.
- The val/test temporal mismatch being load-bearing (val 2011–
  2018, test 2019+) — narrowing the val window or the test window
  is the next experiment that could move the needle.

## Recommendation

1. **Keep canonical LR (4.35e-4).** Don't lower it.
2. **Keep `early_stopping.metric: ndcg@50` with `patience: 5`.** 
   The current setup picks epoch 3, which is within ~15% of the
   epoch-2 peak — acceptable.
3. **Consider adding `num_epochs: 8`** as a hard ceiling on
   training. With patience=5 from epoch 3, training can wander to
   epoch 8 even on flat/zero val — the cap saves wasted compute and
   prevents the "epoch 44 selected" pathology from quarter-LR-style
   trajectories if anyone changes other hyperparams.
4. **The next thing to try is *not* an LR change — it's the val/
   test boundary.** Either narrow the val window to the latest pre-
   test years, or change the train cutoff to bring the model closer
   to the test era. That's the only lever that actually addresses
   the distribution-shift hypothesis.

## Files

- Configs: [config/experiments/lr_sweep/](../../config/experiments/lr_sweep/)
- Scripts: [scripts/advancement_prediction/lr_sweep/](../../scripts/advancement_prediction/lr_sweep/)
- Outputs: `/gpfs/scratch/bty414/opentarget_evidences/23.06/runs/lr_sweep/`

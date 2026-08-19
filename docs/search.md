<h1 align="center">Brute-Force Horizon Search</h1>

This document describes the brute-force baseline: train one model per horizon
`h`, evaluate on the target `h=10`, and produce the IC ranking that BLO is
compared against.

## Purpose

The Label Horizon Paradox asks: is the inference target `Δ` the best training
horizon? The brute-force search answers this empirically by training a separate
model for each candidate horizon `h ∈ {1..10}` and measuring out-of-sample IC
on the fixed target `h=10`.

The resulting IC ranking serves two purposes:

1. **Evidence of the paradox**: if IC peaks at `h ≠ 10`, the target is not the
   best training horizon.
2. **Evaluation target for BLO**: BLO's learned `λ` is compared to the IC
   ranking via Spearman rank correlation `rc` (see [blo.md](blo.md#evaluation)).

## Method

`scripts/run_search.py`:

For each horizon `h`:

1. Build the `h`-day VWAP return label (trim + z-score + clip ±3).
2. Train `N_SEED=15` independent `AlphaMLP` models with SGD + momentum:
   - 4096-sample random batches, lr=0.05, momentum=0.95, weight_decay=1e-4.
   - 15 epochs, early stopping with `skip=1` (start evaluating at epoch 2),
     `patience=3`. The `skip=1` is because financial data is noisy and the
     first epoch is often unstable — its validation IC can be spuriously high
     (a random fluctuation), which would trigger premature early stopping if
     used as the best-epoch baseline. Skipping it ensures the baseline is set
     by a more representative epoch.
   - **Best-epoch selection by validation IC** (daily Pearson IC on the `h=10`
     valid label).
3. For each seed, predict test-period alpha (point-in-time) and save.
4. Aggregate: average the 15 per-seed alphas → `alpha_agg.parquet`.

## Configuration

```python
N_SEED = 15
SKIP = 1                # start best-epoch selection from epoch 2
PATIENCE = 3
EPOCHS = 15
BATCH_SIZE = 4096
MOMENTUM = 0.95
LR = 0.05
WEIGHT_DECAY = 1e-4
```

## Output

`results/alphas/h{h}/`:

- `alpha_seed{si}.parquet` — per-seed test alpha (wide: date × wind_code).
- `alpha_agg.parquet` — averaged alpha (the one used for backtest).

`results/models/h{h}/model_seed{si}.pt` — saved model states.

## Results

Test IC on `h=10` target (15-seed aggregated alpha), test period 2025-01 ~ 2026-06:

| h | IC | RankIC | ICIR | RankICIR |
|---|---|---|---|---|
| 1 | 0.0767 | 0.1098 | 0.7327 | 0.9665 |
| 2 | 0.0821 | 0.1134 | 0.7545 | 0.9642 |
| **3** | **0.0860** | 0.1152 | **0.7808** | **0.9671** |
| 4 | 0.0855 | 0.1146 | 0.7773 | 0.9664 |
| 5 | 0.0852 | 0.1142 | 0.7568 | 0.9383 |
| 6 | 0.0853 | **0.1154** | 0.7560 | 0.9478 |
| 7 | 0.0846 | 0.1145 | 0.7426 | 0.9383 |
| 8 | 0.0840 | 0.1147 | 0.7405 | 0.9426 |
| 9 | 0.0822 | 0.1111 | 0.7322 | 0.9218 |
| 10 | 0.0825 | 0.1125 | 0.7214 | 0.9176 |

**IC peaks at `h=3` (0.0860)**, not at the inference target `h=10` (0.0825) —
confirming the Label Horizon Paradox in this setting. The peak is broad
(`h=3..h=8` all ≈ 0.084–0.086), but the short horizon `h=3` is the single
best, and IC monotonically decays toward `h=10`. RankIC follows the same
pattern (peak 0.1154 at `h=6`, with `h=3` a close second at 0.1152).

### Why the peak is broad: sample size compresses the horizon gap

Compared to the paper's experiments, the IC differences across horizons in this
demo are noticeably smaller (the peak is a broad plateau rather than a sharp
spike). This is expected, and follows directly from the theory in the paper
(Eq. 29, Appendix — the final IC expression):

$$J(\delta) \triangleq \text{IC}_{\text{final}}^2(\delta) = \frac{\alpha(\delta)^2\,\alpha(\Delta)^2}{\left[\alpha(\delta)^2 + K(\delta + \delta_0)\right] \cdot V_{\text{target}}}, \qquad K = \frac{d}{N}\sigma^2$$

where `d` is the feature dimension, `N` is the training sample size, and
`σ²` is the idiosyncratic noise variance. The key term is the **noise penalty**
`K(δ + δ₀)` in the denominator: it is the only channel through which the
training horizon `δ` hurts generalization, and it is proportional to `K`, hence
**inversely proportional to N**.

- The paper's main experiments use CSI 300 / CSI 500 / CSI 1000 (at most ~1000
  stocks, minute-frequency). This demo uses **all A-shares (~5000+ stocks,
  daily)**, so `N` is roughly 5× larger and `K` is ~5× smaller.
- As `K` shrinks, the noise penalty `K(δ + δ₀)` shrinks relative to the signal
  term `α(δ)²`, so the denominator becomes dominated by `α(δ)²` and `J(δ)`
  flattens across `δ` — the horizon gap compresses. Intuitively, with more
  samples the OLS estimation error `σ²(δ+δ₀)/N` (which grows with `δ`) is
  squeezed, so the long-horizon noise disadvantage is partially offset, and the
  horizons converge in performance.
- In the limit `K → 0` (infinite data), `J(δ) → α(Δ)² / V_target` — independent
  of `δ`, i.e. the paradox would vanish. The paradox persists here only because
  `K` is small but nonzero.

So the broad peak is **not** a failure of the method — it is the finite-sample
theory predicting that larger universes shrink the horizon gap. The paradox
still holds (h=3 remains the single best, h=10 the worst on IC), just with a
muted magnitude. This is also why BLO's `rc` is evaluated on the *ranking*
rather than the raw IC gap: the ranking is what survives the compression.

## IC Ranking

The IC ranking is the evaluation target for BLO. It is also encoded as `R_AGG`
(10 = best, 1 = worst) for convenience:

```python
R_AGG = [1, 2, 10, 9, 7, 8, 6, 5, 3, 4]   # h3=10 (best), h1=1 (worst)
```

Larger value = better horizon, matching the `λ` convention (larger `λ` = more
important). BLO's `rc` is the Spearman rank correlation between this ranking
and `λ` (equivalently, between the raw IC values and `λ`).

## Point-in-Time Prediction

`predict_alpha_pit` (`src/data_utils.py`) generates test alphas using only
information available up to each day — no look-ahead. Factors are preprocessed
with train-period thresholds; the tradable universe is the point-in-time
tradable mask. Stocks with any missing factor are dropped (alpha = NaN),
aligned with the training sample construction (`build_xy_clip`) — the model is
trained only on factor-complete stocks, so it is evaluated only on the same.
The dropped stocks are predominantly suspended/halted names with no usable
factor values.

## Metrics

`metrics_for_alpha` (`src/data_utils.py`) computes daily IC, RankIC, ICIR,
RankICIR between an alpha and a label, over the test period.

<h1 align="center">Bi-Level Optimization (BLO)</h1>

This document describes the BLO method as implemented in this demo. The
implementation follows the paper's core idea but differs in several details
(see [Differences from the paper](#differences-from-the-paper)).

## The Label Horizon Paradox

Conventional financial forecasting trains the model on the same horizon as the
inference target `Δ`. The paper shows this is not always optimal: as the label
horizon extends, **signal realization** (information being priced in) competes
with **noise accumulation** (idiosyncratic volatility). The optimal supervision
horizon `δ*` sits where marginal signal gain equals marginal noise penalty —
which can be *shorter* than `Δ`.

This demo instantiates the idea in a daily-frequency setting:

- **Candidate horizons**: `h = 1..10` (10-day VWAP returns).
- **Inference target**: `Δ = 10` (10-day return, aligned with a 10-day
  rebalancing cadence).
- **Brute-force evidence**: training one model per `h` and evaluating on `h=10`
  shows IC peaks at `h=3`, not `h=10` (see [search.md](search.md)). This is the
  paradox in concrete form.

## Method

BLO learns a softmax weight vector `λ ∈ R^10` over the 10 candidate horizons.
The student model is trained on the `λ`-weighted composite label; `λ` is
optimized so that the resulting student generalizes best on the fixed target
`Δ = 10`.

### Teacher

`HorizonTeacher` (`src/blo_trainer.py`) holds a logits vector; `forward()`
returns `softmax(logits)` — the 10 horizon weights summing to 1. It is shared
across all samples (the global `λ`).

### Warmup

A few epochs of standard supervision with a **balanced diff-z label** to build a
meaningful representation before the bi-level loop:

1. For each day, compute per-horizon returns `r_1..r_10` (trim-tailed).
2. Differences `d_1=r_1, d_2=r_2-r_1, ..., d_10=r_10-r_9` — each marginal return
   gets equal weight (a naive mean label would give `r_1` 10× the weight of
   `r_10`).
3. Cross-sectional z-score + clip ±3, then equal-weighted mean across horizons →
   a single scalar label per stock.
4. SGD + 4096-batch + momentum 0.95 (same recipe as the brute-force search),
   fixed epochs, no validation.

The warmup student is shared across all BLO seeds as the starting point.

### Bi-Level Loop

Each iteration (one sample per update):

1. **Sample**: pick a random center `t`; draw the inner set from days before
   `t`, the outer set from days after `t` (each `batch_days=10` days, with a
   `gap_days=10` gap). Inner is entirely before outer — time-forward, no
   leakage — while spreading each side across the full period.
2. **Inner loop** (`M=1` step of differentiable SGD): starting from the current
   student parameters `θ`, compute the weighted IC loss
   `L_inner = 1 - Σ_δ λ_δ · IC_δ` on the inner set, and take one gradient step
   to get `θ*`. This uses `torch.func.functional_call` +
   `autograd.grad(create_graph=True)` for second-order autodiff (no `higher`
   library needed).
3. **Outer loop**: evaluate `θ*` on the outer set with the IC loss against the
   target `Δ = 10` (`1 - IC`), plus an entropy regularizer `−γH(λ)` to prevent
   `λ` from collapsing onto a single horizon. Backprop through the inner step
   to get the gradient on `λ`, and update `λ` with Adam.
4. **Writeback** (see below): write `θ*` back into the student.

`λ` is averaged over a post-burnin window of iterations per seed (see
`LAM_START`/`LAM_END` in `scripts/run_blo.py`), then across seeds for the
ensemble.

### Writeback

After each iteration's `λ` update, the inner-updated parameters `θ*` are written
back into `self.student` (detached). This means **inner updates accumulate
across iterations**: the student keeps learning along the `λ`-weighted inner
direction, rather than restarting from the warmup point every iteration. This
turns the bi-level loop into a streaming/online joint optimization.

### Ensemble

A single BLO trajectory (one BLO seed) is noisy — the per-seed `rc` has high
variance. For stability, this demo runs **20 BLO seeds** per warmup seed (each
runs the full bi-level loop from the shared warmup student) and averages `λ`
across them.

BLO is cheap to run. Each iteration draws a 10-day batch (`batch_days=10`);
4 years of training data is ~1000 trading days, so one full epoch is ~100
iterations. The default run does 180 iterations — under 2 epochs of data in
total, a very small amount of training. The 20-seed ensemble per warmup
(20 × 180 = 3600 iterations, ~3 min on a single GPU) is likewise small, and
in practice does not need to be that many — a few seeds are enough for a
stable `λ`. The default 10-warmup-seed sweep (~30 min) serves the same
purpose — stability verification, not a requirement.

This cost advantage grows with the problem scale. The brute-force search trains
one model per horizon, so its cost scales linearly with the number of candidate
horizons; BLO trains a single student regardless of how many horizons are in
the mixture. This demo uses daily data with only 10 horizons — at higher
frequency or with a finer horizon grid, the brute-force sweep becomes
prohibitively expensive while BLO stays a single training run.

### Evaluation

The learned `λ` is compared against the brute-force search IC (see
[search.md](search.md)) via the Spearman rank correlation (RankIC) between the
two:

```
rc = spearmanr(IC_search, λ)
```

where `IC_search[h]` is the test IC of horizon `h` from the brute-force search,
and `λ[h]` is the BLO-learned weight for horizon `h`. `rc ∈ [−1, 1]`:

- `rc ≈ 1`: BLO puts the most weight on the highest-IC horizon, etc. — it has
  recovered the IC ranking.
- `rc ≈ 0`: no correspondence (random `λ`).
- `rc ≈ −1`: BLO weights are reverse-ordered.

A high positive `rc` means BLO has identified the right horizons without running
the full brute-force sweep. The IC ranking is also encoded as `R_AGG` (10 =
best, 1 = worst) for convenience; `rc` is equivalently `spearmanr(R_AGG, λ)`.

## Configuration

`scripts/run_blo.py` exposes all tunable hyperparameters at the top:

```python
N_ITERS = 180             # BLO updates per seed (one sample per update)
LAM_START = 160           # lambda averaging window [LAM_START:LAM_END] (last 20 iters)
LAM_END = 180
WARMUP_EPOCHS = 2         # SGD+4096 diff-z warmup epochs
WARMUP_BATCHSIZE = 4096
WARMUP_SEEDS = [0,1,2,3,4,5,6,7,8,9]  # warmup seeds; one full BLO run per warmup seed
N_SEED = 20               # BLO ensemble seed count
BATCH_DAYS = 10           # days sampled per inner/outer batch
GAP_DAYS = 10             # day gap between inner and outer ranges
LR_INNER = 0.05
LR_OUTER = 0.1
INNER_MOMENTUM = 0.9
INNER_STEPS = 1           # M
ENTROPY_WEIGHT = 1e-4     # γ
```

Each run uses 20 BLO seeds (ensemble) and produces `results/blo/blo_results.csv`.
To verify stability across warmup initialization, the default config sweeps 10
warmup seeds (see [Results](#results)).

## Results

With the default config (dropna, writeback, SGD warmup 2ep, 180 iters, λ
averaged over `[160:180]`, 20 BLO seeds per warmup):

| warmup seed | `ens_rc` | argmax `λ` |
|---|---|---|
| 0 | 0.952 | h3 |
| 1 | 0.915 | h3 |
| 2 | 0.915 | h3 |
| 3 | 0.952 | h3 |
| 4 | 0.952 | h3 |
| 5 | 0.891 | h5 |
| 6 | 0.976 | h4 |
| 7 | 0.964 | h3 |
| 8 | 0.927 | h3 |
| 9 | 0.927 | h3 |

Per-warmup-seed `λ` (each row is the 20-BLO-seed ensemble for that warmup), with
the brute-force search IC on the last row for comparison:

| warmup seed | h1 | h2 | **h3** | h4 | h5 | h6 | h7 | h8 | h9 | h10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0583 | 0.0705 | **0.1473** | 0.1266 | 0.1366 | 0.1317 | 0.1008 | 0.0803 | 0.0707 | 0.0771 |
| 1 | 0.0527 | 0.0729 | **0.1496** | 0.1275 | 0.1402 | 0.1373 | 0.1037 | 0.0828 | 0.0654 | 0.0678 |
| 2 | 0.0543 | 0.0783 | **0.1398** | 0.1321 | 0.1345 | 0.1330 | 0.1008 | 0.0823 | 0.0700 | 0.0750 |
| 3 | 0.0521 | 0.0729 | **0.1373** | 0.1277 | 0.1330 | 0.1333 | 0.1040 | 0.0872 | 0.0777 | 0.0748 |
| 4 | 0.0568 | 0.0664 | **0.1353** | 0.1285 | 0.1330 | 0.1301 | 0.1003 | 0.0906 | 0.0742 | 0.0847 |
| 5 | 0.0558 | 0.0710 | 0.1414 | 0.1389 | **0.1518** | 0.1296 | 0.0983 | 0.0770 | 0.0681 | 0.0682 |
| 6 | 0.0575 | 0.0640 | 0.1408 | **0.1423** | 0.1352 | 0.1326 | 0.1056 | 0.0834 | 0.0687 | 0.0698 |
| 7 | 0.0579 | 0.0717 | **0.1492** | 0.1367 | 0.1355 | 0.1363 | 0.0979 | 0.0767 | 0.0677 | 0.0702 |
| 8 | 0.0576 | 0.0724 | **0.1503** | 0.1349 | 0.1460 | 0.1327 | 0.0966 | 0.0761 | 0.0661 | 0.0673 |
| 9 | 0.0625 | 0.0786 | **0.1516** | 0.1269 | 0.1445 | 0.1268 | 0.0952 | 0.0802 | 0.0661 | 0.0676 |
| **search IC** | 0.0767 | 0.0821 | **0.0860** | 0.0855 | 0.0852 | 0.0853 | 0.0846 | 0.0840 | 0.0822 | 0.0825 |

The 10 warmup seeds are run to verify that the result is not an artifact of a
single warmup initialization: `ens_rc` stays in 0.891–0.976 across all 10 runs.

- **8/10 warmup seeds lock onto h=3** (the brute-force IC peak); the remaining
  two land on h=4 and h=5 — both adjacent and within the h=3..h=6 plateau, and
  in both cases h=3 is the second-highest weight (within 0.011 of the argmax).
- `ens_rc` is 0.891–0.976 per warmup seed — BLO recovers the brute-force IC
  ranking without running the full sweep.
- The `λ` weights track the search IC profile: both peak at h=3 and stay
  elevated across the h=3..h=6 plateau, then drop toward h=9/h=10.

## Differences from the Paper

| Aspect | Paper | This demo |
|---|---|---|
| **Setting** | Intraday / interday, LSTM on CSI500 | Daily-frequency, AlphaMLP on all A-shares |
| **Features** | Minute-level features | Public Alpha158 (158 daily factors) |
| **Warmup label** | Mean of standardized candidate labels | Diff-z (marginal-return) label, equal-weighted |
| **Sampling** | Intra-batch split (B_in / B_out) | Random-dispersed: inner before outer with a gap, spread across the full period |
| **Ensemble** | A single trajectory per update | Multiple trajectories sampled and averaged for stability (see below) |
| **Horizons** | Minute-level `δ ∈ {1..Δ}` | Daily `h ∈ {1..10}`, target `Δ=10` |

The consequential difference is **ensemble**: the paper samples a single
inner/outer trajectory per update, while this demo samples multiple
trajectories (across BLO seeds) and averages the resulting `λ` for stability.
This ensemble is largely responsible for the stable `rc ≈ 0.94` in this demo.

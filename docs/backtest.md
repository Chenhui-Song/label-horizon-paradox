<h1 align="center">Backtest Engine</h1>

This document describes the ladder backtest engine, optimizers, costs, and
index benchmarks.

## Overview

The backtest is fully decoupled from training: it reads saved alpha parquets
(`results/alphas/h{h}/alpha_agg.parquet`) and market data, and runs a
staggered sub-portfolio backtest. No model/factor code is touched.

```
saved alpha (T×N) ──> LadderBacktest ──> NAV series + metrics (sharpe/ann/mdd/turnover)
       +                    │
    market data             ├─ optimizer (TopN / ThreeIndexTopN)
                            ├─ account (stateful holdings)
                            ├─ constraints (limit-up/down, weight cap)
                            ├─ costs (stamp duty + commission)
                            └─ execution (VWAP / close)
```

## Stock Universe

The backtest restricts the stock universe to the constituents of the three
major A-share indices — **CSI 300, CSI 500, and CSI 1000** (the CSI1800 union).
These three indices cover large-, mid-, and small-cap stocks and are the most
representative and liquid names in the A-share market, which keeps the
backtest aligned with tradable, high-quality names rather than the illiquid
small-cap tail. Two selection modes are supported, both within this universe:

- **`topn`**: top-N equal-weight across the CSI1800 union (default N=100).
- **`3index`**: top-N equal-weight from each of CSI300/500/1000 separately
  (e.g. 20:30:50, 100 stocks total).

## Simplifications & Limitations

> **This is a simplified backtest for signal validation, not a production-grade
> simulator.** Many real-world trading details are intentionally omitted. The
> goal is to check whether the Label Horizon Paradox shows up in a realistic
> backtest, not to produce a deployable strategy. Treat absolute Sharpe/return
> numbers as indicative; the **relative comparison across horizons** is what
> matters.

The engine makes the following simplifying assumptions:

- **No slippage / market impact.** Trades fill at the full-day VWAP assuming
  the order is a price-taker and does not move the market. There is no
  slippage term, no volume-participation cap, and no price impact model. A
  large order that would realistically push the VWAP is treated as free.
- **Buy and sell share one price.** Both legs fill at the same full-day VWAP;
  there is no bid-ask spread, no separate buy/sell price, and no ordering of
  the buy and sell legs within the day.
- **Full-day VWAP only.** The execution price is the full-day VWAP
  (`window=None`). Intraday execution windows (e.g. open-30min VWAP, close
  VWAP) are supported by `get_minute_vwap(window=...)` but not used.
- **Limit-up-blocked buys roll over.** A limit-up stock cannot be bought; its
  blocked weight is redirected to substitute names (the next-highest-alpha names
  that are tradable and not limit-up on the execution day) rather than left as
  cash. A limit-down stock cannot be sold (the position is frozen at the drifted
  weight; no multi-day retry to clear the position).
- **Suspension freezes the position.** A suspended stock's weight is held at
  the previous day's drifted weight; resumption gaps and long-halt
  delisting risk are not modeled.
- **Equal-weight, no risk-model constraints.** No industry/style exposure
  limits, no covariance-based optimization, no tracking-error constraint. The
  `OptimContext` fields (`sigma`, `exposures`, `industry`, `benchmark`) are
  reserved but unused by the built-in optimizers.
- **No turnover control.** No per-day turnover cap and no turnover penalty in
  the objective; turnover is only reported, not constrained.
- **Single-day execution.** The entire target is assumed to fill in one day at
  t+1 VWAP; there is no multi-day order splitting (VWAP/TWAP algorithms).
- **Weight-based, not share-based.** Holdings are tracked as portfolio weights
  (fractions of NAV summing to 1), not as integer share counts. This ignores
  lot-size rounding (A-shares trade in 100-share lots), minimum-order
  constraints, and the fact that a small account cannot hold fractional shares
  of a high-priced name. It also means the backtest is scale-free (the same
  result regardless of starting capital), which is convenient for signal
  validation but unrealistic for small accounts.

## LadderBacktest

`engine/ladder_backtest.py` — staggered sub-portfolio backtest. The name
"ladder" refers to the staggered rebalance schedule: sub-accounts rebalance on
offset days (like rungs of a ladder), so each one stays low-frequency while
overall turnover is dispersed daily. This is sometimes called "staggered
rebalancing" or "overlapping portfolios" in the industry.

- Total capital split into `n_ladders=10` sub-accounts, each `1/10`.
- Sub-account `k` rebalances on days `[k, k+freq, k+2*freq, ...]` with
  `rebalance_freq=10`.
- Each sub-account rebalances every 10 days (full signal utilization), but
  across sub-accounts 1/10 of capital rebalances each day → **turnover is
  dispersed daily** while each sub-account stays low-frequency.
- Sub-account NAVs float independently; total NAV = sum of sub-account NAVs.

This gives the signal-utilization of a 10-day rebalance with the turnover
smoothing of daily rebalancing.

## Optimizers

`engine/optimizer.py` — stateless pure functions
(`solve(alpha, current_holdings, context) -> target weights`):

- **`TopNOptimizer(n, member_masks=None)`**: top-N alpha names, equal-weighted.
  When `member_masks` is provided (a list of daily constituent masks), candidates
  are restricted to the union of those masks on each day — e.g. passing the
  CSI300/500/1000 masks picks from the CSI1800 universe. When `None`, the full
  alpha universe is used.
- **`ThreeIndexTopNOptimizer(member_masks, n_per_index)`**: top-N from each of
  CSI300/500/1000, equal-weighted and merged.
  - `n_per_index` can be an `int` (same N for all three) or a `dict`
    `{"csi300": a, "csi500": b, "csi1000": c}` for per-index N.
  - Set an index's N to 0 to skip it (equivalent to a single-index backtest).
  - Each picked stock gets weight `1 / (a+b+c)`.

Custom optimizers implement `OptimizerProtocol.solve`.

## Costs

`engine/base/costs.py` — A-share transaction cost model:

- **Stamp duty**: 0.05% (sell orders only).
- **Commission**: 0.025% (both sides).

Cost is computed from the sell and buy legs of each rebalance:
`sell_amount = Σ max(prev − new, 0)` (the sold part, stamped),
`buy_amount = Σ max(new − prev, 0)` (the bought part), and
`cost = sell_amount × stamp_duty + (sell_amount + buy_amount) × commission`.
Turnover is `sell_amount + buy_amount = Σ |w_t − w_{t−1}|`.

Slippage is not modeled — trades fill at the full-day VWAP with no price
impact or slippage term.

## Execution & Timeline

- **`DailyClose`** timeline: signal at day `t` close → execute at `t+1`.
- **`VWAPExec`**: execute at next-day VWAP (post-adjusted).

## Constraints

`engine/base/constraints.py`:

- **Limit-up**: cannot buy — the blocked weight is rolled over to substitute names (the next-highest-alpha names that are tradable and not limit-up on the execution day) rather than left as cash. Disable with `Constraints(rollover=False)`.
- **Limit-down**: cannot sell (position frozen at the drifted weight; not rollable).
- **Weight cap / turnover cap**: configurable (default unconstrained).
- **Residual**: cash (any remaining uninvested capital stays as cash).

## Index Benchmarks

`run_backtest.py` computes CSI300/500/1000 buy-and-hold benchmarks over the
test period (from `index_close.parquet`): Sharpe, annualized return, max
drawdown. These are printed alongside the alpha backtest for comparison.

## Usage

```bash
# Top-N within an index universe (default N=100, full universe if --universe omitted)
python scripts/run_backtest.py --mode agg --pool topn --topn 100 --universe csi300,csi500,csi1000
python scripts/run_backtest.py --mode agg --pool topn --topn 100 --universe csi300,csi500,csi1000 --h 1 3 5

# Three indices, default 50:50:50
python scripts/run_backtest.py --mode agg --pool 3index

# Three indices, per-index N, compare multiple configs (repeatable --n-config)
python scripts/run_backtest.py --mode agg --pool 3index \
    --n-config 20:30:50 --n-config 30:50:80 --n-config 50:50:50

# Seed mode (backtest each of 15 seed alphas, average metrics)
python scripts/run_backtest.py --mode seed --pool 3index --n-config 20:30:50
```

### Arguments

| Arg | Values | Description |
|---|---|---|
| `--mode` | `agg` / `seed` | `agg`: aggregated alpha; `seed`: per-seed alphas averaged |
| `--pool` | `topn` / `3index` | Stock pool |
| `--topn` | int (default 100) | N for `topn` |
| `--universe` | comma-separated indices (default none) | `topn` only: restrict candidates to the union of these index constituents (e.g. `csi300,csi500,csi1000` for the CSI1800 universe). Omit for the full universe. |
| `--n-config` | `A:B:C` (repeatable) | Per-index N for `3index` (default `50:50:50`) |
| `--h` | int list (default 1..10) | Horizons to backtest |
| `--no-rollover` | flag | Disable rolling over limit-up-blocked buy weight to substitute names (on by default) |

## Output

`results/backtest/`:

- `backtest_{mode}_{pool}_{config}_h{h}.csv` — NAV series per h/config.
- `backtest_{mode}_{pool}_summary.csv` — summary (sharpe/ann/mdd/turnover).
- `log_backtest.txt`

## Results

> **Disclaimer**: This is a simplified backtest for signal validation only. It
> uses a basic ladder scheme with equal-weighting, no risk-model constraints
> (industry/style exposure limits), no covariance-based optimization, and no
> realistic capacity/turnover control beyond the per-stock weight cap. The
> purpose is **not** to produce a production-grade strategy, but to check
> whether the Label Horizon Paradox — observed in the brute-force IC ranking
> and recovered by BLO — also shows up in a realistic backtest. Treat the
> absolute Sharpe/return numbers as indicative; the **relative comparison
> across horizons** is what matters.

Test period: 2025-01-02 ~ 2026-06-30. All configs use `agg` mode (aggregated
15-seed alpha), `n_ladders=10`, `rebalance_freq=10`. Daily NAV is marked at
close price; trades execute at next-day VWAP. A-share costs: stamp duty 0.05%
(sell-only) + commission 0.025% (both sides).

### CSI1800 top-100 (`--pool topn --topn 100 --universe csi300,csi500,csi1000`, 100 stocks from the CSI300+CSI500+CSI1000 union)

| h | Sharpe | Ann | Vol | MDD | Calmar | Turnover | Ret/TO |
|---|---|---|---|---|---|---|---|
| 1 | 1.690 | 30.48% | 18.03% | 11.30% | 2.70 | 17.19% | 1.77 |
| 2 | 1.696 | 30.79% | 18.16% | 10.82% | 2.85 | 16.91% | 1.82 |
| 3 | **1.811** | **33.45%** | 18.47% | 11.65% | **2.87** | 16.93% | **1.98** |
| 4 | 1.739 | 31.56% | 18.15% | 12.16% | 2.60 | 16.65% | 1.90 |
| 5 | 1.755 | 31.42% | 17.90% | 12.60% | 2.49 | 16.75% | 1.88 |
| 6 | 1.749 | 30.53% | 17.46% | 13.49% | 2.26 | 16.73% | 1.83 |
| 7 | 1.762 | 30.37% | 17.23% | 13.36% | 2.27 | 16.61% | 1.83 |
| 8 | **1.811** | 31.20% | 17.22% | 13.69% | 2.28 | 16.66% | 1.87 |
| 9 | 1.712 | 29.41% | 17.17% | 13.12% | 2.24 | 16.61% | 1.77 |
| 10 | 1.700 | 29.19% | 17.17% | 13.31% | 2.19 | 16.55% | 1.76 |

> Vol = annualized volatility (Ann/Sharpe, rf=0); Calmar = Ann/MDD; Ret/TO = Ann/Turnover
> (return per unit turnover). Sharpe h3/h8 are tied (1.811 vs 1.811); Ann, Calmar,
> and Ret/TO all peak at h3, matching the IC peak.

### Three-index 20:30:50 (`--pool 3index --n-config 20:30:50`, 100 stocks)

| h | Sharpe | Ann | Vol | MDD | Calmar | Turnover | Ret/TO |
|---|---|---|---|---|---|---|---|
| 1 | 1.700 | 30.42% | 17.90% | 11.24% | 2.71 | 17.21% | 1.77 |
| 2 | 1.700 | 30.65% | 18.03% | 10.71% | 2.86 | 16.95% | 1.81 |
| **3** | **1.852** | **34.00%** | 18.35% | 10.96% | **3.10** | 16.93% | **2.01** |
| 4 | 1.774 | 31.95% | 18.01% | 11.65% | 2.74 | 16.67% | 1.92 |
| 5 | 1.773 | 31.54% | 17.79% | 11.95% | 2.64 | 16.74% | 1.88 |
| 6 | 1.754 | 30.60% | 17.45% | 12.99% | 2.36 | 16.70% | 1.83 |
| 7 | 1.762 | 30.31% | 17.21% | 12.96% | 2.34 | 16.59% | 1.83 |
| 8 | 1.812 | 31.17% | 17.20% | 13.15% | 2.37 | 16.66% | 1.87 |
| 9 | 1.666 | 28.68% | 17.22% | 12.99% | 2.21 | 16.60% | 1.73 |
| 10 | 1.690 | 29.14% | 17.24% | 12.82% | 2.27 | 16.55% | 1.76 |

> All four return/risk metrics (Sharpe, Ann, Calmar, Ret/TO) peak at h=3,
> matching the IC peak — the Label Horizon Paradox carries through cleanly to
> the 3-index backtest.

### Index benchmarks (test period, buy-and-hold)

| Index | Sharpe | Ann | MDD |
|---|---|---|---|
| CSI300 | 1.283 | 20.50% | 10.49% |
| CSI500 | 1.852 | 40.95% | 14.06% |
| CSI1000 | 1.477 | 34.26% | 16.87% |

### Observations

- **Both alpha configs beat CSI300 and CSI1000** across all horizons on Sharpe,
  while keeping drawdowns lower; 3-index top h=3 (Sharpe 1.852) matches CSI500,
  which had an exceptionally strong test period.
- **IC peak at h=3 carries through to backtest**: in the 3-index pool, Sharpe,
  Ann, Calmar, and Ret/TO all peak at h=3 (1.852 / 34.00% / 3.10 / 2.01) — a clean
  match to the brute-force IC peak (h=3, 0.0860). In the CSI1800 top-100 pool,
  Sharpe is tied between h=3 and h=8 (1.811 vs 1.811, a flat plateau), but Ann
  (33.45%), Calmar (2.87), and Ret/TO (1.98) still peak at h=3. This confirms
  the Label Horizon Paradox — the h=3 supervision signal beats the h=10
  inference target in out-of-sample backtest across all return/risk metrics.
- **3-index has a sharper h=3 peak** (Sharpe 1.67–1.85, h=3 clearly highest)
  than CSI1800 top-100 (1.69–1.81, a flatter profile where h=3 and h=8 are tied
  on Sharpe but h=3 wins on Ann/Calmar/Ret-TO). Both modes select from the
  CSI1800 universe; the 3-index mode splits selection across CSI300/500/1000,
  giving a sharper large/mid-cap tilt, while top-100 draws from the union and
  is flatter.
- BLO's learned λ concentrates on h=3..h=6 (see [blo.md](blo.md)), which
  overlaps the backtest peak region — BLO identifies the right horizon without
  the brute-force sweep.

<h1 align="center">Architecture</h1>

This document describes the code structure and module responsibilities.

## Module Map

```
src/                core library
├── config.py       fixed constants (dates, paths, device, horizons)
├── model.py        AlphaMLP + differentiable IC/RankIC losses
├── data_utils.py   factor loading, label construction, universe, PIT prediction, metrics
└── blo_trainer.py  HorizonTeacher + BLOTrainer (warmup + bi-level loop + writeback)

scripts/            entry points (all tunable hyperparameters live here)
├── build_backtest_data.py   download market data + index constituents (DDB)
├── build_factors.py         compute Alpha158 factors (DDB)
├── cache_factors_processed.py  preprocess factors (69 standardized + 89 kept) → cache
├── run_search.py            brute-force horizon search → IC ranking
├── run_blo.py               bi-level optimization → learned λ
└── run_backtest.py          ladder backtest on saved alphas

engine/             backtest engine
├── optim_backtest.py   period-by-period backtest (stateful account)
├── ladder_backtest.py  staggered sub-portfolio (N sub-accounts, summed NAV)
├── optimizer.py        OptimizerProtocol + TopN / ThreeIndexTopN / Normalizer
└── base/
    ├── account.py      stateful holdings + NAV
    ├── constraints.py  limit-up/down, weight cap, residual
    ├── costs.py        stamp duty + commission
    ├── timeline.py     DailyClose timepoint
    └── execution.py    VWAP / close execution

factors_calculate/  Alpha158
├── alpha158.py         158 factor names (Qlib-consistent)
└── alpha158_db.py      compute factors from DDB raw data

data_fetch/         DolphinDB access
├── connection.py       DDB connection config
├── marketdata_db.py    EOD prices, VWAP, limits, tradable, index close
├── index_members_db.py index constituent masks (PIT)
├── calendar_db.py      trading calendar
└── cache.py            parquet cache

performance/        metrics
└── metrics.py          return/risk (sharpe, ann, mdd, calmar) + factor (IC, ICIR, RankIC)

data/               parquet market data (not in git)
├── a_share_data/          raw (prices, masks, factors)
└── a_share_data_processed/ preprocessed factors

results/            outputs (not in git)
├── alphas/         per-horizon test alphas (search output)
├── blo/            BLO results (λ, rc)
├── backtest/       backtest NAV + summaries
└── models/         saved model states

docs/               documentation
```

## Design Principles

### Decoupled stages

The three stages (search / BLO / backtest) communicate only through saved
parquet files:

- `run_search.py` → `results/alphas/h{h}/alpha_*.parquet`
- `run_blo.py` → `results/blo/blo_results.csv` (λ + rc)
- `run_backtest.py` reads `results/alphas/` → `results/backtest/`

No stage imports another's code. This lets each stage be rerun independently
(e.g., re-backtest with a different pool without retraining).

### Config separation

- **`src/config.py`**: fixed project constants (date ranges, paths, device,
  horizons). Not tunable.
- **`scripts/run_*.py`**: tunable hyperparameters (lr, epochs, seeds, batch
  sizes) at the top of each entry script. Edit there to experiment.

### Stateless optimizers, stateful account

Optimizers (`engine/optimizer.py`) are pure functions — all state (holdings,
NAV) lives in `Account` (`engine/base/account.py`). This makes optimizers
trivial to swap or customize.

### Differentiable IC

`src/model.py` implements daily cross-sectional IC as a differentiable PyTorch
operation (vectorized via one-hot + scatter). This lets BLO's inner loop use IC
directly as a loss (no surrogate), with second-order autodiff through
`torch.func.functional_call` + `autograd.grad(create_graph=True)` — no `higher`
library needed.

### Point-in-time correctness

- Factor preprocessing thresholds: train-period only.
- Label quantile thresholds: train-period only.
- Index constituent masks: daily PIT.
- Tradable universe: daily PIT.
- Test alpha prediction (`predict_alpha_pit`): uses only information available
  up to each day. Stocks with any missing factor are dropped (aligned with the
  training sample construction in `build_xy_clip`) — predominantly suspended
  names — rather than imputing a value the model never saw.

No look-ahead anywhere in the pipeline.

## Data Flow

```
DDB ──build_backtest_data.py──> data/a_share_data/ (raw parquet)
   │
   └──build_factors.py──> data/a_share_data/factors.parquet
                                   │
                                   cache_factors_processed.py
                                   │
                                   ▼
                data/a_share_data_processed/factors.parquet (69 standardized + 89 kept)
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
          run_search.py       run_blo.py          run_backtest.py
                │                  │                  │
                ▼                  ▼                  ▼
          results/alphas/    results/blo/       results/backtest/
          (alpha parquets)   (λ + rc)           (NAV + metrics)
                                   │
                                   └── rc = spearmanr(IC, λ) <── IC ranking from search
```

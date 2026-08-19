<h1 align="center">Data Pipeline</h1>

This document describes the data layer: market data, Alpha158 factors, label
construction, and factor preprocessing.

## Overview

All market data is read from parquet files under `data/a_share_data/` at
runtime — no database connection is needed for training/backtesting. The
download scripts (`scripts/build_*.py`) pull from DolphinDB once and cache
locally.

```
DolphinDB ──build_backtest_data.py──> data/a_share_data/ (raw parquet)
                                        │
              build_factors.py ─────────┤  (compute Alpha158)
                                        │
              cache_factors_processed.py ──> data/a_share_data_processed/ (preprocessed factors)
                                        │
              run_search / run_blo ──────┴──> training
```

## Data Layout

`data/a_share_data/` (raw, downloaded from DDB):

| File | Shape | Description |
|---|---|---|
| `adjclose.parquet` | T × N (wide) | Post-adjusted close price (NAV mark + return calc) |
| `vwap_adj.parquet` | T × N | Post-adjusted VWAP = raw VWAP × adjustment factor (trade price + label) |
| `limit_up.parquet` | T × N | Limit-up mask (cannot buy) |
| `limit_down.parquet` | T × N | Limit-down mask (cannot sell) |
| `tradable_mask.parquet` | T × N | Tradable mask (excludes suspended stocks) |
| `member_mask_csi300.parquet` | T × N | CSI300 daily constituent mask (PIT) |
| `member_mask_csi500.parquet` | T × N | CSI500 daily constituent mask (PIT) |
| `member_mask_csi1000.parquet` | T × N | CSI1000 daily constituent mask (PIT) |
| `index_label.parquet` | T × N | 0/1/2/3 index label (synthesized from the three masks) |
| `index_close.parquet` | T × 3 | CSI300/500/1000 daily close (benchmark) |
| `factors.parquet` | long (date, code) × 158 | Alpha158 factors |

`data/a_share_data_processed/`:

| File | Description |
|---|---|
| `factors.parquet` | Preprocessed factors (69 standardized + 89 kept), NaN thresholds from train period |

## Date Ranges

Defined in `src/config.py`:

| Constant | Value | Purpose |
|---|---|---|
| `DATA_START` | 2019-06-01 | Data build start (half-year before train, for rolling-60 warmup) |
| `DATA_END` | 2026-06-30 | Data build end |
| `TRAIN_START` / `TRAIN_END` | 2020-01-02 / 2023-12-31 | Training |
| `VALID_START` / `VALID_END` | 2024-01-02 / 2024-12-31 | Validation (best-epoch selection) |
| `TEST_START` / `TEST_END` | 2025-01-02 / 2026-06-30 | Test (final eval + backtest) |

Label quantile thresholds and factor de-extreme thresholds are computed on
`TRAIN_START..TRAIN_END` only — look-ahead safe for valid/test.

## Alpha158 Factors

`factors_calculate/alpha158.py` defines the 158 factor names (consistent with
Qlib):

- **9 candlestick pattern** features (KMID, KLEN, ...).
- **4 same-day price ratios** (OPEN0, HIGH0, LOW0, VWAP0).
- **29 rolling templates × 5 windows** (5,10,20,30,60) = 145 features (ROC, MA,
  STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD,
  CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP,
  VSUMN, VSUMD).

Computation (`factors_calculate/alpha158_db.py`) pulls raw OHLCV from DDB and
computes factors in yearly batches with rolling-60 warmup, concatenated and
de-duplicated.

## Factor Preprocessing

`src/data_utils.py:load_factors_tanhw` applies **intrinsic-class preprocessing**
(not a blanket z-score):

- **69 heavy-tailed factors** (price ratios, MA, ROC, etc.): cross-sectional
  `tanh(median / (IQR·2))` — compresses outliers to `[-1, 1]`. Outliers beyond train-set `q001/q999` are set to NaN.
- **89 bounded factors** (candlestick ratios, statistics): kept as-is (already
  bounded in `[-1, 1]` or stable scale).

This avoids the distortion that a blanket z-score introduces on heavy-tailed
factors. The preprocessed cache is built once by
`scripts/cache_factors_processed.py`.

## Labels

Post-adjusted VWAP return (`src/data_utils.py:make_label`):

```
r_h[t] = vwap_adj[t+h+1] / vwap_adj[t+1] - 1
```

The `+1` offset aligns the entry at next-day VWAP (signal at `t` close →
execute at `t+1` VWAP). Labels are:

1. **Quantile-trimmed**: samples outside `[q0.5, q99.5]` (per-horizon, train-set
   thresholds) are set to NaN — prevents extremes from polluting the
   cross-sectional z-score.
2. **Cross-sectional z-score + clip ±3**: per-day, per-horizon.

For BLO warmup, a **diff-z label** is used: `d_1=r_1, d_2=r_2-r_1, ...,
d_10=r_10-r_9`, z-scored + clipped + equal-weighted mean → single scalar label.

## Data Availability

This repo does **not** redistribute market data due to licensing. The `data/`
directory is gitignored.

The build scripts (`scripts/build_backtest_data.py`, `scripts/build_factors.py`)
pull from an internal DolphinDB instance and are provided **for reference only**:
the DDB connection itself is not accessible. To run the pipeline you have two
options:

1. **Construct the parquet files yourself** following the layout above and place
   them under `data/a_share_data/`. The training/backtest code only reads
   parquet — any source that produces the same schema works.
2. **Adapt the build scripts** to your own data source (e.g. another market-data
   vendor or your own database). The `data_fetch/` module isolates all data
   access; replace its internals and the rest of the pipeline is unchanged.

## Build Scripts (reference)

```bash
# Market data + index constituents (needs DDB)
python scripts/build_backtest_data.py            # download missing (skip existing)
python scripts/build_backtest_data.py --force    # force re-download

# Alpha158 factors (needs DDB)
python scripts/build_factors.py

# Preprocess factors (one-time)
python scripts/cache_factors_processed.py
```

`build_backtest_data.py` does not overwrite existing files by default — only
missing files are downloaded. Use `--force` to re-download everything.

## DolphinDB Connection (reference)

`data_fetch/connection.py` holds the DDB connection config (internal, not
accessible). `data_fetch/` provides:

- `marketdata_db.py`: EOD prices, VWAP, limit status, tradable mask, index close.
- `index_members_db.py`: index constituent masks (PIT).
- `calendar_db.py`: trading calendar.

These modules document the data access patterns used to build the parquet
files; adapt them to your own data source if needed.

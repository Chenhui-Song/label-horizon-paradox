<h1 align="center">The Label Horizon Paradox — Open-Source Demo</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://openreview.net/forum?id=G43CIfmmxh"><img src="https://img.shields.io/badge/ICML-2026-red.svg" alt="ICML 2026"></a>
</p>

An open-source demo of the bi-level optimization (BLO) method from **"The
Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting"**
([ICML 2026](https://openreview.net/forum?id=G43CIfmmxh)).

The paper uncovers a counter-intuitive phenomenon: **the optimal training label
horizon often deviates from the inference target**. Training on the exact target
horizon is not necessarily best; an intermediate horizon can provide a better
signal-to-noise trade-off. This repo provides a self-contained demo that
operationalizes that insight with a BLO framework that **auto-learns the
label-horizon weights** within a single training run, built entirely on the
public **Alpha158** factor set and A-share market data.

> This is a standalone demo, not the paper's experimental codebase. The paper's
> original experimental features are not public. This demo builds the core method
> (BLO + brute-force baseline + ladder backtest) on public daily-frequency
> A-share data and the open-source Alpha158 factor set. Some details differ from
> the paper's description (see [docs/blo.md](docs/blo.md#differences-from-the-paper)).

---

## Task

We forecast the cross-sectional **10-day VWAP return** of A-shares — the return
from the next-day VWAP (`t+1`) to the VWAP 10 days later (`t+11`), i.e. the
target horizon `Δ = 10`. This matches a 10-day rebalancing cadence.

Instead of training on this target directly, we treat the label horizon as a
learnable choice: the model can be supervised by the `h`-day VWAP return for any
`h = 1..10`. The question is which `h` gives the best out-of-sample performance
on the fixed target `Δ = 10`. BLO learns the horizon weights `λ` automatically;
the brute-force search trains one model per `h` to establish the IC ranking for
comparison.

---

## TL;DR

- **Universe**: all A-shares; **features**: Alpha158 (158 factors from
  Microsoft's [Qlib](https://github.com/microsoft/qlib)).
- **Labels**: adjusted VWAP returns for `h = 1..10` (10 candidate horizons).
- **Inference target**: `Δ = 10` (10-day VWAP return, aligned with a 10-day
  rebalancing cadence).
- **Model**: AlphaMLP (`158 → 256 → 128 → 1`).
- **Method**: bi-level optimization learns a softmax weight vector `λ ∈ R^10`
  over the 10 candidate horizons; the student is trained on the λ-weighted
  composite label and evaluated on the fixed target `Δ = 10`. Unlike the
  brute-force search (one model per horizon), BLO trains a single student, so
  its cost advantage grows with the number of candidate horizons.
- **Result**: the learned `λ` recovers the brute-force IC ranking (Spearman
  `rc ≈ 0.94`), confirming that BLO autonomously identifies the near-optimal
  supervision horizon without manual search.

---

## Pipeline

```
Alpha158 factors ────┐
                     ├──> build_tensors ──> BLO trainer ──> learned λ
VWAP labels (h=1..10)┘                       (warmup + bi-level)
                                                    │
                                                    ▼
                    brute-force search ──> IC ranking ──> rc(λ, IC)
                                                    │
                                                    ▼
                    saved alphas ──> LadderBacktest ──> Sharpe / Ann / MDD
```

Three entry scripts, each decoupled:

| Script | Purpose | Output |
|---|---|---|
| `scripts/run_search.py` | Brute-force: train one MLP per `h`, produce the IC ranking | `results/alphas/h{h}/alpha_*.parquet` |
| `scripts/run_blo.py` | BLO: auto-learn horizon weights `λ` | `results/blo/blo_results.csv` |
| `scripts/run_backtest.py` | Ladder backtest on saved alphas (top-N within an index universe, or 3-index) | `results/backtest/*.csv` |

---

## Installation

```bash
git clone https://github.com/Chenhui-Song/label-horizon-paradox.git
cd label-horizon-paradox
pip install -r requirements.txt   # torch, pandas, numpy, pyarrow, dolphindb
```

**Data**: this repo does **not** redistribute market data due to licensing.
The build scripts (`scripts/build_backtest_data.py`, `scripts/build_factors.py`)
pull from an internal DolphinDB instance and are provided for reference only —
the DDB connection itself is not accessible. To run the pipeline, either
construct the parquet files yourself and place them under `data/a_share_data/`
(see the layout in [docs/data.md](docs/data.md#data-layout)), or adapt the
build scripts to your own data source. The `data/` directory is gitignored.

---

## Quick Start

```bash
# 1. Build market data + Alpha158 factors (one-time, ~hours; needs DDB)
python scripts/build_backtest_data.py
python scripts/build_factors.py
python scripts/cache_factors_processed.py

# 2. Brute-force horizon search (baseline IC ranking, ~2h on one GPU)
python scripts/run_search.py

# 3. Bi-level optimization (auto-learn λ, ~3 min per seed; one run is enough,
#    the default 10-seed sweep is only for stability verification)
python scripts/run_blo.py

# 4. Backtest saved alphas
python scripts/run_backtest.py --mode agg --pool topn --topn 100 --universe csi300,csi500,csi1000
```

---

## Key Results

### 1. Brute-force search — the Label Horizon Paradox

Train one AlphaMLP per horizon `h` (15 seeds, aggregated), evaluate test IC on
the fixed target `h=10`. Test period 2025-01 ~ 2026-06.

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

**IC peaks at `h=3` (0.0860), not at the inference target `h=10` (0.0825)** —
the Label Horizon Paradox. The peak is broad (`h=3..h=8` ≈ 0.084–0.086), but
the short horizon `h=3` is the single best. The broad peak is expected: this
demo trains on all A-shares (~5000+ stocks), and per the paper's Eq. 29
(`K = dσ²/N`) the larger sample size `N` compresses the horizon gap. See
[docs/search.md](docs/search.md#why-the-peak-is-broad-sample-size-compresses-the-horizon-gap)
for the derivation. This IC ranking is the evaluation target for BLO.

### 2. BLO — auto-learning the horizon weights

BLO learns a softmax weight vector `λ` over the 10 horizons — the student is
trained on the `λ`-weighted composite label and evaluated on the fixed target
`h=10`. To check stability, we run it 10 independent times. Each run is scored
by `rc` — the rank correlation between the learned `λ` and the brute-force IC
ranking (higher is better; 1.0 means BLO perfectly recovers the ranking without
running the brute-force sweep).

| run | `rc` | argmax `λ` |
|---|---|---|
| 1 | 0.952 | h3 |
| 2 | 0.915 | h3 |
| 3 | 0.915 | h3 |
| 4 | 0.952 | h3 |
| 5 | 0.952 | h3 |
| 6 | 0.891 | h5 |
| 7 | 0.976 | h4 |
| 8 | 0.964 | h3 |
| 9 | 0.927 | h3 |
| 10 | 0.927 | h3 |

**8/10 runs lock onto h=3** (the brute-force IC peak); the other two land on
the adjacent h=4/h=5, with h=3 as the second-highest weight in both. BLO
identifies the right supervision horizon without the brute-force sweep. See
[docs/blo.md](docs/blo.md) for method details and differences from the paper.

### 3. Backtest

Ladder (staggered sub-portfolio) backtest: total capital is split into 10
sub-accounts that rebalance on staggered days (each every 10 days), so each
sub-account stays low-frequency while overall turnover is dispersed daily.
10-day rebalance, VWAP execution, A-share costs. Test period 2025-01 ~ 2026-06.
The stock universe is restricted to the **CSI 300 / CSI 500 / CSI 1000**
constituents (the CSI1800 union) — the most representative and liquid A-share
names — with two selection modes: top-N across the union (default 100), or
top-N from each index separately (e.g. 20:30:50).
This is a **simplified backtest for signal validation only** — an LLM-built
demo framework, not an industrial-grade backtester. It omits many real-world
trading details: no slippage / market impact, buy and sell share one full-day
VWAP price, no volume-participation cap, equal-weighting with no risk-model
constraints, and single-day execution. The goal is to check whether the
paradox shows up in a realistic backtest, not to produce a production
strategy. The relative comparison across horizons is what matters. See
[docs/backtest.md](docs/backtest.md#simplifications--limitations) for the
full list of simplifications.

**CSI1800 top-100** (`--pool topn --topn 100 --universe csi300,csi500,csi1000`, 100 stocks from the CSI300+CSI500+CSI1000 union):

| h | Sharpe | Ann | MDD | Calmar |
|---|---|---|---|---|
| 1 | 1.690 | 30.48% | 11.30% | 2.70 |
| 2 | 1.696 | 30.79% | 10.82% | 2.85 |
| 3 | **1.811** | **33.45%** | 11.65% | **2.87** |
| 4 | 1.739 | 31.56% | 12.16% | 2.60 |
| 5 | 1.755 | 31.42% | 12.60% | 2.49 |
| 6 | 1.749 | 30.53% | 13.49% | 2.26 |
| 7 | 1.762 | 30.37% | 13.36% | 2.27 |
| 8 | 1.811 | 31.20% | 13.69% | 2.28 |
| 9 | 1.712 | 29.41% | 13.12% | 2.24 |
| 10 | 1.700 | 29.19% | 13.31% | 2.19 |

> Sharpe h3/h8 tied (1.811); Ann, Calmar peak at h3. (Calmar = Ann/MDD.)

**Three-index 20:30:50** (CSI300/500/1000 top 20/30/50, 100 stocks):

| h | Sharpe | Ann | MDD | Calmar |
|---|---|---|---|---|
| 1 | 1.700 | 30.42% | 11.24% | 2.71 |
| 2 | 1.700 | 30.65% | 10.71% | 2.86 |
| **3** | **1.852** | **34.00%** | 10.96% | **3.10** |
| 4 | 1.774 | 31.95% | 11.65% | 2.74 |
| 5 | 1.773 | 31.54% | 11.95% | 2.64 |
| 6 | 1.754 | 30.60% | 12.99% | 2.36 |
| 7 | 1.762 | 30.31% | 12.96% | 2.34 |
| 8 | 1.812 | 31.17% | 13.15% | 2.37 |
| 9 | 1.666 | 28.68% | 12.99% | 2.21 |
| 10 | 1.690 | 29.14% | 12.82% | 2.27 |

> All return/risk metrics (Sharpe, Ann, Calmar, Ret/TO) peak at h=3 — a clean
> match to the IC peak (h=3, 0.0860).

Across the two pools and the IC/Sharpe/Ann/Calmar/Ret-TO metrics, **h=3 is
broadly the best overall** — the only tie is Sharpe in the CSI1800 top-100 pool
(h3 = h8 = 1.811), but h=3 wins on Ann, Calmar, and Ret/TO there too. The Label
Horizon Paradox carries through from IC to backtest.

---

## Documentation

- [docs/data.md](docs/data.md) — Data pipeline: Alpha158 factors, VWAP labels, factor preprocessing, data download.
- [docs/search.md](docs/search.md) — Brute-force horizon search and the IC ranking baseline.
- [docs/blo.md](docs/blo.md) — Bi-level optimization: method, writeback, and differences from the paper.
- [docs/backtest.md](docs/backtest.md) — Ladder backtest engine, optimizers, costs, index benchmarks.
- [docs/architecture.md](docs/architecture.md) — Code structure and module responsibilities.

---

## Repository Layout

```
label-horizon-paradox/
├── src/                  # core: config, model, data_utils, blo_trainer
├── scripts/              # entry points: build_*.py, run_search.py, run_blo.py, run_backtest.py
├── engine/               # backtest engine (ladder, optimizer, account, costs, execution)
│   └── base/             #   account, constraints, costs, timeline, execution
├── factors_calculate/    # Alpha158 factor names + DDB computation
├── data_fetch/           # DolphinDB market data + index constituents
├── performance/          # return/risk + IC/RankIC metrics
├── data/                 # parquet market data
│   ├── a_share_data/          # raw (prices, masks, factors)
│   └── a_share_data_processed/# preprocessed factors (69 standardized + 89 kept)
├── results/              # alphas/, blo/, backtest/, models/
└── docs/                 # documentation
```

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{
song2026the,
  title={The Label Horizon Paradox: Rethinking Supervision Targets in Financial Forecasting},
  author={Chen-Hui Song and Shuoling Liu and Liyuan Chen},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=G43CIfmmxh}
}
```

---

## Disclaimer

This project is for **academic research purposes only** and does **not
constitute investment advice**. The backtest is a simplified signal-validation
framework, not a production simulator; real trading involves risks, costs, and
constraints not modeled here (see [docs/backtest.md](docs/backtest.md#simplifications--limitations)
for details). The authors are not liable for any financial decisions made
based on this code or its results.

---

## License

This project is released under the [MIT License](LICENSE).


"""Decoupled backtest: run ladder staggered sub-portfolio backtests from saved alpha parquets.

Fully decoupled from training: only reads alpha parquets under results/alphas/h{h}/
plus data market data; does not touch models/factors.

Two alpha modes (--mode):
  - agg (default): read alpha_agg.parquet for each h, backtest once
  - seed: read alpha_seed{si}.parquet (15 files) for each h, backtest each then average metrics

Two stock pools (--pool, do not run both at once; specify to switch):
  - topn: select top N equal-weight from an index universe (default N=100).
    --universe csi300,csi500,csi1000 restricts candidates to the CSI1800 union;
    omit for the full universe.
  - 3index: select top N equal-weight from each of the three indices.
    --n-config A:B:C: per-index N as csi300:csi500:csi1000 (default 50:50:50;
      repeatable to compare multiple configs, e.g. --n-config 20:30:50 --n-config 30:50:80)
  3index runs also print CSI300/500/1000 buy-and-hold benchmarks over the test period.

Backtest config:
  - n_ladders=10, rebalance_freq=10 (rebalance every 10 days, 10 staggered sub-accounts)
  - DailyClose (t close signal -> t+1 execution)
  - VWAPExec (next-day VWAP execution)
  - CostModel (stamp duty 0.05% sell-only + commission 0.025 per 10k both sides)
  - Constraints(max_weight=None, participation_rate=None, residual='cash', rollover=True:
    limit-up-blocked buy weight rolls over to substitute names; limit-down/suspension frozen)

Outputs (results/backtest/):
  - backtest_{mode}_{pool}_{config}_h{h}.csv: NAV series (nav) for each h/config
  - backtest_{mode}_{pool}_summary.csv: summary (sharpe/annualized/drawdown/turnover)
  - log_backtest.txt

Usage:
  python scripts/run_backtest.py --mode agg --pool topn --topn 100 --universe csi300,csi500,csi1000  # CSI1800 top-100
  python scripts/run_backtest.py --mode agg --pool 3index            # agg + 50:50:50 per index
  python scripts/run_backtest.py --mode agg --pool 3index --n-config 20:30:50 --n-config 30:50:80  # compare
  python scripts/run_backtest.py --mode seed --pool topn --h 5       # seed mode single h
  python scripts/run_backtest.py --mode agg --pool topn --h 1 5 10   # specify multiple h
"""
from __future__ import annotations
import sys, warnings, time, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # project root directory
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from engine import LadderBacktest, TopNOptimizer, ThreeIndexTopNOptimizer
from engine.base.execution import VWAPExec
from engine.base.timeline import DailyClose
from engine.base.costs import CostModel
from engine.base.constraints import Constraints
from src.config import RAW_DATA_DIR

ALPHA_DIR = _ROOT / "results" / "alphas"
OUT_DIR = _ROOT / "results" / "backtest"
LOG = OUT_DIR / "log_backtest.txt"

N_LADDERS = 10
REBALANCE_FREQ = 10
N_SEED = 15
HORIZONS = list(range(1, 11))
_lines = []


def p(msg=""):
    print(msg, flush=True)
    _lines.append(str(msg))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w") as f:
        f.write("\n".join(_lines))


def make_optimizer(pool: str, topn: int, n_per_index, universe=None):
    """Build an optimizer based on the stock pool. Returns (optimizer, member_masks_or_None).

    n_per_index: int (same N per index) or dict {"csi300":..,"csi500":..,"csi1000":..}.
    universe: for topn pool, optional comma-separated index names (e.g. "csi300,csi500,csi1000")
        restricting candidates to the union of those constituents. None = full universe.
    """
    if pool == "topn":
        masks = None
        if universe:
            masks = []
            for name in universe.split(","):
                name = name.strip().lower()
                p = RAW_DATA_DIR / f"member_mask_{name}.parquet"
                if not p.exists():
                    raise FileNotFoundError(f"universe mask not found: {p}")
                masks.append(pd.read_parquet(p))
        return TopNOptimizer(n=topn, member_masks=masks), masks
    elif pool == "3index":
        masks = {
            "csi300": pd.read_parquet(RAW_DATA_DIR / "member_mask_csi300.parquet"),
            "csi500": pd.read_parquet(RAW_DATA_DIR / "member_mask_csi500.parquet"),
            "csi1000": pd.read_parquet(RAW_DATA_DIR / "member_mask_csi1000.parquet"),
        }
        return ThreeIndexTopNOptimizer(member_masks=masks, n_per_index=n_per_index), masks
    raise ValueError(f"unknown pool: {pool} (supported: topn/3index)")


def index_benchmarks():
    """CSI300/500/1000 buy-and-hold metrics over the test period."""
    from src.config import TEST_START, TEST_END
    from performance import metrics as M
    ic = pd.read_parquet(RAW_DATA_DIR / "index_close.parquet")
    s, e = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    out = {}
    for name, code in {"CSI300": "000300.SH", "CSI500": "000905.SH", "CSI1000": "000852.SH"}.items():
        close = ic[code].loc[s:e].dropna()
        ret = close.pct_change().dropna()
        out[name] = {"sharpe": M.sharpe(ret), "ann": M.annualized_return(ret), "mdd": M.max_drawdown(ret)}
    return out


def run_one(alpha: pd.DataFrame, optimizer, rollover: bool = True) -> dict:
    """Run one ladder backtest, return a metrics dict."""
    r = LadderBacktest(
        alpha_signals=alpha,
        optimizer=optimizer,
        n_ladders=N_LADDERS,
        rebalance_freq=REBALANCE_FREQ,
        timepoint=DailyClose(),
        execution=VWAPExec(),
        cost_model=CostModel(),
        constraints=Constraints(max_weight=None, participation_rate=None, residual="cash", rollover=rollover),
    ).run()
    return {
        "nav": r.nav,
        "sharpe": float(r.sharpe),
        "ann": float(r.annualized_return),
        "mdd": float(r.max_drawdown),
        "turnover": float(r.avg_turnover),
        "end_nav": float(r.nav.iloc[-1]) if len(r.nav) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Decoupled ladder backtest")
    ap.add_argument("--mode", choices=["agg", "seed"], default="agg",
                    help="alpha mode: agg=read aggregated alpha and backtest once, seed=read each seed and backtest then average")
    ap.add_argument("--pool", choices=["topn", "3index"], default="topn",
                    help="stock pool: topn=top N within an index universe, 3index=top N from each of the three indices")
    ap.add_argument("--topn", type=int, default=100, help="number of stocks for topn mode (default 100)")
    ap.add_argument("--n-config", action="append", default=None, metavar="A:B:C",
                    help="3index per-index N as csi300:csi500:csi1000 (default 50:50:50). Repeatable to compare multiple configs.")
    ap.add_argument("--h", type=int, nargs="+", default=None, help="specify h (default h1..10)")
    ap.add_argument("--universe", type=str, default=None,
                    help="topn pool only: comma-separated index constituents to restrict candidates to their union, e.g. 'csi300,csi500,csi1000' for the CSI1800 universe. Default = full universe.")
    ap.add_argument("--no-rollover", action="store_true",
                    help="disable rolling over limit-up-blocked buy weight to substitute names (on by default)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons = args.h if args.h else HORIZONS

    # Build the list of N configs to run for 3index (each is a dict or None).
    if args.pool == "3index":
        if args.n_config:
            n_configs = []
            for cfg in args.n_config:
                parts = cfg.split(":")
                if len(parts) != 3:
                    raise ValueError(f"--n-config expects csi300:csi500:csi1000, got {cfg}")
                n_configs.append({"name": cfg.replace(":", "_"),
                                  "n": {"csi300": int(parts[0]), "csi500": int(parts[1]), "csi1000": int(parts[2])}})
        else:
            n_configs = [{"name": "50_50_50", "n": {"csi300": 50, "csi500": 50, "csi1000": 50}}]
    else:
        uni_tag = f"_{'_'.join(u.strip().lower() for u in args.universe.split(','))}" if args.universe else ""
        n_configs = [{"name": f"topn{args.topn}{uni_tag}", "n": None}]

    idx_bm = index_benchmarks() if args.pool == "3index" else None

    p("=" * 90)
    p(f"Decoupled ladder backtest | mode={args.mode} pool={args.pool}")
    if args.pool == "topn":
        p(f"  topn={args.topn}")
    else:
        for c in n_configs:
            total = sum(c["n"].values()) if isinstance(c["n"], dict) else 3 * c["n"]
            p(f"  3index config {c['name']}: {c['n']} (total {total})")
    p(f"  n_ladders={N_LADDERS} rebalance_freq={REBALANCE_FREQ} DailyClose+VWAP, stamp duty 0.05% sell-only + commission 0.025% both sides, limit-up buy rolled over to substitutes, limit-down/suspension frozen")
    p(f"  horizons={horizons}")
    if idx_bm:
        p(f"  index benchmarks: " + " | ".join(f"{k} sharpe={v['sharpe']:.3f} ann={v['ann']:.3f}" for k, v in idx_bm.items()))
    p("=" * 90)

    rows = []
    t_all = time.time()
    for h in horizons:
        h_dir = ALPHA_DIR / f"h{h}"
        if not h_dir.exists():
            p(f"\n--- h{h}: alpha directory does not exist, skip ---")
            continue

        # Load alpha once per h (agg: single; seed: average metrics across seeds)
        if args.mode == "agg":
            path = h_dir / "alpha_agg.parquet"
            if not path.exists():
                p(f"\n--- h{h}: {path.name} does not exist, skip ---")
                continue
            alpha = pd.read_parquet(path)
            alphas = [(alpha, 1)]
        else:  # seed
            seed_alphas = []
            for si in range(N_SEED):
                path = h_dir / f"alpha_seed{si}.parquet"
                if path.exists():
                    seed_alphas.append(pd.read_parquet(path))
            if not seed_alphas:
                p(f"\n--- h{h}: no seed alpha, skip ---")
                continue
            alphas = [(a, 1) for a in seed_alphas]

        for cfg in n_configs:
            optimizer, _ = make_optimizer(args.pool, args.topn, cfg["n"], universe=args.universe)
            t1 = time.time()
            # run_one over each alpha in `alphas`, average metrics
            res_list = [run_one(a, optimizer, rollover=not args.no_rollover) for a, _ in alphas]
            avg = {
                "sharpe": float(np.mean([r["sharpe"] for r in res_list])),
                "ann": float(np.mean([r["ann"] for r in res_list])),
                "mdd": float(np.mean([r["mdd"] for r in res_list])),
                "turnover": float(np.mean([r["turnover"] for r in res_list])),
                "end_nav": float(np.mean([r["end_nav"] for r in res_list])),
            }
            nav0 = res_list[0]["nav"]
            tag = f"{args.mode}_{args.pool}_{cfg['name']}"
            p(f"--- h{h} [{tag}] sharpe={avg['sharpe']:.3f} ann={avg['ann']*100:.2f}% "
              f"mdd={avg['mdd']*100:.2f}% turnover={avg['turnover']*100:.2f}% (took {time.time()-t1:.0f}s)")
            pd.DataFrame({"date": nav0.index, "nav": nav0.values}).to_csv(
                OUT_DIR / f"backtest_{tag}_h{h}.csv", index=False)
            rows.append({"h": h, "mode": args.mode, "pool": args.pool, "config": cfg["name"], **avg,
                         "n_seed": len(alphas) if args.mode == "seed" else 1})

        pd.DataFrame(rows).to_csv(OUT_DIR / f"backtest_{args.mode}_{args.pool}_summary.csv", index=False)

    # Summary table
    p(f"\n{'='*90}")
    p(f"Summary: mode={args.mode} pool={args.pool}")
    if args.pool == "3index":
        for cfg in n_configs:
            p(f"\n--- config {cfg['name']} ---")
            p(f"{'h':>3} | {'sharpe':>7} | {'ann':>8} | {'mdd':>7} | {'turnover':>7}")
            p("-" * 50)
            for r in [x for x in rows if x["config"] == cfg["name"]]:
                p(f"{r['h']:>3} | {r['sharpe']:>7.3f} | {r['ann']*100:>7.2f}% | {r['mdd']*100:>6.2f}% | {r['turnover']*100:>6.2f}%")
        if idx_bm:
            p(f"\nIndex benchmarks (test period):")
            for k, v in idx_bm.items():
                p(f"  {k}: sharpe={v['sharpe']:.3f} ann={v['ann']*100:.2f}% mdd={v['mdd']*100:.2f}%")
    else:
        p(f"{'h':>3} | {'sharpe':>7} | {'ann':>8} | {'mdd':>7} | {'turnover':>7} | {'end_nav':>7}")
        p("-" * 90)
        for r in rows:
            p(f"{r['h']:>3} | {r['sharpe']:>7.3f} | {r['ann']*100:>7.2f}% | {r['mdd']*100:>6.2f}% | {r['turnover']*100:>6.2f}% | {r['end_nav']:>7.4f}")
    p(f"\nNAV saved to: backtest_{args.mode}_{args.pool}_<config>_h{{h}}.csv")
    p(f"Summary saved to: backtest_{args.mode}_{args.pool}_summary.csv")
    p(f"Total time {time.time()-t_all:.0f}s")
    p(f"\n*** Done ***")


if __name__ == "__main__":
    main()

"""Download backtest data to data/a_share_data/ (DATA_START~DATA_END).

Responsibilities: pull market data / index constituents from DolphinDB,
process them, and store locally as parquet. At runtime, backtest/training
reads directly from data/ without connecting to the database.

Date range and output path are read from src/config.py (DATA_START, DATA_END,
RAW_DATA_DIR). Data starts from 2019-06 (half-year before TRAIN_START=2020)
to provide rolling-window warmup for factor computation.

Outputs (data/a_share_data/):
  - adjclose.parquet:       post-adjusted close price (NAV mark price + return calc)
  - vwap_adj.parquet:       post-adjusted VWAP = raw VWAP x adjustment factor (trade price + label)
  - limit_up.parquet:       limit-up mask (cannot buy)
  - limit_down.parquet:     limit-down mask (cannot sell)
  - tradable_mask.parquet:  tradable mask (excludes suspended stocks)
  - member_mask_csi300/500/1000.parquet: daily constituent masks for the three indices (PIT)
  - index_label.parquet:    0/1/2/3 index label (synthesized from the three masks)
  - index_close.parquet:    CSI300/500/1000 daily close (benchmark for backtest comparison)

Does not overwrite existing files: each file is checked for existence before
download; existing files are skipped (unless --force).

Usage:
  python scripts/build_backtest_data.py              # download missing files (no overwrite)
  python scripts/build_backtest_data.py --force      # force re-download of everything
  python scripts/build_backtest_data.py --start 2025-01-02 --end 2025-01-10  # custom range
"""
from __future__ import annotations
import sys, warnings, time, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from data_fetch import marketdata_db as md
from data_fetch.index_members_db import get_index_member_mask
from data_fetch.connection import dc
from src.config import DATA_START, DATA_END, RAW_DATA_DIR

INDEX_CODES = {"member_mask_csi300": "000300.SH",
               "member_mask_csi500": "000905.SH",
               "member_mask_csi1000": "000852.SH"}


def _save(name: str, df, out_dir: Path, force: bool) -> bool:
    """Save parquet; skip if it already exists and force is False. Returns whether it was actually written."""
    path = out_dir / f"{name}.parquet"
    if path.exists() and not force:
        print(f"  [skip] {name}.parquet already exists ({path.stat().st_size/1e6:.1f}M)", flush=True)
        return False
    df.to_parquet(path)
    print(f"  [saved] {name}.parquet shape={df.shape} ({path.stat().st_size/1e6:.1f}M)", flush=True)
    return True


def build_prices(out_dir: Path, start: str, end: str, force: bool):
    """Download price/status data: adjclose, vwap_adj, limit_up, limit_down, tradable_mask."""
    print(f"\n[1] Price/status data {start}~{end}...", flush=True)
    t0 = time.time()

    # Post-adjusted close price
    adjclose = md.get_eod_prices(start, end, field="S_DQ_ADJCLOSE")
    _save("adjclose", adjclose, out_dir, force)

    # Post-adjusted VWAP = raw VWAP x adjustment factor
    vwap_raw = md.get_minute_vwap(start, end, window=None)
    adjfactor = md.get_eod_prices(start, end, field="S_DQ_ADJFACTOR")
    common_s = vwap_raw.columns.intersection(adjfactor.columns)
    common_d = vwap_raw.index.intersection(adjfactor.index)
    vwap_adj = vwap_raw.loc[common_d, common_s].multiply(adjfactor.loc[common_d, common_s])
    _save("vwap_adj", vwap_adj, out_dir, force)

    # Limit up / limit down
    limit_up, limit_down = md.get_limit_status(start, end)
    _save("limit_up", limit_up, out_dir, force)
    _save("limit_down", limit_down, out_dir, force)

    # Tradable mask
    tradable = md.get_tradable_mask(start, end)
    _save("tradable_mask", tradable, out_dir, force)

    # Index close prices (CSI300/500/1000, for benchmark comparison)
    index_codes = ["000300.SH", "000905.SH", "000852.SH"]
    index_close = md.get_index_close(start, end, codes=index_codes)
    _save("index_close", index_close, out_dir, force)

    print(f"  Price/status took {time.time()-t0:.0f}s", flush=True)


def build_members(out_dir: Path, start: str, end: str, force: bool):
    """Download constituent masks for the three indices + synthesize index_label."""
    print(f"\n[2] Index constituents {start}~{end}...", flush=True)
    t0 = time.time()
    masks = {}
    for fn, code in INDEX_CODES.items():
        m = get_index_member_mask(code, start, end)
        _save(fn, m, out_dir, force)
        masks[fn] = m
        print(f"  {fn}({code}): shape={m.shape}, took {time.time()-t0:.0f}s", flush=True)

    # index_label: 0/1/2/3 (csi300>csi500>csi1000, mutually exclusive priority)
    print("  Synthesizing index_label...", flush=True)
    all_cols = masks["member_mask_csi300"].columns
    for k in masks:
        all_cols = all_cols.union(masks[k].columns)
    all_idx = masks["member_mask_csi300"].index
    for k in masks:
        all_idx = all_idx.union(masks[k].index)
    m300 = masks["member_mask_csi300"].reindex(index=all_idx, columns=all_cols, fill_value=False)
    m500 = masks["member_mask_csi500"].reindex(index=all_idx, columns=all_cols, fill_value=False)
    m1000 = masks["member_mask_csi1000"].reindex(index=all_idx, columns=all_cols, fill_value=False)
    label = pd.DataFrame(0, index=all_idx, columns=all_cols, dtype=np.int8)
    label[m1000] = 3
    label[m500] = 2
    label[m300] = 1
    _save("index_label", label, out_dir, force)
    print(f"  index_label: shape={label.shape}, distribution: {label.stack().value_counts().sort_index().to_dict()}", flush=True)
    print(f"  Constituents took {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download backtest data to data/")
    ap.add_argument("--force", action="store_true", help="force re-download (overwrite existing)")
    ap.add_argument("--start", default=DATA_START, help=f"start date (default {DATA_START})")
    ap.add_argument("--end", default=DATA_END, help=f"end date (default {DATA_END})")
    args = ap.parse_args()

    out_dir = RAW_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Download backtest data {args.start}~{args.end} -> {out_dir} ===", flush=True)
    print(f"  force={args.force} (existing files {'overwritten' if args.force else 'skipped'})", flush=True)
    t0 = time.time()
    build_prices(out_dir, args.start, args.end, args.force)
    build_members(out_dir, args.start, args.end, args.force)
    dc.close_ddb()

    # Validation
    print(f"\n=== Validation ===", flush=True)
    for f in sorted(out_dir.iterdir()):
        if f.suffix == ".parquet":
            print(f"  {f.name} ({f.stat().st_size/1e6:.1f}M)", flush=True)
    print(f"\n=== Done, total time {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()

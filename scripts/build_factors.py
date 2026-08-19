"""Compute all-A ALPHA158 factors (DATA_START~DATA_END), save to data/.

From raw DDB market data, use factors_calculate.alpha158_db.compute_alpha158 to
compute 158 factors. Computed in batches (about one year per batch, including
rolling60 warmup), concatenated, de-duplicated, then truncated to the target range.

Date range and output path are read from src/config.py (DATA_START, DATA_END,
RAW_DATA_DIR).

Outputs (data/a_share_data/):
  - factors.parquet:       full factors (long: date x wind_code x 158 factors)
  - tradable_mask.parquet: tradable mask (wide: date x wind_code)

Does not overwrite existing files: skipped if the file already exists (unless --force).

Usage:
  python scripts/build_factors.py              # compute full (skip if exists)
  python scripts/build_factors.py --force      # force recompute and overwrite
  python scripts/build_factors.py --start 2025-01-02 --end 2025-01-10  # custom range
"""
from __future__ import annotations
import sys, warnings, time, argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # project root directory
_SRC = _ROOT / "src"
for _p in (_HERE, _ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from src.config import DATA_START, DATA_END, RAW_DATA_DIR
from factors_calculate.alpha158_db import compute_alpha158, ALPHA158_NAMES
from data_fetch.marketdata_db import get_tradable_mask
from data_fetch.connection import dc

# Batches: about one year per batch, pushed forward for warmup (overlaps with
# the previous batch for de-dup). Each batch starts 6 months before its target
# year to provide rolling-60 warmup. Save truncated to DATA_START..DATA_END.
BATCHES = [
    ("2019-01-01", "2020-06-30"),  # 2019-06 warmup + early 2020
    ("2019-06-01", "2020-12-31"),  # 2020
    ("2020-06-01", "2021-12-31"),  # 2021
    ("2021-06-01", "2022-12-31"),  # 2022
    ("2022-06-01", "2023-12-31"),  # 2023
    ("2023-06-01", "2024-12-31"),  # 2024
    ("2024-06-01", "2025-12-31"),  # 2025
    ("2025-06-01", "2026-06-30"),  # 2026
]


def _save(name: str, df, out_dir: Path, force: bool) -> bool:
    """Save parquet; skip if it already exists and force is False."""
    path = out_dir / f"{name}.parquet"
    if path.exists() and not force:
        print(f"  [skip] {name}.parquet already exists ({path.stat().st_size/1e6:.1f}M)", flush=True)
        return False
    df.to_parquet(path)
    print(f"  [saved] {name}.parquet shape={df.shape} ({path.stat().st_size/1e6:.1f}M)", flush=True)
    return True


def build(start: str, end: str, out_dir: Path, force: bool, batches=None):
    """Compute factors in batches + concatenate + truncate + save."""
    print(f"\n=== Build all-A factors {start}~{end} (each batch has 6-month warmup) ===", flush=True)
    print(f"  ALPHA158 factor count: {len(ALPHA158_NAMES)}", flush=True)
    t0 = time.time()
    bs = batches if batches is not None else BATCHES
    parts = []
    for i, (s, e) in enumerate(bs):
        print(f"[{i+1}/{len(bs)}] Computing factors {s}~{e}...", flush=True)
        t1 = time.time()
        long_df = compute_alpha158(s, e, index_code=None, codes=None, as_frame="long")
        print(f"    done: {long_df.shape}, took {time.time()-t1:.0f}s", flush=True)
        parts.append(long_df)
    print(f"Concatenating {len(parts)} batches...", flush=True)
    long_df = pd.concat(parts)
    long_df = long_df[~long_df.index.duplicated(keep="first")]  # de-dup (warmup overlap)
    print(f"  After concat + de-dup: {long_df.shape}, took {time.time()-t0:.0f}s", flush=True)

    # Truncate to the save range
    dates = long_df.index.get_level_values("date")
    long_df = long_df[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    print(f"  Truncated to {start}~{end}: {long_df.shape}", flush=True)
    per_day = long_df.groupby(level="date").size()
    print(f"  Stocks per day: min/median/max = {per_day.min()}/{int(per_day.median())}/{per_day.max()}", flush=True)

    # Save factors
    _save("factors", long_df, out_dir, force)

    # Get tradable mask
    print(f"\nGetting tradable mask {start}~{end}...", flush=True)
    tradable = get_tradable_mask(start, end)
    factor_dates = long_df.index.get_level_values("date").unique()
    tradable = tradable.loc[tradable.index.intersection(factor_dates)]
    _save("tradable_mask", tradable, out_dir, force)

    print(f"\n=== Done, total time {time.time()-t0:.0f}s ===", flush=True)
    print(f"Outputs saved to: {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Compute all-A ALPHA158 factors")
    ap.add_argument("--force", action="store_true", help="force recompute (overwrite existing)")
    ap.add_argument("--start", default=DATA_START, help=f"start date (default {DATA_START})")
    ap.add_argument("--end", default=DATA_END, help=f"end date (default {DATA_END})")
    args = ap.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"force={args.force} (existing files {'overwritten' if args.force else 'skipped'})", flush=True)
    build(args.start, args.end, RAW_DATA_DIR, args.force)
    dc.close_ddb()


if __name__ == "__main__":
    main()

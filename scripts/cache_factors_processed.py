"""Cache factor preprocessing results (classified by semantics): 69 tanh + 89 kept.

Classification basis (factor semantics):
  - tanh class (69): price ratios / heavy-tailed, need cross-sectional comparison
    -> train-set q001/q999 out-of-bounds set to NaN + cross-sectional median/IQR*2 + tanh
  - kept class (89): naturally bounded ratios / statistics / correlation coefficients,
    no cross-sectional comparison needed -> leave raw values, do not remove extremes
  (KMID2/KSFT2/KUP2/KLOW2 are candlestick ratios in [-1,1], moved from tanh to kept;
   KLOW/LOW0/MIN are price ratios and stay in tanh)

NaN thresholds are computed over the training period (TRAIN_START..TRAIN_END,
look-ahead safe).
Output to a_share_data_processed/factors.parquet; original data is left untouched.
"""
from __future__ import annotations
import sys, warnings, time
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

from src.config import RAW_DATA_DIR, PROCESSED_DIR, TRAIN_START, TRAIN_END
from factors_calculate.alpha158 import ALPHA158_NAMES

# Classification after semantic correction: 73 tanh - 4 (candlestick ratios) = 69 tanh
TANH_FACTOR = {
    'KMID','KLEN','KSFT','KUP','KLOW',          # candlestick shape (price ratios, divided by o)
    'OPEN0','HIGH0','LOW0','VWAP0',              # same-day price ratios (divided by c)
    'ROC5','ROC10','ROC20','ROC30','ROC60',
    'MA5','MA10','MA20','MA30','MA60',
    'STD5','STD10','STD20','STD30','STD60',
    'BETA5','BETA10','BETA20','BETA30','BETA60',
    'RESI5','RESI10','RESI20','RESI30','RESI60',
    'MAX5','MAX10','MAX20','MAX30','MAX60',
    'MIN5','MIN10','MIN20','MIN30','MIN60',
    'QTLU5','QTLU10','QTLU20','QTLU30','QTLU60',
    'QTLD5','QTLD10','QTLD20','QTLD30','QTLD60',
    'VMA5','VMA10','VMA20','VMA30','VMA60',
    'VSTD5','VSTD10','VSTD20','VSTD30','VSTD60',
    'WVMA5','WVMA10','WVMA20','WVMA30','WVMA60',
}
# KMID2/KSFT2/KUP2/KLOW2 moved to the kept class (candlestick ratios, divided by h-l, bounded in [-1,1])

SRC_DIR = RAW_DATA_DIR
DST_DIR = PROCESSED_DIR
DST_DIR.mkdir(parents=True, exist_ok=True)

# NaN threshold computation window: TRAIN_START..TRAIN_END (2020-2023).
# Uses the training period only (look-ahead safe; valid/test use this threshold).
# Avoids the 2019 warmup period whose factors may contain rolling-window NaNs
# that would pollute the threshold estimate.
THR_START = TRAIN_START
THR_END = TRAIN_END


def main():
    t0 = time.time()
    print(f"=== Cache factor preprocessing (classified by semantics: 69 tanh + 89 kept) ===")
    print(f"  NaN threshold window: {THR_START}~{THR_END} (training period, look-ahead safe; valid/test use this threshold)")
    print(f"  tanh class: {len(TANH_FACTOR)} (price ratios / heavy-tailed, q001/q999 out-of-bounds NaN + cross-sectional median/IQR*2 + tanh)")
    print(f"  kept class: {158-len(TANH_FACTOR)} (bounded ratios / statistics, leave raw values, do not remove extremes)")
    print(f"  output: {DST_DIR}/factors.parquet")

    long_df = pd.read_parquet(SRC_DIR / "factors.parquet")
    factor_cols = ALPHA158_NAMES
    n_tanh = len([c for c in factor_cols if c in TANH_FACTOR])
    n_keep = len(factor_cols) - n_tanh
    print(f"\n[1] Raw data: shape={long_df.shape}, tanh={n_tanh} kept={n_keep}")

    # ---- 1) tanh class: TRAIN_START..TRAIN_END q001/q999 out-of-bounds set to NaN ----
    t1 = time.time()
    thr_s, thr_e = pd.Timestamp(THR_START), pd.Timestamp(THR_END)
    thr_idx = (long_df.index.get_level_values("date") >= thr_s) & \
              (long_df.index.get_level_values("date") <= thr_e)
    lo = {}; hi = {}
    for c in factor_cols:
        if c not in TANH_FACTOR:
            continue  # kept class does not compute thresholds, does not remove extremes
        s = long_df.loc[thr_idx, c].replace([np.inf, -np.inf], np.nan)
        q = s.quantile([0.001, 0.999])
        lo[c] = float(q.loc[0.001]); hi[c] = float(q.loc[0.999])
    n_drop = 0
    for c in factor_cols:
        if c not in TANH_FACTOR:
            continue
        col = long_df[c].replace([np.inf, -np.inf], np.nan)
        mask = (col < lo[c]) | (col > hi[c])
        n_drop += int(mask.sum())
        long_df[c] = col.mask(mask)
    print(f"[2] tanh class out-of-bounds NaN (train-set q001/q999 threshold, applied to full period): {n_drop} points "
          f"({100*n_drop/(len(long_df)*n_tanh):.4f}%) {time.time()-t1:.1f}s")

    # ---- 2) tanh class: cross-sectional median/IQR*2 + tanh ----
    t2 = time.time()
    for c in factor_cols:
        if c not in TANH_FACTOR:
            continue
        col = long_df[c]
        g = col.groupby(level="date", group_keys=False)
        med = g.transform("median")
        q25 = g.transform(lambda x: x.quantile(0.25))
        q75 = g.transform(lambda x: x.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        z = (col - med) / (iqr * 2.0)
        z = z.replace([np.inf, -np.inf], np.nan)
        long_df[c] = np.tanh(z)
    print(f"[3] tanh class cross-sectional tanh (IQR*2): {time.time()-t2:.1f}s")

    # ---- 3) kept class: leave raw values (only clean inf to NaN) ----
    t3 = time.time()
    for c in factor_cols:
        if c in TANH_FACTOR:
            continue
        col = long_df[c]
        long_df[c] = col.replace([np.inf, -np.inf], np.nan)
    print(f"[4] kept class raw values kept (only inf->NaN): {time.time()-t3:.1f}s")

    # ---- 4) save ----
    t4 = time.time()
    out = DST_DIR / "factors.parquet"
    long_df.to_parquet(out)
    print(f"[5] saved: {out} ({time.time()-t4:.1f}s)")

    # Validation
    chk = pd.read_parquet(out)
    print(f"\nValidation: shape={chk.shape}")
    tanh_c = [c for c in factor_cols if c in TANH_FACTOR]
    keep_c = [c for c in factor_cols if c not in TANH_FACTOR]
    print(f"  tanh class {len(tanh_c)} range: [{chk[tanh_c].min().min():.3f}, {chk[tanh_c].max().max():.3f}] (should be ~[-1,1])")
    print(f"  kept class {len(keep_c)} range: [{chk[keep_c].min().min():.3f}, {chk[keep_c].max().max():.3f}] (raw values)")
    print(f"\nTotal time {time.time()-t0:.1f}s *** Done ***")


if __name__ == "__main__":
    main()

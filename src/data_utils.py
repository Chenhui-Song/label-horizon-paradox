"""Data pipeline: factor preprocessing, label construction, universe, prediction.

Provides the shared data utilities used by both the brute-force search
(``run_search.py``) and the bi-level optimization (``run_blo.py``):

  - ``load_factors_tanhw``: read the preprocessed factor cache (69 tanh + 89 keep).
  - ``load_label_data``: read post-adjusted VWAP.
  - ``make_label``: build an h-horizon label (post-adjusted VWAP return, with
    optional quantile trimming).
  - ``compute_label_quantile``: training-set label quantile thresholds.
  - ``build_xy_clip``: build (X, y, date_ids) with cross-sectional z-score + clip.
  - ``get_tradable`` / ``get_train_union_tradable``: tradable masks.
  - ``predict_alpha_pit``: point-in-time alpha prediction.
  - ``metrics_for_alpha``: daily IC / RankIC / ICIR / RankICIR.

All market data is read directly from ``data/a_share_data/`` (post-adjusted
parquet files); no database connection is needed at runtime.
"""
from __future__ import annotations

import sys
import warnings
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
import torch

from src.config import DEVICE, RAW_DATA_DIR, PROCESSED_DIR, TRAIN_START, TRAIN_END
from factors_calculate.alpha158 import ALPHA158_NAMES


# ---- Factor intrinsic classification (69 tanh + 89 keep) ----
# 69 heavy-tailed factors (price ratios / MA / ROC etc.): cross-sectional
#   tanh(median / IQR*2) compressed to [-1, 1].
# 89 bounded factors (candlestick ratios / statistics etc.): kept as-is.
# KMID2/KSFT2/KUP2/KLOW2 are candlestick ratios (divided by h-l, bounded in
# [-1, 1]) and are moved to the keep class.
TANH_FACTOR = {
    'KMID', 'KLEN', 'KSFT', 'KUP', 'KLOW',
    'OPEN0', 'HIGH0', 'LOW0', 'VWAP0',
    'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60',
    'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
    'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60',
    'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60',
    'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60',
    'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60',
    'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60',
    'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60',
    'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60',
    'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60',
}


def load_factors_tanhw(use_cache=True):
    """Load factors with intrinsic-class preprocessing.

    89 factors kept as-is, 69 cross-sectional tanh(IQR*2); tanh-class outliers
    beyond the training-set q001/q999 are set to NaN.

    use_cache=True: read the preprocessed cache (PROCESSED_DIR/factors.parquet),
        saving several minutes. Cache spec: 69 tanh + 89 keep, NaN thresholds computed on
        TRAIN_START..TRAIN_END (look-ahead safe), keep-class not de-extremed.
    use_cache=False: process on the fly (thresholds use TRAIN_START..TRAIN_END).
    """
    factor_cols = ALPHA158_NAMES
    if use_cache:
        print("[1] Loading preprocessed factor cache (69 tanh + 89 keep)...")
        long_df = pd.read_parquet(PROCESSED_DIR / "factors.parquet")
        n_tanh = len([c for c in factor_cols if c in TANH_FACTOR])
        print(f"  tanh: {n_tanh} | keep: {len(factor_cols) - n_tanh} | shape={long_df.shape}")
        return long_df, factor_cols

    print("[1] Loading raw factor cache (on-the-fly, 69 tanh + 89 keep)...")
    long_df = pd.read_parquet(RAW_DATA_DIR / "factors.parquet")
    n_tanh = len([c for c in factor_cols if c in TANH_FACTOR])
    n_keep = len(factor_cols) - n_tanh
    print(f"  tanh: {n_tanh} | keep: {n_keep}")

    # ---- 1) Per-factor q001/q999 thresholds on the training set ----
    tr_start = pd.Timestamp(TRAIN_START)
    tr_end = pd.Timestamp(TRAIN_END)
    train_idx = (long_df.index.get_level_values("date") >= tr_start) & \
                (long_df.index.get_level_values("date") <= tr_end)
    lo = {}
    hi = {}
    for c in factor_cols:
        s = long_df.loc[train_idx, c].replace([np.inf, -np.inf], np.nan)
        q = s.quantile([0.001, 0.999])
        lo[c] = float(q.loc[0.001])
        hi[c] = float(q.loc[0.999])
    n_drop_total = 0
    for c in factor_cols:
        col = long_df[c].replace([np.inf, -np.inf], np.nan)
        mask = (col < lo[c]) | (col > hi[c])
        n_drop_total += int(mask.sum())
        long_df[c] = col.mask(mask)
    print(f"  Outliers set to NaN (train q001/q999 thresholds): {n_drop_total} points "
          f"({100 * n_drop_total / (len(long_df) * len(factor_cols)):.4f}%)")

    # ---- 2) Cross-sectional tanh / keep as-is ----
    for c in factor_cols:
        col = long_df[c]
        if c in TANH_FACTOR:
            g = col.groupby(level="date", group_keys=False)
            med = g.transform("median")
            q25 = g.transform(lambda x: x.quantile(0.25))
            q75 = g.transform(lambda x: x.quantile(0.75))
            iqr = (q75 - q25).replace(0, np.nan)
            z = (col - med) / (iqr * 2.0)  # IQR*2: 77% linear region, 23% saturated
            z = z.replace([np.inf, -np.inf], np.nan)
            long_df[c] = np.tanh(z)
        else:
            long_df[c] = col
    return long_df, factor_cols


def load_label_data():
    """Load post-adjusted VWAP (wide frame)."""
    return pd.read_parquet(RAW_DATA_DIR / "vwap_adj.parquet")


def make_label(vwap_adj, h, lo=None, hi=None):
    """Build an h-horizon label: post-adjusted VWAP return, optional quantile trim.

    label = vwap_adj[t+h+1] / vwap_adj[t+1] - 1
    Quantile trimming (prevents extremes from polluting the cross-sectional
    z-score statistics):
      - Samples outside [lo, hi] are set to NaN (thresholds from the training
        set, no look-ahead).
      - Thresholds are computed per h (distributions differ across horizons).
    """
    ret = vwap_adj.shift(-(h + 1)) / vwap_adj.shift(-1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    if lo is not None and hi is not None:
        ret = ret.mask((ret < lo) | (ret > hi))
    ret.index.name = "date"
    return ret


def build_xy_clip(factor_long, ret, dates_set, factor_cols, mask, clip=3.0):
    """Build (X, y, date_ids); label is cross-sectional z-scored then clipped to +/-clip.

    Same clipping for train/valid/test.
    """
    X_parts, y_parts, d_parts = [], [], []
    fac_dates_set = set(factor_long.index.get_level_values("date").unique())
    ret_dates_set = set(ret.index)
    mask_dates_set = set(mask.index) if mask is not None else set()
    for tj in sorted(dates_set):
        if tj not in fac_dates_set or tj not in ret_dates_set:
            continue
        X_j = factor_long.xs(tj, level="date")[factor_cols]
        r_j = ret.loc[tj]
        common = X_j.index.intersection(r_j.index)
        X_j = X_j.loc[common]
        r_j = r_j.loc[common]
        if tj in mask_dates_set:
            in_idx = mask.loc[tj]
            in_idx = in_idx[in_idx].index
            common = common.intersection(in_idx)
            X_j = X_j.loc[common]
            r_j = r_j.loc[common]
        r_j = r_j.replace([np.inf, -np.inf], np.nan).dropna()
        X_j = X_j.loc[r_j.index]
        m = X_j.notna().all(axis=1)
        X_j = X_j[m]
        r_j = r_j[m]
        if len(r_j) < 20:
            continue
        r_mean = r_j.mean()
        r_std = r_j.std()
        if r_std < 1e-12:
            continue
        z = ((r_j - r_mean) / r_std).clip(-clip, clip)
        X_parts.append(X_j.to_numpy(dtype=np.float32))
        y_parts.append(z.to_numpy(dtype=np.float32))
        d_parts.append(np.full(len(X_j), hash(tj) % (2**31), dtype=np.int64))
    if not X_parts:
        return None, None, None
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    d = np.concatenate(d_parts, axis=0)
    return X, y, d


def compute_label_quantile(vwap_adj_train, h, q_lo=0.005, q_hi=0.995):
    """Quantile thresholds from the training-set label distribution (per h)."""
    r = (vwap_adj_train.shift(-(h + 1)) / vwap_adj_train.shift(-1) - 1.0)
    r = r.replace([np.inf, -np.inf], np.nan)
    s = r.stack()
    return float(s.quantile(q_lo)), float(s.quantile(q_hi))


def get_tradable(start, end):
    m = pd.read_parquet(RAW_DATA_DIR / "tradable_mask.parquet")
    return m.loc[pd.Timestamp(start):pd.Timestamp(end)]


def get_train_union_tradable(full_start, full_end, train_start, train_end):
    """Look-ahead-safe training universe: union of stocks tradable during the
    training period, broadcast to the full range.
    """
    train_mask = get_tradable(train_start, train_end)
    in_train = train_mask.any(axis=0)
    full_mask = get_tradable(full_start, full_end)
    out = full_mask.copy()
    out.loc[:, :] = False
    cols = in_train.index.intersection(out.columns)
    out[cols] = in_train[cols].values
    return out


def predict_alpha_pit(model, factor_long, factor_cols, test_dates, member_mask):
    """Point-in-time alpha prediction: only stocks tradable on each day are scored.

    Stocks not in ``member_mask`` on a given day are dropped (alpha = NaN) so
    they never enter the backtest selection pool.

    Stocks with any NaN/inf factor are dropped (alpha = NaN). This is aligned
    with the training sample construction (``build_xy_clip``), which drops any
    row with a missing factor — the model is trained only on factor-complete
    stocks, so it is evaluated only on the same. The dropped stocks are
    predominantly suspended/halted names with no usable factor values; dropping
    them keeps the test universe consistent with training rather than imputing
    a neutral value that the model never saw.
    """
    alpha_preds = {}
    model.eval()
    with torch.no_grad():
        for t in sorted(test_dates):
            if t not in factor_long.index.get_level_values("date"):
                continue
            X_t = factor_long.xs(t, level="date")[factor_cols]
            if t in member_mask.index:
                in_idx = member_mask.loc[t]
                in_idx = in_idx[in_idx].index
                X_t = X_t.loc[X_t.index.intersection(in_idx)]
            if len(X_t) < 20:
                continue
            # Drop stocks with any NaN/inf factor (aligned with build_xy_clip).
            # These are mostly suspended names; imputing would let the model
            # score stocks it was never trained on.
            X_t = X_t.replace([np.inf, -np.inf], np.nan)
            m = X_t.notna().all(axis=1)
            X_t = X_t[m]
            if len(X_t) < 20:
                continue
            X_arr = X_t.to_numpy(dtype=np.float32)
            pred = model(torch.as_tensor(X_arr, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            alpha_preds[t] = pd.Series(pred, index=X_t.index)
    return pd.DataFrame(alpha_preds).T


def metrics_for_alpha(alpha, ret):
    """Daily IC / RankIC / ICIR / RankICIR for an alpha matrix.

    ``alpha`` and ``ret`` are (date x code) wide frames or Series.
    """
    ics, rankics = [], []
    for t in alpha.index.intersection(ret.index):
        a = alpha.loc[t].dropna()
        r = ret.loc[t]
        common = a.index.intersection(r.index)
        a = a.loc[common]
        r = r.loc[common].dropna()
        a = a.loc[r.index]
        if len(a) < 20:
            continue
        ics.append(a.corr(r))
        rankics.append(a.corr(r, method="spearman"))
    ic = np.array([x for x in ics if not np.isnan(x)])
    rc = np.array([x for x in rankics if not np.isnan(x)])
    return {
        "IC": ic.mean() if len(ic) else np.nan,
        "ICIR": ic.mean() / ic.std() if len(ic) and ic.std() > 0 else np.nan,
        "RankIC": rc.mean() if len(rc) else np.nan,
        "RankICIR": rc.mean() / rc.std() if len(rc) and rc.std() > 0 else np.nan,
    }

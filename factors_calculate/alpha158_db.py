"""Qlib Alpha158 factor library.

Computes 158 daily features per the Microsoft Qlib default ``Alpha158`` dataset spec:
  - 9 candlestick shape factors (KMID/KLEN/..., using only same-day OHLC)
  - 4 same-day price ratios (OPEN0/HIGH0/LOW0/VWAP0, relative to close)
  - 145 rolling features (29 templates x 5 windows [5,10,20,30,60])

Formulas per the Microsoft Qlib ``Alpha158`` spec (see
https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py).
All prices are unified to the **post-adjusted** scale:
  - OHLC uses ``S_DQ_ADJOPEN/ADJHIGH/ADJLOW/ADJCLOSE``
  - VWAP is corrected by the adjustment factor: ``VWAP_adj = S_DQ_AVGPRICE x S_DQ_ADJFACTOR``
  - volume uses raw ``S_DQ_VOLUME`` (Qlib semantics is raw volume)

Supports PIT filtering by index constituents (CSI300/CSI500, etc.), no survivorship bias.

Usage:
    from factors_calculate.alpha158_db import compute_alpha158

    # Long table (ML-friendly): index=(date, wind_code), columns=158 factors
    long_df = compute_alpha158("2023-01-03", "2024-12-31", index_code="000300.SH")

    # Wide table dict (for backtest): {factor_name: T x N wide table}
    wide = compute_alpha158(..., as_frame="wide")
    wide["ROC20"]  # feed directly to FactorBacktest

    # Both
    long_df, wide = compute_alpha158(..., as_frame="both")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.calendar_db import trading_dates
from data_fetch.marketdata_db import _fetch_eod, _to_wide

__all__ = ["compute_alpha158", "ALPHA158_NAMES", "ALPHA158_WINDOWS"]

# Rolling windows (Alpha158 default)
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]

# Small constant to prevent division by zero (Qlib formulas uniformly use 1e-12)
_EPS = 1e-12

# ───────────────────────── Factor names ─────────────────────────

# A. 9 candlestick shape factors
_KBAR_NAMES = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

# B. 4 same-day price ratios
_PRICE_NAMES = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]

# C. 29 rolling feature templates, each expanded over 5 windows -> 145
_ROLLING_TEMPLATES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI",          # price trend and volatility
    "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",         # interval position and quantile
    "IMAX", "IMIN", "IMXD",                               # high/low point time structure
    "CORR", "CORD",                                       # price-volume correlation
    "CNTP", "CNTN", "CNTD",                               # up/down day count statistics
    "SUMP", "SUMN", "SUMD",                               # return direction strength
    "VMA", "VSTD", "WVMA",                                # volume mean and volatility
    "VSUMP", "VSUMN", "VSUMD",                            # volume change direction strength
]

_ROLLING_NAMES = [f"{t}{d}" for t in _ROLLING_TEMPLATES for d in ALPHA158_WINDOWS]

# All 158 factor names (fixed order)
ALPHA158_NAMES = _KBAR_NAMES + _PRICE_NAMES + _ROLLING_NAMES
assert len(ALPHA158_NAMES) == 158, f"Alpha158 should have 158 factors, got {len(ALPHA158_NAMES)}"


# ───────────────────────── Base data ─────────────────────────


def _fetch_ohlcv(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    codes: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch and build 6 base wide tables (all post-adjusted).

    Returns:
        dict: open/high/low/close/vwap/volume, all (T x N) wide tables.
        vwap is corrected to the post-adjusted scale by the adjustment factor.
    """
    fields = [
        "S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE",
        "S_DQ_AVGPRICE", "S_DQ_ADJFACTOR", "S_DQ_VOLUME",
    ]
    key = cache_key("alpha158_ohlcv", str(start), str(end))
    if not force_refresh:
        cached = read_cache("alpha158_ohlcv", key)
        if cached is not None:
            long_all = cached
        else:
            long_all = _fetch_eod(start, end, fields)
            write_cache("alpha158_ohlcv", key, long_all)
    else:
        long_all = _fetch_eod(start, end, fields)
        write_cache("alpha158_ohlcv", key, long_all)

    open_ = _to_wide(long_all, "S_DQ_ADJOPEN")
    high = _to_wide(long_all, "S_DQ_ADJHIGH")
    low = _to_wide(long_all, "S_DQ_ADJLOW")
    close = _to_wide(long_all, "S_DQ_ADJCLOSE")
    avgprice = _to_wide(long_all, "S_DQ_AVGPRICE")
    factor = _to_wide(long_all, "S_DQ_ADJFACTOR")
    volume = _to_wide(long_all, "S_DQ_VOLUME")

    # Correct VWAP to the post-adjusted scale: raw VWAP x adjustment factor
    vwap = avgprice * factor

    # Align to close's index/columns (primary axis)
    idx = close.index
    cols = close.columns
    open_ = open_.reindex(index=idx, columns=cols)
    high = high.reindex(index=idx, columns=cols)
    low = low.reindex(index=idx, columns=cols)
    vwap = vwap.reindex(index=idx, columns=cols)
    volume = volume.reindex(index=idx, columns=cols)

    if codes is not None:
        codes_set = set(codes)
        keep = [c for c in cols if c in codes_set]
        open_ = open_[keep]
        high = high[keep]
        low = low[keep]
        close = close[keep]
        vwap = vwap[keep]
        volume = volume[keep]

    return {"open": open_, "high": high, "low": low, "close": close,
            "vwap": vwap, "volume": volume}


# ───────────────────────── Rolling operators (Qlib -> pandas) ─────────────────────────
# Roll column-wise over (T x N) wide tables, computing the whole table at once with a
# numpy stride view (replacing rolling.apply, hundreds of times faster). Within the
# window the time index t = 0..d-1 (0 = farthest day, d-1 = current day).


def _rolling_slope(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-day linear regression slope (against time t=0..d-1), vectorized.

    slope = cov(t, x) / var(t), where t is 0..d-1 within the window (mean (d-1)/2,
    variance known).
    """
    n = d
    t_mean = (n - 1) / 2.0
    t_var = (n * n - 1) / 12.0  # sum((t-t_mean)^2) for t=0..n-1
    # Rolling sums: x, x*t (using cumulative t weights, most recent day has the largest weight)
    # rolling.apply is still slow; use convolution/cumulative instead. Here use rolling.sum with weighting.
    # x_t corresponds to the t-th element in the window (0 = farthest). Build a weighted series:
    # each row multiplied by its position within the window.
    # Concise implementation: rolling sum of x, rolling sum of x*t.
    # x*t where t is the "offset from the window start", equivalent to rank-1 within the rolling window.
    # Cumulative method: sum_{i} x_i * i = sum_{k} (cumulative tail).
    # To avoid complexity, compute all at once with a numpy stride (column loop but vectorized rows).
    arr = df.to_numpy(dtype=float)
    T, N = arr.shape
    out = np.full((T, N), np.nan)
    # Within the rolling window t=0..d-1, need sum(x*t) and sum(x).
    # sum(x*t) = sum_{j=0}^{d-1} x[i-d+1+j] * j
    # Differential recursion: define S_k = sum_{j=0}^{k} ... is tedious; here directly use a sliding window view.
    if T >= d:
        # Build a sliding window view (T-d+1, d, N)
        shape = (T - d + 1, d, N)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        # Each window (d, N), t = 0..d-1
        t = np.arange(d).reshape(d, 1)
        x_sum = windows.sum(axis=1)                      # (T-d+1, N)
        xt_sum = (windows * t).sum(axis=1)               # (T-d+1, N)
        x_mean = x_sum / n
        cov = xt_sum / n - t_mean * x_mean
        slope = cov / t_var
        out[d - 1:] = slope
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _rolling_rsquare_resi(df: pd.DataFrame, d: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling d-day linear regression R-squared and end-point residual, vectorized.

    Returns (rsquare, resi). resi = x[-1] - (slope*(d-1) + intercept).
    """
    n = d
    t_mean = (n - 1) / 2.0
    t_var = (n * n - 1) / 12.0
    arr = df.to_numpy(dtype=float)
    T, N = arr.shape
    rsq = np.full((T, N), np.nan)
    resi = np.full((T, N), np.nan)
    if T >= d:
        shape = (T - d + 1, d, N)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        t = np.arange(d).reshape(d, 1)
        x_sum = windows.sum(axis=1)
        xt_sum = (windows * t).sum(axis=1)
        x_mean = x_sum / n
        x2_sum = (windows ** 2).sum(axis=1)
        cov = xt_sum / n - t_mean * x_mean
        slope = cov / t_var
        intercept = x_mean - slope * t_mean
        pred_last = slope * (d - 1) + intercept
        last = windows[:, -1, :]                          # last row of the window = current day
        resi_val = last - pred_last
        # R^2 = slope^2 * var(t) / var(x) (linear regression goodness of fit, equivalent to 1-ss_res/ss_tot)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_var = x2_sum / n - x_mean ** 2
            r = slope ** 2 * t_var / x_var
            r = np.where(x_var > 0, r, 0.0)
        rsq[d - 1:] = r
        resi[d - 1:] = resi_val
    return (pd.DataFrame(rsq, index=df.index, columns=df.columns),
            pd.DataFrame(resi, index=df.index, columns=df.columns))


def _rolling_idxmax(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-day position of the max (0 = farthest day, d-1 = current day), vectorized."""
    arr = df.to_numpy(dtype=float)
    T, N = arr.shape
    out = np.full((T, N), np.nan)
    if T >= d:
        shape = (T - d + 1, d, N)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        out[d - 1:] = np.argmax(windows, axis=1).astype(float)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _rolling_idxmin(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-day position of the min (0 = farthest day, d-1 = current day), vectorized."""
    arr = df.to_numpy(dtype=float)
    T, N = arr.shape
    out = np.full((T, N), np.nan)
    if T >= d:
        shape = (T - d + 1, d, N)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        out[d - 1:] = np.argmin(windows, axis=1).astype(float)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# ───────────────────────── Feature computation ─────────────────────────


def _kbar_features(o, h, l, c) -> dict[str, pd.DataFrame]:
    """9 candlestick shape features (using only same-day OHLC)."""
    up_body = np.maximum(o, c)          # Greater($open, $close)
    dn_body = np.minimum(o, c)          # Less($open, $close)
    hl = h - l
    return {
        "KMID": (c - o) / o,
        "KLEN": hl / o,
        "KMID2": (c - o) / (hl + _EPS),
        "KUP": (h - up_body) / o,
        "KUP2": (h - up_body) / (hl + _EPS),
        "KLOW": (dn_body - l) / o,
        "KLOW2": (dn_body - l) / (hl + _EPS),
        "KSFT": (2 * c - h - l) / o,
        "KSFT2": (2 * c - h - l) / (hl + _EPS),
    }


def _price_features(o, h, l, c, vwap) -> dict[str, pd.DataFrame]:
    """4 same-day price ratios (relative to close)."""
    return {
        "OPEN0": o / c,
        "HIGH0": h / c,
        "LOW0": l / c,
        "VWAP0": vwap / c,
    }


def _rolling_features(o, h, l, c, vwap, vol) -> dict[str, pd.DataFrame]:
    """145 rolling features (29 templates x 5 windows)."""
    feats: dict[str, pd.DataFrame] = {}
    # Pre-compute common intermediates
    log_vol = np.log(vol + 1.0)
    ret = c.pct_change(fill_method=None)                       # close/Ref(close,1) - 1
    ret = ret.where(vol.notna() & c.notna())                   # suspended days do not count returns
    vol_chg = vol.pct_change(fill_method=None) + 1.0           # volume/Ref(volume,1)
    vol_chg = vol_chg.where(vol.notna())
    abs_ret = (c / c.shift(1) - 1.0).abs()
    wvma_raw = abs_ret * vol                                   # |close/Ref-1| * volume

    for d in ALPHA158_WINDOWS:
        # ── Price trend and volatility ──
        feats[f"ROC{d}"] = c.shift(d) / c
        feats[f"MA{d}"] = c.rolling(d).mean() / c
        feats[f"STD{d}"] = c.rolling(d).std() / c
        feats[f"BETA{d}"] = _rolling_slope(c, d) / c
        feats[f"RSQR{d}"], feats[f"RESI{d}"] = _rolling_rsquare_resi(c, d)
        feats[f"RESI{d}"] = feats[f"RESI{d}"] / c

        # ── Interval position and quantile ──
        feats[f"MAX{d}"] = h.rolling(d).max() / c
        feats[f"MIN{d}"] = l.rolling(d).min() / c
        feats[f"QTLU{d}"] = c.rolling(d).quantile(0.8) / c
        feats[f"QTLD{d}"] = c.rolling(d).quantile(0.2) / c
        # Rank: time-series quantile of the current value within the past d-day series
        feats[f"RANK{d}"] = c.rolling(d).rank(pct=True)
        max_d = h.rolling(d).max()
        min_d = l.rolling(d).min()
        feats[f"RSV{d}"] = (c - min_d) / (max_d - min_d + _EPS)

        # ── High/low point time structure (vectorized) ──
        imax = _rolling_idxmax(h, d)
        imin = _rolling_idxmin(l, d)
        feats[f"IMAX{d}"] = imax / d
        feats[f"IMIN{d}"] = imin / d
        feats[f"IMXD{d}"] = (imax - imin) / d

        # ── Price-volume correlation ──
        feats[f"CORR{d}"] = c.rolling(d).corr(log_vol)
        feats[f"CORD{d}"] = ret.rolling(d).corr(np.log(vol_chg.where(vol_chg > 0) + 1.0))

        # ── Up/down day count statistics ──
        up_day = (c > c.shift(1)).astype(float)
        dn_day = (c < c.shift(1)).astype(float)
        feats[f"CNTP{d}"] = up_day.rolling(d).mean()
        feats[f"CNTN{d}"] = dn_day.rolling(d).mean()
        feats[f"CNTD{d}"] = up_day.rolling(d).mean() - dn_day.rolling(d).mean()

        # ── Return direction strength ──
        diff = c.diff()                                        # close - Ref(close,1)
        up_amt = diff.clip(lower=0.0)
        dn_amt = (-diff).clip(lower=0.0)                       # positive part of Ref(close,1) - close
        abs_amt = diff.abs()
        feats[f"SUMP{d}"] = up_amt.rolling(d).sum() / (abs_amt.rolling(d).sum() + _EPS)
        feats[f"SUMN{d}"] = dn_amt.rolling(d).sum() / (abs_amt.rolling(d).sum() + _EPS)
        feats[f"SUMD{d}"] = (up_amt.rolling(d).sum() - dn_amt.rolling(d).sum()) / (
            abs_amt.rolling(d).sum() + _EPS
        )

        # ── Volume mean and volatility ──
        # Note: the denominator is the day's vol; on suspended days vol=0 divides by _EPS and explodes to 1e18.
        # Set suspended days to NaN (no volume, the factor is meaningless).
        vol_halted = vol <= 0  # suspended days (vol=0)
        feats[f"VMA{d}"] = (vol.rolling(d).mean() / (vol + _EPS)).where(~vol_halted)
        feats[f"VSTD{d}"] = (vol.rolling(d).std() / (vol + _EPS)).where(~vol_halted)
        feats[f"WVMA{d}"] = wvma_raw.rolling(d).std() / (wvma_raw.rolling(d).mean() + _EPS)

        # ── Volume change direction strength ──
        vdiff = vol.diff()                                     # volume - Ref(volume,1)
        vup = vdiff.clip(lower=0.0)
        vdn = (-vdiff).clip(lower=0.0)
        vabs = vdiff.abs()
        feats[f"VSUMP{d}"] = vup.rolling(d).sum() / (vabs.rolling(d).sum() + _EPS)
        feats[f"VSUMN{d}"] = vdn.rolling(d).sum() / (vabs.rolling(d).sum() + _EPS)
        feats[f"VSUMD{d}"] = (vup.rolling(d).sum() - vdn.rolling(d).sum()) / (
            vabs.rolling(d).sum() + _EPS
        )

    return feats


# ───────────────────────── Main entry ─────────────────────────


def compute_alpha158(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    index_code: str | None = None,
    codes: list[str] | None = None,
    as_frame: str = "long",
    drop_non_member: bool = False,
    force_refresh: bool = False,
):
    """Compute Alpha158 factors.

    Args:
        start/end: date range (closed interval).
        index_code: Wind index code (e.g. "000300.SH"/"000905.SH"). If given, filter by
            PIT constituents: columns take the union of constituents over the range,
            marked in-index day by day. None = use codes or the whole market.
        codes: explicit ticker list (effective when index_code is None). None = whole market.
        as_frame: output shape:
            - "long": long table, index=(date, wind_code), columns=158 factors (ML-friendly).
            - "wide": dict[factor_name -> T x N wide table] (for backtest).
            - "both": (long_df, wide_dict).
        drop_non_member: when index_code is given, whether to drop factor values on
            non-in-index days (set to NaN).
            False = keep the union columns, still compute factors on non-in-index days
            (suitable for ML, configure the mask yourself);
            True = set non-in-index days to NaN (more rigorous for backtest).
        force_refresh: force refresh the market data cache.

    Returns:
        Per as_frame: a long table / wide table dict / tuple of both.
    """
    if as_frame not in ("long", "wide", "both"):
        raise ValueError(f"as_frame must be 'long'/'wide'/'both', got {as_frame!r}")

    # 1. Determine the ticker set
    member_mask: pd.DataFrame | None = None
    if index_code is not None:
        from data_fetch.index_members_db import get_index_member_mask

        member_mask = get_index_member_mask(index_code, start, end, force_refresh=force_refresh)
        codes = sorted(member_mask.columns.tolist())  # union of constituents over the range

    # 2. Fetch base OHLCV (all post-adjusted)
    ohlcv = _fetch_ohlcv(start, end, codes=codes, force_refresh=force_refresh)
    o, h, l, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
    vwap, vol = ohlcv["vwap"], ohlcv["volume"]

    # 3. Compute 158 factors
    feats: dict[str, pd.DataFrame] = {}
    feats.update(_kbar_features(o, h, l, c))
    feats.update(_price_features(o, h, l, c, vwap))
    feats.update(_rolling_features(o, h, l, c, vwap, vol))

    # Arrange in fixed order
    feats = {name: feats[name] for name in ALPHA158_NAMES}

    # 4. Constituent PIT: set non-in-index days to NaN
    if member_mask is not None and drop_non_member:
        mask = member_mask.reindex(index=c.index, columns=c.columns).fillna(False)
        for name in feats:
            feats[name] = feats[name].where(mask)

    # 5. Assemble output
    if as_frame == "wide":
        return feats

    # Long table: each factor wide table -> stack, then concatenate horizontally
    long_parts = []
    for name in ALPHA158_NAMES:
        s = feats[name].stack()  # (date, wind_code) -> value
        s.name = name
        long_parts.append(s)
    long_df = pd.concat(long_parts, axis=1)
    long_df.index.names = ["date", "wind_code"]

    if as_frame == "long":
        return long_df
    return long_df, feats

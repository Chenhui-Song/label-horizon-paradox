"""Alpha158 factor name definitions (consistent with the official Qlib).

This file only defines factor names and does not compute
factors. Factors are computed by scripts/build_factors.py and stored in
data/a_share_data/. This file is referenced by training/backtesting
code for factor names.
"""
from __future__ import annotations

__all__ = ["ALPHA158_NAMES", "ALPHA158_WINDOWS"]

# Rolling windows (Alpha158 default)
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]

# ───────────────────────── Factor names ─────────────────────────

# A. 9 candlestick pattern features
_KBAR_NAMES = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

# B. 4 same-day price ratios
_PRICE_NAMES = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]

# C. 29 rolling feature templates, each expanded over 5 windows -> 145 features
_ROLLING_TEMPLATES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI",          # price trend and volatility
    "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",         # range position and quantiles
    "IMAX", "IMIN", "IMXD",                               # high/low point time structure
    "CORR", "CORD",                                       # price-volume correlation
    "CNTP", "CNTN", "CNTD",                               # up/down day counts
    "SUMP", "SUMN", "SUMD",                               # return direction strength
    "VMA", "VSTD", "WVMA",                                # volume mean and volatility
    "VSUMP", "VSUMN", "VSUMD",                            # volume change direction strength
]

_ROLLING_NAMES = [f"{t}{d}" for t in _ROLLING_TEMPLATES for d in ALPHA158_WINDOWS]

# All 158 factor names (fixed order)
ALPHA158_NAMES = _KBAR_NAMES + _PRICE_NAMES + _ROLLING_NAMES
assert len(ALPHA158_NAMES) == 158, f"Alpha158 should have 158 factors, got {len(ALPHA158_NAMES)}"

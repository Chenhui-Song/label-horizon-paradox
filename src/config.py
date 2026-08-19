"""Shared base configuration: date ranges, paths, device, horizons.

These are fixed project constants (not tunable hyperparameters). Tunable
training hyperparameters live in each entry script (``scripts/run_search.py``,
``scripts/run_blo.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch

# ---------------------------------------------------------------------------
# Date ranges (chronological split, look-ahead safe)
# ---------------------------------------------------------------------------
TRAIN_START, TRAIN_END = "2020-01-02", "2023-12-31"
VALID_START, VALID_END = "2024-01-02", "2024-12-31"
TEST_START, TEST_END = "2025-01-02", "2026-06-30"

# Data build date range. Data starts from 2019-06 (half-year before TRAIN_START
# to provide rolling-60 warmup for factor computation). Label quantile thresholds
# are computed on TRAIN_START..TRAIN_END (look-ahead safe for valid/test).
DATA_START = "2019-06-01"
DATA_END = "2026-06-30"

# ---------------------------------------------------------------------------
# Device and horizons
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = list(range(1, 11))   # h = 1..10
TARGET_H = 10                    # Delta = 10 (target horizon)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = _ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "a_share_data"
PROCESSED_DIR = DATA_DIR / "a_share_data_processed"

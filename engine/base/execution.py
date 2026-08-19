"""Execution model: execution price + volume matrices.

Based on the time-point model's (TimePoint) exec_window and exec_lag, produces:
  - exec_price: (T x N) execution price matrix (trade price), index=execution day
  - exec_volume: (T x N) window volume matrix, index=execution day

Reads parquet directly from data/a_share_data/, no database connection.
  - VWAPExec.exec_price -> vwap_adj.parquet (already back-adjusted and cleaned)
  - CloseExec.exec_price -> adjclose.parquet (back-adjusted close)
  - exec_volume -> not supported (ladder backtest does not enable the volume constraint),
    returns an empty DataFrame

The execution price matrix is the core of the vectorized engine -- as long as the execution
price can be precomputed into a matrix, the engine formulas stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import RAW_DATA_DIR
from .timeline import TimePoint

__all__ = ["ExecutionModel", "CloseExec", "VWAPExec"]


def _load(name: str, start, end, codes=None) -> pd.DataFrame:
    """Read parquet from data and slice by date/stock."""
    path = RAW_DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Please run scripts/build_backtest_data.py first")
    df = pd.read_parquet(path)
    df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if codes is not None:
        codes_set = set(codes)
        df = df[[c for c in df.columns if c in codes_set]]
    return df


class ExecutionModel:
    """Execution model base class.

    The matrices returned by exec_price / exec_volume are indexed by **execution day**;
    the caller aligns them to signal days by exec_lag.
    """

    def exec_price(
        self,
        trading_dates: pd.DatetimeIndex,
        tp: TimePoint,
        codes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the (T x N) execution price matrix, index=execution day. Implemented by subclasses."""
        raise NotImplementedError

    def exec_volume(
        self,
        trading_dates: pd.DatetimeIndex,
        tp: TimePoint,
        codes: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return the (T x N) window volume matrix, index=execution day."""
        raise NotImplementedError


class CloseExec(ExecutionModel):
    """Close-price execution (daily-frequency baseline).

    exec_price = back-adjusted close price (adjclose); exec_volume not supported (returns empty).
    index = trading day (= execution day).
    """

    def exec_price(self, trading_dates, tp, codes=None):
        start, end = trading_dates[0], trading_dates[-1]
        return _load("adjclose", start, end, codes=codes)

    def exec_volume(self, trading_dates, tp, codes=None):
        return pd.DataFrame(index=trading_dates)


class VWAPExec(ExecutionModel):
    """Window VWAP execution (intraday).

    exec_price = vwap_adj.parquet (back-adjusted and cleaned VWAP); exec_volume not supported.
    index = execution day (one row per day).

    The vwap_adj in data has already been back-adjusted and cleaned (invalid values
    replaced with the close), so it can be read directly without further cleaning.
    """

    def exec_price(self, trading_dates, tp, codes=None):
        start, end = trading_dates[0], trading_dates[-1]
        return _load("vwap_adj", start, end, codes=codes)

    def exec_volume(self, trading_dates, tp, codes=None):
        return pd.DataFrame(index=trading_dates)

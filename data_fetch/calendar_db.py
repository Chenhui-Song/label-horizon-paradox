"""Trading calendar.

Data source: DDB WINDDATA_MONTH.AShareCalendar (SSE exchange, A-share trading days).
The single source of truth for the backtest timeline. Comes with local cache.
"""

from __future__ import annotations

import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.connection import dc

__all__ = ["trading_dates", "align_to_trading", "is_trading_day", "next_trading_day", "prev_trading_day"]

# A-share calendar uses the SSE; SSE and SZSE trading days are identical
_EXCH = "SSE"
_CACHE_NAME = "calendar"


def _fetch_all() -> pd.DataFrame:
    """Pull all SSE trading days from DDB; returns a DatetimeIndex."""
    s = dc.ddb()
    df = s.run(
        f'''
        select TRADE_DAYS from loadTable("dfs://WINDDATA_MONTH", "AShareCalendar")
        where S_INFO_EXCHMARKET = "{_EXCH}"
        order by TRADE_DAYS asc
        '''
    )
    # TRADE_DAYS is a "YYYYMMDD" string -> Timestamp
    dates = pd.to_datetime(df["TRADE_DAYS"], format="%Y%m%d")
    return dates


def _get_all_dates() -> pd.DatetimeIndex:
    """Get all trading days (with cache)."""
    key = cache_key(_CACHE_NAME, _EXCH, "all")
    cached = read_cache(_CACHE_NAME, key)
    if cached is not None:
        return pd.DatetimeIndex(cached["date"])
    dates = _fetch_all()
    write_cache(_CACHE_NAME, key, pd.DataFrame({"date": dates}))
    return dates


def trading_dates(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Return the trading days within [start, end] (closed interval).

    Args may be 'YYYY-MM-DD' strings or Timestamps. Returns an ascending DatetimeIndex.
    """
    all_dates = _get_all_dates()
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    mask = (all_dates >= start) & (all_dates <= end)
    return all_dates[mask]


def align_to_trading(date: str | pd.Timestamp, direction: str = "forward") -> pd.Timestamp:
    """Align a date to the nearest trading day.

    direction:
        'forward'  -> next trading day on or after the date
        'backward' -> previous trading day on or before the date
        'nearest'  -> closest trading day
    """
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    if direction == "forward":
        after = all_dates[all_dates >= date]
        return after[0] if len(after) else all_dates[-1]
    if direction == "backward":
        before = all_dates[all_dates <= date]
        return before[-1] if len(before) else all_dates[0]
    if direction == "nearest":
        idx = all_dates.get_indexer([date], method="nearest")[0]
        return all_dates[idx]
    raise ValueError(f"direction must be forward/backward/nearest, got {direction}")


def is_trading_day(date: str | pd.Timestamp) -> bool:
    """Check whether a date is a trading day."""
    return pd.Timestamp(date) in _get_all_dates()


def next_trading_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """Next trading day (excluding the date itself)."""
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    after = all_dates[all_dates > date]
    return after[0] if len(after) else all_dates[-1]


def prev_trading_day(date: str | pd.Timestamp) -> pd.Timestamp:
    """Previous trading day (excluding the date itself)."""
    date = pd.Timestamp(date)
    all_dates = _get_all_dates()
    before = all_dates[all_dates < date]
    return before[-1] if len(before) else all_dates[0]

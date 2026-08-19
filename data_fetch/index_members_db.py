"""Index constituents (PIT, Point-in-Time).

Data source: DDB WINDDATA_MONTH.AIndexMembers.

Determines whether a stock is in an index on a given day via the
S_CON_INDATE/S_CON_OUTDATE interval, with no survivorship bias.
Supports CSI300 (000300.SH) / CSI500 (000905.SH) / CSI1000 (000852.SH)
and any Wind index code.

Usage:
    from data_fetch.index_members_db import (
        get_index_members, get_index_member_mask,
    )
    # constituents on a given day
    codes = get_index_members("000300.SH", "2024-06-28")
    # daily in-index mask over a range (T x N wide table)
    mask = get_index_member_mask("000300.SH", "2024-01-02", "2024-06-30")
"""

from __future__ import annotations

import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.connection import dc

__all__ = ["get_index_members", "get_index_member_mask", "INDEX_CSI300", "INDEX_CSI500", "INDEX_CSI1000"]

_MEMBER_TABLE = 'loadTable("dfs://WINDDATA_MONTH", "AIndexMembers")'

# Common index codes
INDEX_CSI300 = "000300.SH"
INDEX_CSI500 = "000905.SH"
INDEX_CSI1000 = "000852.SH"


def _fetch_members(index_code: str) -> pd.DataFrame:
    """Pull the full historical constituent entry/exit records for an index (long table).

    Returns:
        DataFrame[wind_code, entry, remove]: entry/remove are Timestamps;
        remove being NaT means still in the index today.
    """
    s = dc.ddb()
    df = s.run(
        f'''
        select S_CON_WINDCODE as wind_code, S_CON_INDATE as indate,
               S_CON_OUTDATE as outdate
        from {_MEMBER_TABLE}
        where S_INFO_WINDCODE = "{index_code}"
        '''
    )
    df["entry"] = pd.to_datetime(df["indate"], format="%Y%m%d", errors="coerce")
    df["remove"] = pd.to_datetime(df["outdate"], format="%Y%m%d", errors="coerce")
    return df[["wind_code", "entry", "remove"]]


def get_index_member_mask(
    index_code: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Daily in-index constituent mask wide table (True=the stock is in the index on that day).

    Based on the S_CON_INDATE/S_CON_OUTDATE interval of AIndexMembers (PIT, no
    survivorship bias). An empty OUTDATE means still in the index today.

    Args:
        index_code: Wind index code, e.g. "000300.SH" / "000905.SH".
        start/end: date range.
        force_refresh: force refresh the cache.

    Returns:
        (T x N) boolean wide table, index=date (trading-day alignment is the
        caller's responsibility), columns=wind_code.
    """
    key = cache_key("idxmember", index_code, str(start), str(end))
    if not force_refresh:
        cached = read_cache("idxmember", key)
        if cached is not None:
            return cached

    df = _fetch_members(index_code)

    # Align the date axis with the trading calendar to avoid non-trading-day rows
    from data_fetch.calendar_db import trading_dates

    dates = trading_dates(start, end)
    dates = pd.DatetimeIndex(dates)  # ensure it is an Index (Series[-1] would KeyError)
    codes = sorted(df["wind_code"].unique())
    mask = pd.DataFrame(False, index=dates, columns=codes)
    last_date = dates[-1]
    for row in df.itertuples():
        if pd.isna(row.entry):
            continue
        remove = row.remove if pd.notna(row.remove) else last_date
        seg = (dates >= row.entry) & (dates <= remove)
        if row.wind_code in mask.columns:
            mask.loc[seg, row.wind_code] = True
    write_cache("idxmember", key, mask)
    return mask


def get_index_members(
    index_code: str,
    date: str | pd.Timestamp,
    force_refresh: bool = False,
) -> list[str]:
    """In-index constituents of an index on a given trading day (PIT).

    Args:
        index_code: Wind index code.
        date: trading day.
        force_refresh: force refresh the cache.

    Returns:
        Sorted list of in-index constituent wind_codes on that day.
    """
    date = pd.Timestamp(date)
    mask = get_index_member_mask(index_code, date, date, force_refresh=force_refresh)
    if date not in mask.index:
        return []
    return sorted(mask.columns[mask.loc[date]].tolist())

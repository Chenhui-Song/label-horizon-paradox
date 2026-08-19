"""Market data layer.

Data sources:
  - Daily: DDB WINDDATA_MONTH.AShareEODPrices (main market data, DOUBLE prices,
    with built-in adjustment / limit status / trading status)
  - Minute: DDB OLAP_Minute.minute (INT prices need /10000, used for intraday
    signals / execution)

All market data is returned as **wide tables** (index=date, columns=wind_code),
matching the matrix shape of the vectorized engine. Comes with local parquet cache.

Key conventions:
  - Returns are computed from post-adjusted prices by default to avoid ex-dividend gaps.
  - Limit up/down detection: S_DQ_CLOSE >= S_DQ_LIMIT (limit up, cannot buy) /
    <= S_DQ_STOPPING (limit down, cannot sell).
  - Minute-bar prices are also INT and need /10000; time is of TIME type,
    filtered by minute(time) for the window.
"""

from __future__ import annotations

import pandas as pd

from data_fetch.cache import cache_key, read_cache, write_cache
from data_fetch.connection import dc

__all__ = [
    # daily
    "get_eod_prices",
    "get_returns",
    "get_limit_status",
    "get_tradable_mask",
    "get_volume",
    # index
    "get_index_close",
    # minute
    "get_minute_vwap",
    "get_minute_volume",
    "get_minute_returns",
    "get_minute_price",
]

_DAY_TABLE = 'loadTable("dfs://WINDDATA_MONTH", "AShareEODPrices")'
_MIN_TABLE = 'loadTable("dfs://OLAP_Minute", "minute")'
_PRICE_SCALE = 10000  # minute-bar INT price scale factor


# ───────────────────────── Daily market data ─────────────────────────


def _ddb_date_range(start: str | pd.Timestamp, end: str | pd.Timestamp) -> str:
    """Generate a DDB date range literal: 2024.01.01:2024.12.31."""
    s = pd.Timestamp(start).strftime("%Y.%m.%d")
    e = pd.Timestamp(end).strftime("%Y.%m.%d")
    return f"{s}:{e}"


def _fetch_eod(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    fields: list[str],
) -> pd.DataFrame:
    """Pull a daily market data long table from DDB; returns a long table with wind_code/TRADE_DT + the specified fields."""
    field_str = ", ".join(["S_INFO_WINDCODE", "TRADE_DT", *fields])
    dr = _ddb_date_range(start, end)
    s = dc.ddb()
    df = s.run(
        f'''
        select {field_str} from {_DAY_TABLE}
        where TRADE_DT between {dr}
        '''
    )
    return df


def _to_wide(long_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Long table -> wide table (index=TRADE_DT, columns=wind_code, values=value_col)."""
    return long_df.pivot(index="TRADE_DT", columns="S_INFO_WINDCODE", values=value_col).sort_index()


def get_eod_prices(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    field: str = "S_DQ_ADJCLOSE",
    codes: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Daily price wide table (index=date, columns=wind_code).

    Args:
        field: price field, default S_DQ_ADJCLOSE (post-adjusted close). Options:
            S_DQ_CLOSE (raw close) / S_DQ_OPEN / S_DQ_ADJOPEN (post-adjusted open), etc.
        codes: specify tickers; None = whole market.
        force_refresh: force refresh the cache.

    Returns a wide table with date as index and wind_code as columns.
    """
    key = cache_key("eod", field, str(start), str(end))
    if not force_refresh:
        cached = read_cache("eod", key)
        if cached is not None:
            df = cached
        else:
            df = _to_wide(_fetch_eod(start, end, [field]), field)
            write_cache("eod", key, df)
    else:
        df = _to_wide(_fetch_eod(start, end, [field]), field)
        write_cache("eod", key, df)
    if codes is not None:
        codes_set = set(codes)
        df = df[[c for c in df.columns if c in codes_set]]
    return df


def get_returns(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    field: str = "S_DQ_ADJCLOSE",
    codes: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Daily return wide table (pct_change based on adjusted prices).

    Returns a (T x N) return matrix, matching the shape of the vectorized engine.
    """
    px = get_eod_prices(start, end, field=field, codes=codes, force_refresh=force_refresh)
    return px.pct_change(fill_method=None)


def get_limit_status(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limit up/down status matrices.

    Returns (limit_up, limit_down), two (T x N) boolean matrices:
        limit_up=True  -> limit up (cannot buy)
        limit_down=True -> limit down (cannot sell)
    Detection: S_DQ_CLOSE >= S_DQ_LIMIT (limit up) / <= S_DQ_STOPPING (limit down).
    """
    key_up = cache_key("limit_up", str(start), str(end))
    key_dn = cache_key("limit_dn", str(start), str(end))
    if not force_refresh:
        cu = read_cache("limit_up", key_up)
        cd = read_cache("limit_dn", key_dn)
        if cu is not None and cd is not None:
            return cu, cd
    df = _fetch_eod(start, end, ["S_DQ_CLOSE", "S_DQ_LIMIT", "S_DQ_STOPPING"])
    close = _to_wide(df, "S_DQ_CLOSE")
    lim = _to_wide(df, "S_DQ_LIMIT")
    stop = _to_wide(df, "S_DQ_STOPPING")
    # Align the three tables to the same index/columns
    lim = lim.reindex(index=close.index, columns=close.columns)
    stop = stop.reindex(index=close.index, columns=close.columns)
    limit_up = (close >= lim).fillna(False)
    limit_down = (close <= stop).fillna(False)
    write_cache("limit_up", key_up, limit_up)
    write_cache("limit_dn", key_dn, limit_down)
    return limit_up, limit_down


def get_tradable_mask(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Tradable mask wide table (True=tradable).

    Excludes suspended stocks (S_DQ_TRADESTATUS == '停牌'). Limit-up/down lock
    is handled separately by the constraints module by trade direction (cannot
    buy at limit up, cannot sell at limit down); not excluded here.
    """
    key = cache_key("tradable", str(start), str(end))
    if not force_refresh:
        cached = read_cache("tradable", key)
        if cached is not None:
            return cached
    df = _fetch_eod(start, end, ["S_DQ_TRADESTATUS"])
    mask = _to_wide(df, "S_DQ_TRADESTATUS")
    tradable = mask != "停牌"
    tradable = tradable.fillna(False)
    write_cache("tradable", key, tradable)
    return tradable


def get_volume(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Daily volume wide table (unit: lots, S_DQ_VOLUME). Used for volume participation constraints."""
    key = cache_key("dayvol", str(start), str(end))
    if not force_refresh:
        cached = read_cache("dayvol", key)
        if cached is not None:
            return cached
    df = _fetch_eod(start, end, ["S_DQ_VOLUME"])
    wide = _to_wide(df, "S_DQ_VOLUME")
    write_cache("dayvol", key, wide)
    return wide


# ───────────────────────── Minute market data ─────────────────────────


def _fetch_minute_window(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window: tuple[str, str] | None,
    agg: str,
) -> pd.DataFrame:
    """Aggregate the minute-bar table by window; returns a long table (wind_code, date, value).

    agg: 'vwap' -> turover.sum()/volumw.sum()
         'volume' -> volumw.sum()
    window: e.g. ('10:30','11:30'); None = full day.
    """
    dr = _ddb_date_range(start, end)
    if agg == "vwap":
        # Note: turover/volumw are both BIGINT; DDB integer division truncates
        # decimals (e.g. 11.55 -> 11). Must cast to double() before dividing,
        # otherwise VWAP is all integers (placeholder values), deviating from
        # close by >5% and being judged invalid.
        val_expr = "double(turover.sum()) / double(volumw.sum())"
        val_col = "vwap"
    elif agg == "volume":
        val_expr = "volumw.sum()"
        val_col = "volume"
    else:
        raise ValueError(f"agg must be vwap/volume, got {agg}")

    where_time = ""
    if window is not None:
        t0, t1 = window
        # DDB pair syntax: minute(time) between 10:30m:11:30m (both ends are MINUTE type)
        where_time = f" and minute(time) between {t0}m:{t1}m"

    sql = f'''
        select wind_code, date, {val_expr} as {val_col}
        from {_MIN_TABLE}
        where date between {dr}{where_time}
        group by wind_code, date
    '''
    s = dc.ddb()
    df = s.run(sql)
    # VWAP = turover (yuan) / volumw (shares); already real yuan/shares, no /10000 needed.
    # (Only the OHLC price fields are INT and need /10000; turover/volumw are real values.)
    return df


def get_minute_vwap(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window: tuple[str, str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Window VWAP wide table (index=date, columns=wind_code).

    VWAP = turnover (yuan) / volume (shares); both are real values, no /10000
    scaling (only OHLC INT fields need /10000).
    window: e.g. ('10:30','11:30') / ('14:50','15:00'); None = full-day VWAP.
    """
    wkey = "all" if window is None else f"{window[0]}-{window[1]}"
    key = cache_key("mvwap", wkey, str(start), str(end))
    if not force_refresh:
        cached = read_cache("mvwap", key)
        if cached is not None:
            return cached
    df = _fetch_minute_window(start, end, window, "vwap")
    wide = df.pivot(index="date", columns="wind_code", values="vwap").sort_index()
    write_cache("mvwap", key, wide)
    return wide


def get_minute_volume(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window: tuple[str, str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Window volume wide table (unit: shares). Used for volume participation constraints."""
    wkey = "all" if window is None else f"{window[0]}-{window[1]}"
    key = cache_key("mvol", wkey, str(start), str(end))
    if not force_refresh:
        cached = read_cache("mvol", key)
        if cached is not None:
            return cached
    df = _fetch_minute_window(start, end, window, "volume")
    wide = df.pivot(index="date", columns="wind_code", values="volume").sort_index()
    write_cache("mvol", key, wide)
    return wide


def get_minute_returns(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Minute return matrix (index=datetime, columns=wind_code).

    pct_change of minute close prices (/10000), used for intraday position
    return accumulation.
    """
    key = cache_key("mret", str(start), str(end))
    if not force_refresh:
        cached = read_cache("mret", key)
        if cached is not None:
            return cached
    dr = _ddb_date_range(start, end)
    s = dc.ddb()
    df = s.run(
        f'''
        select wind_code, concat(date, time) as dt, close
        from {_MIN_TABLE}
        where date between {dr}
        '''
    )
    df["close"] = df["close"] / _PRICE_SCALE
    df["dt"] = pd.to_datetime(df["dt"])
    wide = df.pivot(index="dt", columns="wind_code", values="close").sort_index()
    rets = wide.pct_change()
    write_cache("mret", key, rets)
    return rets


def get_minute_price(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    at_time: str = "10:30",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Snapshot price wide table at a given time point (index=date, columns=wind_code).

    Takes the close price of the minute bar at at_time (e.g. '10:30') each day,
    used for intraday signal data cutoff. Price already /10000.
    """
    key = cache_key("mprice", at_time, str(start), str(end))
    if not force_refresh:
        cached = read_cache("mprice", key)
        if cached is not None:
            return cached
    dr = _ddb_date_range(start, end)
    s = dc.ddb()
    df = s.run(
        f'''
        select wind_code, date, close
        from {_MIN_TABLE}
        where date between {dr} and minute(time) = {at_time}m
        '''
    )
    df["close"] = df["close"] / _PRICE_SCALE
    wide = df.pivot(index="date", columns="wind_code", values="close").sort_index()
    write_cache("mprice", key, wide)
    return wide


# ───────────────────────── Index market data ─────────────────────────

_INDEX_TABLE = 'loadTable("dfs://WINDDATA_MONTH", "AIndexEODPrices")'


def get_index_close(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    codes: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Index daily close prices (wide table: index=date, columns=index code).

    Data source: DDB WINDDATA_MONTH.AIndexEODPrices (official Wind index close,
    not adjusted). Used as benchmark for backtest comparison.

    Args:
        start/end: date range.
        codes: list of index codes (e.g. ["000300.SH", "000905.SH", "000852.SH"]).
            None = all indices in the table.
        force_refresh: force refresh the cache.

    Returns:
        (T x N) wide table, index=date, columns=index code, values=close price.
    """
    key = cache_key("idxclose", str(start), str(end), str(codes))
    if not force_refresh:
        cached = read_cache("idxclose", key)
        if cached is not None:
            return cached
    dr = _ddb_date_range(start, end)
    s = dc.ddb()
    code_filter = ""
    if codes is not None:
        code_list = ", ".join(f'"{c}"' for c in codes)
        code_filter = f" and S_INFO_WINDCODE in [{code_list}]"
    df = s.run(
        f'''
        select S_INFO_WINDCODE as wind_code, TRADE_DT as date, S_DQ_CLOSE as close
        from {_INDEX_TABLE}
        where TRADE_DT between {dr}{code_filter}
        '''
    )
    wide = df.pivot(index="date", columns="wind_code", values="close").sort_index()
    write_cache("idxclose", key, wide)
    return wide

"""Performance evaluation: return/risk metrics + factor metrics (IC/ICIR/RankIC)."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    # return/risk
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "calmar",
    "nav",
    # factor metrics
    "ic_series",
    "rank_ic_series",
    "icir",
    "ic_mean",
    "ic_win_rate",
    # turnover
    "turnover_series",
]


# ── Return/risk metrics ──


def nav(returns: pd.Series) -> pd.Series:
    """Cumulative net asset value, starting at 1.0."""
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Annualized return."""
    if len(returns) == 0:
        return 0.0
    total = (1 + returns).prod()
    years = len(returns) / periods
    if years <= 0:
        return 0.0
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series, periods: int = 252) -> float:
    """Annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods)


def sharpe(returns: pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    """Sharpe ratio (annualized). rf is the annualized risk-free rate."""
    vol = annualized_vol(returns, periods)
    if vol == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods)
    return (ann_ret - rf) / vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (positive value, e.g. 0.3 means a 30% drawdown)."""
    v = nav(returns)
    peak = v.cummax()
    dd = (v - peak) / peak
    return abs(dd.min())


def calmar(returns: pd.Series, periods: int = 252) -> float:
    """Calmar ratio (annualized return / max drawdown)."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return annualized_return(returns, periods) / mdd


# ── Factor metrics ──


def ic_series(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """IC series: Pearson correlation between factor values and next-period returns per period.

    Args:
        factor: factor values (T x N).
        forward_returns: next-period returns (T x N), aligned with factor (row t holds
            the return of period t+1 corresponding to the period-t factor).

    Returns:
        IC series, index=dates.
    """
    common_cols = factor.columns.intersection(forward_returns.columns)
    f = factor[common_cols]
    r = forward_returns[common_cols]
    # row-wise correlation
    ic = f.corrwith(r, axis=1)
    return ic


def rank_ic_series(factor: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
    """Rank IC series: Spearman rank correlation between factor values and next-period returns per period."""
    common_cols = factor.columns.intersection(forward_returns.columns)
    f = factor[common_cols].rank()
    r = forward_returns[common_cols].rank()
    return f.corrwith(r, axis=1)


def ic_mean(ic: pd.Series) -> float:
    """IC mean."""
    return ic.mean() if len(ic) else 0.0


def icir(ic: pd.Series) -> float:
    """ICIR = IC mean / IC standard deviation."""
    std = ic.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return ic.mean() / std


def ic_win_rate(ic: pd.Series) -> float:
    """IC win rate (proportion of periods with IC > 0)."""
    if len(ic) == 0:
        return 0.0
    return (ic > 0).mean()


# ── Turnover ──


def turnover_series(holdings: pd.DataFrame) -> pd.Series:
    """Turnover rate series: sum of |delta w| per period.

    Args:
        holdings: holding weight matrix (T x N).

    Returns:
        Turnover rate per period, index=dates. First period is 0 (no previous value).
    """
    delta = holdings.diff().abs().sum(axis=1)
    delta.iloc[0] = holdings.iloc[0].abs().sum()  # initial position build
    return delta

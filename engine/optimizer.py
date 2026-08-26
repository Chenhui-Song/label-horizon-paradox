"""Portfolio optimizer: protocol + built-in implementations.

Protocol OptimizerProtocol.solve(alpha, current_holdings, context) -> target holdings.
State lives in Account, not in the optimizer (the optimizer is a stateless pure function).

Built-in implementations:
  - NormalizerOptimizer: normalize the positive part of alpha into weights (unconstrained)
  - TopNOptimizer: take the top N equally weighted
  - ThreeIndexTopNOptimizer: take the top N from each of the three major indices, equally weighted
    (per-index N supported via a dict; set an index's N to 0 to skip it, which is
    equivalent to a single-index backtest)

Customization: implement OptimizerProtocol and pass it as a parameter to OptimBacktest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "OptimizerProtocol",
    "OptimContext",
    "NormalizerOptimizer",
    "TopNOptimizer",
    "ThreeIndexTopNOptimizer",
]


class OptimizerProtocol(Protocol):
    """Optimizer protocol: solve for new target holdings from current holdings + alpha + context."""

    def solve(
        self,
        alpha: pd.Series,
        current_holdings: pd.Series,
        context: "OptimContext",
    ) -> pd.Series:
        """Return target holding weights (N,)."""
        ...


@dataclass
class OptimContext:
    """Environment info needed by the optimizer; the framework fills it in, the optimizer
    takes what it needs.

    Attributes:
        max_weight: per-stock weight cap.
        max_turnover: turnover cap (one-sided). None = unconstrained.
        (sigma/exposures/industry/benchmark are reserved fields, not used by built-in optimizers)
    """

    sigma: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None
    exposure_limits: dict[str, float] | None = None
    industry: pd.Series | None = None
    industry_limit: float | None = None
    benchmark: pd.Series | None = None
    max_weight: float = 0.05
    max_turnover: float | None = None


# -- Built-in implementations --


@dataclass
class NormalizerOptimizer:
    """Simplest: normalize the positive part of alpha into weights (unconstrained).

    Negative alpha is set to 0 (long-only), positive alpha is normalized.
    """

    def solve(self, alpha, current_holdings, context):
        pos = alpha.clip(lower=0).fillna(0.0)
        total = pos.sum()
        if total <= 0:
            return pd.Series(0.0, index=alpha.index)
        return pos / total


@dataclass
class TopNOptimizer:
    """Take the top N alpha names, equally weighted.

    Args:
        n: number of names to hold.
        member_masks: optional list of daily constituent masks (T x N boolean wide
            frames, index=date). When provided, candidates are restricted to the
            union of these masks on each day (e.g. CSI300+CSI500+CSI1000 to pick
            from the CSI1800 universe). When None, the full alpha universe is used.
    """

    n: int = 50
    member_masks: list | None = None

    def solve(self, alpha, current_holdings, context):
        w = pd.Series(0.0, index=alpha.index)
        a = alpha.dropna()
        if self.member_masks:
            # Union of the provided constituent masks on the signal day.
            # Each mask may have different columns; reindex to alpha's index
            # (fill False) before the union so it covers all members.
            date = alpha.name
            union = pd.Series(False, index=a.index)
            for mask in self.member_masks:
                if date not in mask.index:
                    continue
                m = mask.loc[date]
                union = union | m.reindex(union.index, fill_value=False)
            if union.any():
                a = a.loc[a.index.intersection(union[union].index)]
        a = a.sort_values(ascending=False)
        if len(a) >= self.n:
            w.loc[a.head(self.n).index] = 1.0 / self.n
        elif len(a) > 0:
            w.loc[a.index] = 1.0 / len(a)
        return w


@dataclass
class ThreeIndexTopNOptimizer:
    """Take the top N alpha names from each of the three major indices, equally weighted.

    Each day, based on that day's index constituents, take the top N alpha names within
    CSI300/CSI500/CSI1000 respectively, equally weighted. Looks up the day's constituent mask
    via alpha.name (the date).

    Args:
        member_masks: {index_name: (T x N boolean wide table)} daily constituent masks for the
            three major indices. The key must contain 'csi300'/'csi500'/'csi1000'.
        n_per_index: top N per index. int (same N for all three) or dict mapping
            ``{"csi300": n1, "csi500": n2, "csi1000": n3}`` for per-index N. Default 50.
            Total target holdings = sum of the three Ns; each picked stock gets equal weight
            ``1 / total_N`` (cash residual if fewer stocks are available).
    """

    member_masks: dict
    n_per_index: int = 50

    def _n_for(self, key: str):
        """Return N for a given index key ('csi300'/'csi500'/'csi1000')."""
        if isinstance(self.n_per_index, dict):
            return self.n_per_index.get(key, 0)
        return self.n_per_index

    def solve(self, alpha: pd.Series, current_holdings: pd.Series, context: OptimContext) -> pd.Series:
        date = alpha.name  # date (OptimBacktest passes alpha.iloc[i], name = that row's date)
        w = pd.Series(0.0, index=alpha.index)
        a = alpha.dropna()
        if len(a) == 0:
            return w

        # Each index's constituents for the day (boolean Series, index=wind_code)
        masks = {}
        for key in self.member_masks:
            kl = key.lower()
            if "csi300" in kl:
                masks["csi300"] = self.member_masks[key]
            elif "csi500" in kl:
                masks["csi500"] = self.member_masks[key]
            elif "csi1000" in kl:
                masks["csi1000"] = self.member_masks[key]

        ns = {k: self._n_for(k) for k in ("csi300", "csi500", "csi1000")}
        total_n = sum(ns.values())
        if total_n == 0:
            return w
        for key in ("csi300", "csi500", "csi1000"):
            n = ns[key]
            if n <= 0:
                continue
            mask = masks.get(key)
            if mask is None or date not in mask.index:
                continue
            members = mask.loc[date]
            # Stocks that are constituents and have an alpha value
            valid = members[members].index.intersection(a.index)
            if len(valid) == 0:
                continue
            a_sub = a.loc[valid].sort_values(ascending=False)
            pick = a_sub.head(n).index
            if len(pick) > 0:
                w.loc[pick] = 1.0 / total_n  # equal weight across all three indices' picks
        return w


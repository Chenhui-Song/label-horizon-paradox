"""Constraint handling: target position -> executable position.

Constraint types:
  - Limit up/down: limit-up stocks have their buy weight set to 0 (cannot buy in),
    limit-down stocks have their sell weight set to 0 (cannot sell out)
  - Suspension: not tradable, weight frozen at the previous day
  - Volume cap: per-stock rebalance amount <= volume x participation_rate
  - Weight cap: per-stock max weight max_weight

Residual weight handling after constraints: keep as cash (conservative, close to real trading),
no re-normalization. Switchable via residual='cash'/'renormalize'.

All operations are vectorized, acting on (T x N) matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["Constraints", "apply_constraints"]


@dataclass
class Constraints:
    """Constraint config.

    Args:
        max_weight: per-stock max weight, default 0.05 (5%). None = unlimited.
        participation_rate: volume participation cap, default 0.05 (5%). None = unlimited.
        residual: residual weight handling, 'cash' (keep as cash) or 'renormalize' (re-normalize).
        rollover: when a target name is limit-up (cannot buy in) on the execution day,
            redirect its blocked weight to substitute names — the next-highest-alpha names
            that are tradable and not limit-up — instead of leaving it as cash. Default
            True. Limit-down (cannot sell) and suspensions are real constraints and stay
            frozen (not rollable). Requires the signal-day alpha to be passed to
            ``apply_target``.
    """

    max_weight: float | None = 0.05
    participation_rate: float | None = 0.05
    residual: Literal["cash", "renormalize"] = "cash"
    rollover: bool = True


def apply_constraints(
    target_weights: pd.DataFrame,
    prev_holdings: pd.DataFrame,
    limit_up: pd.DataFrame,
    limit_down: pd.DataFrame,
    tradable: pd.DataFrame,
    exec_volume: pd.DataFrame | None = None,
    capital: float = 1.0,
    constraints: Constraints | None = None,
) -> pd.DataFrame:
    """Apply constraints to target positions and return the executable holdings matrix.

    Args:
        target_weights: target weights (T x N), index=date, columns=wind_code.
        prev_holdings: previous day's holdings (T x N), aligned with target_weights.
            Typically prev_holdings = target_weights.shift(1).fillna(0).
        limit_up: limit-up mask (T x N), True = limit up (cannot buy in).
        limit_down: limit-down mask (T x N), True = limit down (cannot sell out).
        tradable: tradable mask (T x N), True = tradable (suspensions excluded).
        exec_volume: volume (T x N), used for the volume constraint. None = unconstrained.
        capital: capital size, used to convert weights into a turnover-amount participation constraint. Default 1.0.
        constraints: constraint config; None = default Constraints().

    Returns:
        Executable holdings matrix (T x N), same shape as target_weights.
    """
    if constraints is None:
        constraints = Constraints()

    # Align all matrices to target_weights' index/columns
    idx, cols = target_weights.index, target_weights.columns
    w = target_weights.reindex(index=idx, columns=cols).fillna(0.0)
    prev = prev_holdings.reindex(index=idx, columns=cols).fillna(0.0)
    lu = limit_up.reindex(index=idx, columns=cols).fillna(False)
    ld = limit_down.reindex(index=idx, columns=cols).fillna(False)
    td = tradable.reindex(index=idx, columns=cols).fillna(False)

    # 1. Suspension: not tradable, weight frozen at the previous day
    #    Where tradable=False, holdings = prev_holdings
    frozen = ~td  # suspended or not tradable
    w = w.where(~frozen, prev)

    # 2. Limit up/down:
    #    Limit up (lu=True): cannot buy in -> target weight cannot exceed previous day, take min(w, prev)
    #    Limit down (ld=True): cannot sell out -> target weight cannot go below previous day, take max(w, prev)
    buy_blocked = lu & (w > prev)
    w = w.where(~buy_blocked, prev)
    sell_blocked = ld & (w < prev)
    w = w.where(~sell_blocked, prev)

    # 3. Weight cap: per-stock max_weight
    if constraints.max_weight is not None:
        w = w.clip(upper=constraints.max_weight)
        # When shorting is allowed the lower bound is also limited (here long-only, lower bound 0)
        w = w.clip(lower=0.0)

    # 3b. Total weight cap: frozen old holdings from limit-down/suspension + new target may cause sum > 1 (implicit leverage).
    #     Frozen stocks that cannot be sold are kept, the new target is scaled down proportionally to ensure sum <= 1 (remainder stays as cash).
    total = w.sum(axis=1)
    over = total > 1.0
    if over.any():
        # For rows with sum>1, scale down proportionally so sum=1 (preserving frozen stocks' relative weights)
        scale = (1.0 / total.where(over, 1.0)).where(over, 1.0)
        w = w.mul(scale, axis=0)

    # 4. Volume constraint: per-stock rebalance amount <= volume x participation_rate
    if constraints.participation_rate is not None and exec_volume is not None:
        ev = exec_volume.reindex(index=idx, columns=cols).fillna(0.0)
        max_trade_w = (ev * constraints.participation_rate) / capital
        delta = (w - prev).abs()
        over = delta > max_trade_w.where(max_trade_w > 0, np.inf)
        # Scale the excess proportionally
        scale = np.where(over, max_trade_w / delta.replace(0, np.nan), 1.0)
        scale = pd.DataFrame(scale, index=idx, columns=cols).fillna(1.0)
        w = prev + (w - prev) * scale

    # 5. Residual weight handling
    if constraints.residual == "renormalize":
        total = w.sum(axis=1).replace(0, np.nan)
        w = w.div(total, axis=0).fillna(0.0)
    # 'cash': no handling, remainder stays as cash (weight sum < 1)

    return w

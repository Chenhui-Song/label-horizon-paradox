"""Account state: stateful holdings / cash / NAV management.

Core method apply_target: target position -> constraints -> execution -> update holdings/cash/NAV.
Advances period by period (stateful), equivalent to the vectorized constraint logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constraints import Constraints, apply_constraints

__all__ = ["Account", "AccountSnapshot"]


@dataclass
class AccountSnapshot:
    """Account snapshot for a period (for historical records)."""

    date: pd.Timestamp
    holdings: pd.Series         # holding weights for the day
    nav: float                  # NAV
    cash: float                 # cash weight
    turnover: float             # turnover for the day
    cost: float                 # cost for the day
    gross_return: float         # gross return for the day
    net_return: float           # net return for the day


@dataclass
class Account:
    """Stateful account.

    Attributes:
        holdings: current holding weights (N,), index=wind_code.
        nav: cumulative NAV (starts at 1.0).
        cash: cash weight (the remainder of 1 - holdings.abs().sum(); for long-only it is 1 - sum).
        history: list of per-period snapshots.
    """

    holdings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    nav: float = 1.0
    cash: float = 1.0
    history: list[AccountSnapshot] = field(default_factory=list)

    def settle(self, date: pd.Timestamp, returns: pd.Series) -> AccountSnapshot:
        """Settle the current period's holding returns, update NAV, and let holdings drift with
        returns (held unchanged).

        Called on non-execution days: settle P&L with current holdings x period returns, NAV compounds;
        also update holdings to the drifted weights (weights change with each stock's move, normalized),
        so the next apply_target's prev_hold reflects the true drifted holdings and turnover is computed correctly.

        Drift formula: w_i' = w_i x (1+r_i) / (1+gross_ret), where gross_ret = sum w_i x r_i.
        After normalization the weights sum to 1, consistent with NAV compounding by (1+gross_ret).

        Args:
            date: current period date.
            returns: current period returns (N,).

        Returns:
            The settlement snapshot.
        """
        idx = self.holdings.index
        rets = returns.reindex(idx).fillna(0.0)
        gross_ret = float((self.holdings * rets).sum())
        self.nav *= (1 + gross_ret)
        # Holdings drift: held unchanged, weights change with price moves (normalized to keep sum=1)
        if 1 + gross_ret > 0:
            drifted = self.holdings * (1 + rets) / (1 + gross_ret)
            self.holdings = drifted
        snap = AccountSnapshot(
            date=date,
            holdings=self.holdings.copy(),
            nav=self.nav,
            cash=self.cash,
            turnover=0.0,
            cost=0.0,
            gross_return=gross_ret,
            net_return=gross_ret,
        )
        self.history.append(snap)
        return snap

    def apply_target(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series,
        limit_up: pd.Series,
        limit_down: pd.Series,
        tradable: pd.Series,
        exec_volume: pd.Series | None = None,
        exec_price: pd.Series | None = None,
        mark_price: pd.Series | None = None,
        prev_close: pd.Series | None = None,
        constraints: Constraints | None = None,
        cost_model: "CostModel | None" = None,
        alpha: pd.Series | None = None,
    ) -> AccountSnapshot:
        """Rebalance: target position -> constraints -> execution -> update holdings.

        Two-segment NAV computation (when executing at VWAP):
          Segment 1: old holdings, previous close -> day's VWAP, return = old holdings x (VWAP/prev_close - 1), minus cost
          Segment 2: new holdings, day's VWAP -> day's close, return = new holdings x (close/VWAP - 1)
        Segment 2 (close vs VWAP after buying) is the new holdings' intraday
        return from the VWAP fill to the close, not slippage; slippage is not
        modeled separately.
        Without exec_price, degrades to close-price execution (only cost is charged, returns computed by settle).

        Args:
            date: current period date (execution day).
            target_weights: target weights (N,).
            limit_up/limit_down/tradable: current period constraint masks (N,).
            exec_volume: current period volume (optional).
            exec_price: execution price (VWAP). When provided, enables two-segment NAV computation.
            mark_price: mark price (day's close price), used in segment 2.
            prev_close: previous day's close price, used in segment 1. Must be provided when exec_price is given.
            constraints: constraint config.
            cost_model: transaction cost model (stamp duty on sells + commission both sides). None = zero cost.

        Returns:
            The rebalance snapshot.
        """
        if constraints is None:
            constraints = Constraints()
        if cost_model is None:
            from .costs import CostModel
            cost_model = CostModel()

        # Align index
        idx = target_weights.index
        prev_hold = self.holdings.reindex(idx).fillna(0.0)
        target = target_weights.reindex(idx).fillna(0.0)
        lu = limit_up.reindex(idx).fillna(False)
        ld = limit_down.reindex(idx).fillna(False)
        td = tradable.reindex(idx).fillna(False)

        # Constraints: target -> executable (per-period version, reuses apply_constraints' single-period logic)
        prev_df = prev_hold.to_frame().T
        prev_df.index = [date]
        target_df = target.to_frame().T
        target_df.index = [date]
        lu_df = lu.to_frame().T
        lu_df.index = [date]
        ld_df = ld.to_frame().T
        ld_df.index = [date]
        td_df = td.to_frame().T
        td_df.index = [date]
        ev_df = exec_volume.to_frame().T if exec_volume is not None else None
        if ev_df is not None:
            ev_df.index = [date]

        exec_hold = apply_constraints(
            target_weights=target_df,
            prev_holdings=prev_df,
            limit_up=lu_df,
            limit_down=ld_df,
            tradable=td_df,
            exec_volume=ev_df,
            constraints=constraints,
        )
        new_hold = exec_df_to_series(exec_hold, idx)

        # Rollover: weight blocked by limit-up (cannot buy in) is redirected to
        # substitute names — the next-highest-alpha names that are tradable and
        # not limit-up today — instead of being left as cash. Limit-down (cannot
        # sell) and suspensions are real constraints and stay frozen.
        if constraints.rollover and alpha is not None:
            new_hold = _rollover_limit_up(new_hold, target, prev_hold, lu, td, alpha)

        # Turnover + exact cost (stamp duty on sells + commission both sides)
        delta = new_hold - prev_hold
        sell_amount = float((-delta).clip(lower=0).sum())  # sold weights (stamp duty)
        buy_amount = float(delta.clip(lower=0).sum())      # bought weights
        turnover = sell_amount + buy_amount                # == |delta|.sum()
        cost = cost_model.cost_from_legs(sell_amount, buy_amount)

        # Two-segment NAV computation (VWAP execution)
        seg1_ret = 0.0  # old holdings: previous close -> VWAP
        seg2_ret = 0.0  # new holdings: VWAP -> day's close
        if exec_price is not None and mark_price is not None and prev_close is not None:
            ep = exec_price.reindex(idx).fillna(0.0)
            mp = mark_price.reindex(idx).fillna(0.0)
            pc = prev_close.reindex(idx).fillna(0.0)
            # Segment 1: old holdings, previous close -> VWAP
            valid1 = (pc > 0) & (ep > 0)
            ret1 = pd.Series(0.0, index=idx)
            ret1.loc[valid1] = ep.loc[valid1] / pc.loc[valid1] - 1.0
            seg1_ret = float((prev_hold * ret1).sum())
            # Segment 2: new holdings, VWAP -> day's close
            valid2 = (ep > 0) & (mp > 0)
            ret2 = pd.Series(0.0, index=idx)
            ret2.loc[valid2] = mp.loc[valid2] / ep.loc[valid2] - 1.0
            seg2_ret = float((new_hold * ret2).sum())
            # Segment 1 return -> minus cost -> segment 2 return
            self.nav *= (1 + seg1_ret) * (1 - cost) * (1 + seg2_ret)
            # Drift new holdings over segment 2 (VWAP -> close), normalized to sum=1,
            # consistent with settle's drift so the next day's prev_hold is accurate.
            if 1 + seg2_ret > 0:
                new_hold = new_hold * (1 + ret2) / (1 + seg2_ret)
        else:
            # No VWAP: degrade to close-price execution, returns computed by settle, here only charge cost
            self.nav *= (1 - cost)

        self.holdings = new_hold
        self.cash = max(0.0, 1.0 - new_hold.sum())

        snap = AccountSnapshot(
            date=date,
            holdings=new_hold.copy(),
            nav=self.nav,
            cash=self.cash,
            turnover=turnover,
            cost=cost,
            gross_return=seg1_ret + seg2_ret,
            net_return=seg1_ret + seg2_ret - cost,
        )
        self.history.append(snap)
        return snap


def exec_df_to_series(df: pd.DataFrame, idx: pd.Index) -> pd.Series:
    """Single-row DataFrame -> Series."""
    return df.iloc[0].reindex(idx).fillna(0.0)


def _rollover_limit_up(new_hold, target, prev_hold, lu, td, alpha):
    """Redirect limit-up-blocked buy weight to substitute names by alpha rank.

    The weight the target wanted to buy into limit-up names but could not
    (``target - prev_hold`` where limit-up blocked the buy) is collected and
    reallocated equally to the highest-alpha names that are tradable and not
    limit-up today, up to the per-name target weight. Substitutes are picked
    from names not already at their target weight, so the rollover only fills
    the cash gap left by the blocked buys.

    Args:
        new_hold: post-constraint holdings (limit-up names already frozen at prev).
        target: target weights.
        prev_hold: previous holdings.
        lu: limit-up mask (True = cannot buy).
        td: tradable mask (True = tradable).
        alpha: signal-day alpha (used to rank substitutes).

    Returns:
        Updated holdings with blocked weight rolled over to substitutes.
    """
    # Weight blocked by limit-up: target wanted to buy (target > prev_hold) but
    # the name is limit-up, so new_hold was frozen at prev_hold.
    blocked = (target > prev_hold) & lu
    blocked_w = float((target[blocked] - prev_hold[blocked]).sum())
    if blocked_w <= 1e-10:
        return new_hold
    # Candidate substitutes: tradable, not limit-up, has an alpha value, and not
    # already at/above its target weight (so adding weight is a real fill).
    a = alpha.reindex(new_hold.index)
    cand = td & (~lu) & a.notna() & (new_hold < target - 1e-12)
    # Also allow names with zero target but positive alpha (fresh substitutes)
    cand_fresh = td & (~lu) & a.notna() & (target <= 1e-12) & (new_hold < 1e-12)
    cand = cand | cand_fresh
    if cand.sum() == 0:
        return new_hold
    # Rank candidates by alpha descending; fill blocked weight equally until used up.
    cand_alpha = a[cand].sort_values(ascending=False)
    # Per-name cap: don't exceed the target weight of the original pool (equal-weight
    # top-N gives 1/N per name; use the max target weight as the per-substitute cap).
    per_name_cap = float(target[target > 0].max()) if (target > 0).any() else 0.0
    if per_name_cap <= 0:
        per_name_cap = blocked_w  # fallback: give all to the top candidate
    remaining = blocked_w
    out = new_hold.copy()
    for code in cand_alpha.index:
        if remaining <= 1e-12:
            break
        add = min(per_name_cap, remaining)
        out[code] = out[code] + add
        remaining -= add
    return out

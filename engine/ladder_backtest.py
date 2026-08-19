"""Staggered sub-portfolio backtest (LadderBacktest): capital split into N independent
sub-accounts with staggered rebalance days.

Design:
  - Total capital split into N sub-accounts, each 1/N (default N=10)
  - Sub-account k's rebalance signal days = [k, k+freq, k+2*freq, ...] (offset=k)
  - Each sub-account rebalances every freq days, selecting stocks using the day's alpha signal
  - Non-rebalance days only settle returns (holdings unchanged)
  - Sub-account NAVs float independently, no rebalancing between them
  - Total portfolio NAV = sum of N sub-account NAVs

Effect: every day 1/N of capital is rebalancing, the rest is held -> overall turnover is
spread across each day, but each sub-account still rebalances at low frequency every freq days
(full signal utilization + turnover dispersion).

Reuses OptimBacktest (adds a rebalance_offset parameter); each sub-account runs independently,
NAVs are summed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .optim_backtest import OptimBacktest, OptimBacktestResult
from .base.account import Account
from .base.costs import CostModel
from .base.constraints import Constraints
from .base.execution import ExecutionModel
from .base.timeline import TimePoint
from .optimizer import OptimizerProtocol

__all__ = ["LadderBacktest", "LadderBacktestResult"]


@dataclass
class LadderBacktest:
    """Staggered sub-portfolio backtest.

    Args:
        alpha_signals: alpha signal matrix (T x N).
        optimizer: optimizer (shared across sub-accounts).
        n_ladders: number of sub-accounts (default 10), capital 1/n_ladders each.
        rebalance_freq: rebalance frequency of each sub-account (default 10).
            Sub-account k's rebalance days = [k, k+freq, k+2*freq, ...].
        cost_model / timepoint / execution / constraints: same as OptimBacktest.
    """

    alpha_signals: pd.DataFrame
    optimizer: OptimizerProtocol
    n_ladders: int = 10
    rebalance_freq: int = 10
    cost_model: CostModel | None = None
    timepoint: TimePoint = field(default_factory=TimePoint)
    execution: ExecutionModel | None = None
    constraints: Constraints = field(default_factory=Constraints)
    benchmark: str = "000300.SH"

    def run(self) -> "LadderBacktestResult":
        """Run N sub-accounts and sum their NAVs."""
        if self.cost_model is None:
            self.cost_model = CostModel()
        n = max(1, int(self.n_ladders))
        weight = 1.0 / n  # initial capital of each sub-account

        sub_results = []
        sub_navs = []  # NAV series of each sub-account (multiplied by weight)
        sub_holdings = []  # holdings of each sub-account (in weight units, multiplied by weight)
        for k in range(n):
            sub = OptimBacktest(
                alpha_signals=self.alpha_signals,
                optimizer=self.optimizer,
                cost_model=self.cost_model,
                timepoint=self.timepoint,
                execution=self.execution,
                rebalance_freq=self.rebalance_freq,
                rebalance_offset=k,  # staggered: sub-account k starts rebalancing on day k
                constraints=self.constraints,
                benchmark=self.benchmark,
            ).run()
            sub_results.append(sub)
            # Sub-account NAV x weight (capital share), as the contribution to the total portfolio
            sub_navs.append(sub.nav * weight)
            # Holdings also x weight (weight from the total portfolio's perspective)
            sub_holdings.append(sub.holdings * weight)

        # Total portfolio NAV = sum of sub-account NAVs
        total_nav = sum(sub_navs) if sub_navs else pd.Series(dtype=float)
        # Total portfolio daily return = total_nav.pct_change
        total_returns = total_nav.pct_change().fillna(0.0)
        # Total portfolio holdings = sum of sub-account holdings (weights stacked)
        if sub_holdings and len(sub_holdings[0]) > 0:
            total_holdings = sum(sub_holdings)
        else:
            total_holdings = pd.DataFrame()
        # Total turnover = sum of sub-account turnovers x weight (daily)
        sub_turnovers = [r.turnover * weight for r in sub_results]
        total_turnover = sum(sub_turnovers) if sub_turnovers else pd.Series(dtype=float)
        # Total cost likewise
        sub_costs = [r.cost * weight for r in sub_results]
        total_cost = sum(sub_costs) if sub_costs else pd.Series(dtype=float)

        return LadderBacktestResult(
            nav=total_nav,
            returns=total_returns,
            holdings=total_holdings,
            turnover=total_turnover,
            cost=total_cost,
            sub_results=sub_results,
            n_ladders=n,
        )


@dataclass
class LadderBacktestResult:
    """Staggered sub-portfolio backtest result."""

    nav: pd.Series
    returns: pd.Series
    holdings: pd.DataFrame
    turnover: pd.Series
    cost: pd.Series
    sub_results: list[OptimBacktestResult]
    n_ladders: int

    @property
    def sharpe(self) -> float:
        from performance import metrics as M
        return M.sharpe(self.returns)

    @property
    def annualized_return(self) -> float:
        from performance import metrics as M
        return M.annualized_return(self.returns)

    @property
    def max_drawdown(self) -> float:
        from performance import metrics as M
        return M.max_drawdown(self.returns)

    @property
    def avg_turnover(self) -> float:
        return self.turnover.mean() if len(self.turnover) else 0.0

    def report(self) -> None:
        print("=" * 55)
        print(f"Staggered Sub-Portfolio Backtest Result ({self.n_ladders} sub-accounts)")
        print("=" * 55)
        if len(self.nav) == 0:
            print("  (no data)")
            return
        print(f"  Backtest period: {self.nav.index[0].date()} ~ {self.nav.index[-1].date()}")
        print(f"  Trading days:    {len(self.nav)}")
        print(f"  Final NAV:       {self.nav.iloc[-1]:.4f}")
        print(f"  Annual return:   {self.annualized_return*100:.2f}%")
        print(f"  Sharpe:          {self.sharpe:.3f}")
        print(f"  Max drawdown:    {self.max_drawdown*100:.2f}%")
        print(f"  Avg turnover:    {self.avg_turnover*100:.2f}%")
        print("=" * 55)

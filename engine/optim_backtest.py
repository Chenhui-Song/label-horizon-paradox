"""Portfolio optimization backtest engine (period-by-period iteration).

Stateful: each period's holdings depend on the previous period (controls turnover),
T rounds of iteration.
Input: T x N alpha signal (produced outside the framework, read from saved parquet).
Holdings source: customizable optimizer (OptimizerProtocol).
Reuses the base layer: Account + constraints + costs + execution.

Market data is read directly from data/a_share_data/ parquet,
no dependency on the data/ data-access interface.

Typical usage:
    from engine import OptimBacktest, TopNOptimizer
    result = OptimBacktest(
        alpha_signals=alpha_df,   # T x N
        optimizer=TopNOptimizer(n=300),
    ).run()
    result.nav.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import RAW_DATA_DIR
from performance import metrics as M
from .base.account import Account
from .base.constraints import Constraints
from .base.costs import CostModel
from .base.execution import ExecutionModel
from .base.timeline import DailyClose, TimePoint
from .optimizer import NormalizerOptimizer, OptimContext, OptimizerProtocol

__all__ = ["OptimBacktest", "OptimBacktestResult"]


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


def _get_returns(start, end, codes=None) -> pd.DataFrame:
    """Daily returns (pct_change computed from back-adjusted close)."""
    px = _load("adjclose", start, end, codes=codes)
    return px.pct_change(fill_method=None)


def _get_limit_status(start, end, codes=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limit-up/limit-down status (limit_up, limit_down)."""
    lu = _load("limit_up", start, end, codes=codes)
    ld = _load("limit_down", start, end, codes=codes)
    return lu, ld


def _get_tradable_mask(start, end, codes=None) -> pd.DataFrame:
    """Tradable mask (True = tradable, not suspended)."""
    return _load("tradable_mask", start, end, codes=codes)


def _get_eod_adjclose(start, end, codes=None) -> pd.DataFrame:
    """Back-adjusted close price (mark price)."""
    return _load("adjclose", start, end, codes=codes)


@dataclass
class OptimBacktest:
    """Portfolio optimization backtest (period-by-period iteration).

    Args:
        alpha_signals: alpha signal matrix (T x N), index=date, columns=wind_code.
        optimizer: optimizer (implements OptimizerProtocol); default NormalizerOptimizer.
        cost_model: transaction cost model; default CostModel().
        timepoint: time-point model; default DailyClose (t close signal -> t+1 execution).
        execution: execution model; None = close-price execution,
            VWAPExec() = VWAP execution. When provided, the execution price is taken
            from execution; NAV is still settled at the close price (settle unchanged).
        rebalance_freq: rebalance frequency, rebalance every N trading days (default 1 = daily).
            Non-rebalance days only settle returns (NAV updates daily), no rebalancing.
        rebalance_offset: rebalance start offset (default 0). Rebalance signal days are
            [offset, offset+freq, offset+2*freq, ...]. Used for staggered sub-portfolios
            (LadderBacktest sub-accounts have different offsets), default 0.
        constraints: constraint config; default Constraints().
        benchmark: benchmark code, default "000300.SH".
    """

    alpha_signals: pd.DataFrame
    optimizer: OptimizerProtocol = field(default_factory=NormalizerOptimizer)
    cost_model: CostModel | None = None
    timepoint: TimePoint = field(default_factory=DailyClose)
    execution: ExecutionModel | None = None
    rebalance_freq: int = 1
    rebalance_offset: int = 0
    constraints: Constraints = field(default_factory=Constraints)
    benchmark: str = "000300.SH"

    def run(self) -> "OptimBacktestResult":
        """Run the backtest and return the result object."""
        alpha = self.alpha_signals
        if self.cost_model is None:
            self.cost_model = CostModel()

        dates = alpha.index
        start, end = dates[0], dates[-1]

        # Fetch market data (read from data)
        returns = _get_returns(start, end)
        limit_up, limit_down = _get_limit_status(start, end)
        tradable = _get_tradable_mask(start, end)

        common = alpha.columns.intersection(returns.columns)
        alpha = alpha[common]
        returns = returns.reindex(index=alpha.index, columns=common)
        limit_up = limit_up.reindex(index=alpha.index, columns=common).fillna(False)
        limit_down = limit_down.reindex(index=alpha.index, columns=common).fillna(False)
        tradable = tradable.reindex(index=alpha.index, columns=common).fillna(False)

        # Execution model: VWAP execution price + close mark price (NAV uses close price)
        # When execution is provided, apply_target uses VWAP to split the day's return
        # into two segments (old holdings close->VWAP, new holdings VWAP->close);
        # settle still uses close-price returns.
        exec_px: pd.DataFrame | None = None
        mark_px: pd.DataFrame | None = None
        if self.execution is not None:
            exec_px = self.execution.exec_price(dates, self.timepoint, codes=list(common))
            exec_px = exec_px.reindex(index=alpha.index, columns=common)
            mark_px = _get_eod_adjclose(start, end, codes=list(common))
            mark_px = mark_px.reindex(index=alpha.index, columns=common)

        # Period-by-period iteration
        # Timeline: signal day i close produces signal -> i+lag day VWAP execution
        # Key: apply_target is triggered on the [execution day] (not the signal day),
        #   ensuring settle always uses the pre-execution holdings.
        #   - Execution day i (i in exec_days): apply_target computes two-segment NAV
        #     (old holdings close[i-1]->VWAP[i], new holdings VWAP[i]->close[i]),
        #     covering the i-1->i return, no settle (avoids double counting).
        #   - Non-execution day: settle computes the close[i-1]->close[i] return with current holdings.
        account = Account()
        # Cost model (stamp duty on sells + commission both sides); slippage is
        # not modeled — the two-segment NAV only splits return at the VWAP price
        cost_model = self.cost_model
        lag = self.timepoint.exec_lag
        freq = max(1, int(self.rebalance_freq))

        # Previous day's close price (used in segment 1)
        prev_close_px = mark_px.shift(1) if mark_px is not None else None

        def _do_apply(exec_idx: int, signal_idx: int) -> None:
            """Execute the target holdings of the signal_idx signal on exec_idx day."""
            sig_alpha = alpha.iloc[signal_idx]
            target = self.optimizer.solve(
                sig_alpha, account.holdings,
                OptimContext(max_weight=self.constraints.max_weight or 0.05),
            )
            kwargs = dict(
                date=alpha.index[exec_idx],
                target_weights=target,
                limit_up=limit_up.iloc[exec_idx],
                limit_down=limit_down.iloc[exec_idx],
                tradable=tradable.iloc[exec_idx],
                constraints=self.constraints,
                cost_model=cost_model,
            )
            if self.constraints.rollover:
                kwargs["alpha"] = sig_alpha
            if exec_px is not None and mark_px is not None and prev_close_px is not None:
                kwargs["exec_price"] = exec_px.iloc[exec_idx]
                kwargs["mark_price"] = mark_px.iloc[exec_idx]
                kwargs["prev_close"] = prev_close_px.iloc[exec_idx]
            account.apply_target(**kwargs)

        # Execution day set: offset+lag, offset+lag+freq, ... (each rebalance signal day + lag)
        # Signal day offset executes on offset+lag; signal day offset+k*freq executes on offset+k*freq+lag.
        offset = max(0, int(self.rebalance_offset))
        exec_days = set()
        sig_indices = [i for i in range(offset, len(alpha), freq)]
        for sig_idx in sig_indices:
            exec_idx = sig_idx + lag
            if exec_idx < len(alpha):
                exec_days.add(exec_idx)

        # Advance day by day: apply_target on execution days, settle on non-execution days
        for i in range(1, len(alpha)):
            if i in exec_days:
                # Execution day: execute the target holdings of the i-lag signal (two-segment NAV covers i-1->i)
                _do_apply(i, i - lag)
            else:
                # Non-execution day: holdings unchanged, settle returns
                account.settle(alpha.index[i], returns.iloc[i])

        # Aggregate results: history contains both settle and apply_target snapshots, aggregated by date
        history = account.history
        if not history:
            return OptimBacktestResult(
                nav=pd.Series(dtype=float),
                returns=pd.Series(dtype=float),
                holdings=pd.DataFrame(),
                turnover=pd.Series(dtype=float),
                cost=pd.Series(dtype=float),
                account=account,
            )

        # Aggregate by date: same-day settle (returns) + apply_target (turnover/cost/new holdings)
        df = pd.DataFrame(
            {
                "date": [h.date for h in history],
                "nav": [h.nav for h in history],
                "net_return": [h.net_return for h in history],
                "turnover": [h.turnover for h in history],
                "cost": [h.cost for h in history],
                "holdings": [h.holdings for h in history],
            }
        )
        # Same day: take last for NAV, sum for returns/turnover/cost, take last for holdings
        agg = df.groupby("date").agg(
            nav=("nav", "last"),
            net_return=("net_return", "sum"),
            turnover=("turnover", "sum"),
            cost=("cost", "sum"),
            holdings=("holdings", "last"),
        )
        nav = agg["nav"]
        rets = agg["net_return"]
        turnover = agg["turnover"]
        cost = agg["cost"]
        holdings = pd.DataFrame(list(agg["holdings"]), index=agg.index)

        return OptimBacktestResult(
            nav=nav,
            returns=rets,
            holdings=holdings,
            turnover=turnover,
            cost=cost,
            account=account,
        )


@dataclass
class OptimBacktestResult:
    """Portfolio optimization backtest result."""

    nav: pd.Series
    returns: pd.Series
    holdings: pd.DataFrame
    turnover: pd.Series
    cost: pd.Series
    account: Account

    @property
    def sharpe(self) -> float:
        return M.sharpe(self.returns)

    @property
    def annualized_return(self) -> float:
        return M.annualized_return(self.returns)

    @property
    def max_drawdown(self) -> float:
        return M.max_drawdown(self.returns)

    @property
    def avg_turnover(self) -> float:
        return self.turnover.mean() if len(self.turnover) else 0.0

    def report(self) -> None:
        print("=" * 50)
        print("Portfolio Optimization Backtest Result")
        print("=" * 50)
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
        print("=" * 50)

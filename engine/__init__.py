"""Backtest engine: ladder staggered sub-portfolio backtest.

Backtests from saved alpha parquet files, decoupled from training.
Market data is read from data/a_share_data/, no database connection.
"""

from .optim_backtest import OptimBacktest, OptimBacktestResult
from .ladder_backtest import LadderBacktest, LadderBacktestResult
from .optimizer import (
    OptimContext,
    OptimizerProtocol,
    NormalizerOptimizer,
    TopNOptimizer,
    ThreeIndexTopNOptimizer,
)

__all__ = [
    "OptimBacktest",
    "OptimBacktestResult",
    "LadderBacktest",
    "LadderBacktestResult",
    "OptimizerProtocol",
    "OptimContext",
    "NormalizerOptimizer",
    "TopNOptimizer",
    "ThreeIndexTopNOptimizer",
]

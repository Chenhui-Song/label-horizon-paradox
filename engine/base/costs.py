"""Transaction cost model.

A-share cost composition:
  - Stamp duty: 0.05% (sell orders only)
  - Commission: 0.025% (both sides, some brokers include fees)
  - Slippage: not modeled. The two-segment NAV (old holdings
    previous close->VWAP + new holdings VWAP->day's close) only splits the
    day's return at the VWAP execution price; it does not capture slippage.

Costs are based on the actual execution price (not the close price). The
rebalance delta `w_t - w_{t-1}` is split into a sell leg and a buy leg:
  sell_amount = sum max(prev - new, 0)   (the part being sold)
  buy_amount  = sum max(new - prev, 0)   (the part being bought)
  turnover    = sell_amount + buy_amount = sum |new - prev|
Stamp duty is levied only on the sell leg; commission is levied on both legs:
  cost = sell_amount * stamp_duty + (sell_amount + buy_amount) * commission
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["CostModel"]


@dataclass
class CostModel:
    """A-share transaction cost model.

    Args:
        stamp_duty: stamp duty rate, sell orders only. Default 0.0005 (0.05%).
        commission: commission rate, both sides. Default 0.00025 (0.025%).
        slippage: slippage rate, reserved for reference (not modeled; the two-segment NAV only splits return at the VWAP price, slippage is not included in the rate).
    """

    stamp_duty: float = 0.0005
    commission: float = 0.00025
    slippage: float = 0.001

    def cost_from_legs(self, sell_amount: float, buy_amount: float) -> float:
        """Cost from the sell and buy legs of a rebalance.

          cost = sell_amount * stamp_duty + (sell_amount + buy_amount) * commission

        Stamp duty is levied only on sells; commission on both sides.

        Args:
            sell_amount: sum of the sold weights, max(prev - new, 0).
            buy_amount: sum of the bought weights, max(new - prev, 0).

        Returns:
            Cost in return units (deducted from NAV).
        """
        return sell_amount * self.stamp_duty + (sell_amount + buy_amount) * self.commission

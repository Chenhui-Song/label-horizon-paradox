"""Time-point model: decouples signal time from execution time.

Each strategy defines three parameters:
  - signal_time: signal generation time (factor data is cut off here), e.g. "15:00" (close)
  - exec_window: execution window (after signal_time), e.g. ("14:50","15:00") or None (all day)
  - exec_lag: lag of the execution day relative to the signal day, 0 = same-day execution,
    1 = next-day execution (default)

Lookahead red line: factor/signal data cutoff = signal_time; execution price = exec_window after signal_time.
The two do not overlap.

The daily-close model is a special case: signal_time="15:00", exec_window=None, exec_lag=1.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["TimePoint", "DailyClose"]


@dataclass
class TimePoint:
    """Time-point model: defines signal time, execution window, execution lag.

    Args:
        signal_time: signal generation time, "HH:MM" format. "15:00" = close.
        exec_window: execution window, a ("HH:MM","HH:MM") tuple; None = all day.
        exec_lag: lag of the execution day relative to the signal day, 0 = same day, 1 = next day (default).
    """

    signal_time: str = "15:00"
    exec_window: tuple[str, str] | None = None
    exec_lag: int = 1

    def __post_init__(self) -> None:
        if self.exec_lag not in (0, 1):
            raise ValueError(f"exec_lag must be 0 or 1, got {self.exec_lag}")
        _parse_hhmm(self.signal_time)
        if self.exec_window is not None:
            _parse_hhmm(self.exec_window[0])
            _parse_hhmm(self.exec_window[1])

    @property
    def is_intraday(self) -> bool:
        """Whether it is an intraday signal (signal_time earlier than close)."""
        return self.signal_time != "15:00"


class DailyClose(TimePoint):
    """Daily close (default baseline): close signal -> next-day execution."""

    def __init__(self) -> None:
        super().__init__(signal_time="15:00", exec_window=None, exec_lag=1)


def _parse_hhmm(s: str) -> tuple[int, int]:
    """Parse 'HH:MM' -> (hour, minute)."""
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Time format should be 'HH:MM', got {s!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid time {s!r}")
    return h, m

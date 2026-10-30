"""Execute one approved long barrier plan through entry and exit.

This is a single-trade historical simulation, not portfolio accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pyarrow as pa

from quant_research.backtesting.minute_entry import (
    MinuteEntryConfig,
    MinuteEntryFill,
    simulate_next_minute_entry,
)
from quant_research.backtesting.minute_exit import (
    MinuteExitConfig,
    MinuteExitFill,
    simulate_minute_exit,
)
from quant_research.strategy.barrier_trade_plan import BarrierTradePlan


@dataclass(frozen=True, slots=True)
class MinuteTrade:
    """Completed trade with entry/exit fills and net realized P&L."""

    entry: MinuteEntryFill
    exit: MinuteExitFill

    @property
    def net_pnl(self) -> float:
        return self.exit.net_pnl


def simulate_minute_trade(
    *,
    plan: BarrierTradePlan,
    bars: pa.Table,
    session_close: datetime,
    entry_config: MinuteEntryConfig | None = None,
    exit_config: MinuteExitConfig | None = None,
) -> MinuteTrade | None:
    """Resolve one long plan without reusing past minute-open prices.

    Return None if no entry is possible. Once entered, missing exit bars raise
    rather than silently treating an incomplete horizon as a timeout.
    No second position can be opened inside this single-trade lifecycle.
    """
    if session_close.tzinfo is None or session_close.utcoffset() != timedelta(0):
        raise ValueError("session_close must be UTC-aware")
    if not {"timestamp", "security_id"}.issubset(bars.column_names):
        raise ValueError("Missing minute bar columns")

    entry = simulate_next_minute_entry(
        plan=plan,
        bars=bars,
        session_close=session_close,
        config=entry_config,
    )
    if entry is None:
        return None

    # Entry fills at minute OPEN; the bar ending one minute later is eligible
    # for exit evaluation. Never include prior bars in the exit simulator.
    first_end = entry.filled_at + timedelta(minutes=1)
    deadline = entry.filled_at + timedelta(minutes=entry.max_holding_minutes)
    timestamps = bars.column("timestamp").to_pylist()
    selected: list[int] = []
    for index, stamp in enumerate(timestamps):
        if not isinstance(stamp, datetime):
            raise ValueError("Invalid minute timestamp")
        if first_end <= stamp <= deadline:
            selected.append(index)
    if not selected:
        raise ValueError("Insufficient minute bars after entry")
    # `take` preserves the original schema and provides contiguous source bars.
    future = bars.take(pa.array(selected, type=pa.int64()))
    exit_fill = simulate_minute_exit(entry=entry, bars=future, config=exit_config)
    if exit_fill.filled_at > session_close:
        raise ValueError("Exit would occur after session close")
    return MinuteTrade(entry=entry, exit=exit_fill)

"""Cash-only accounting for sequential, completed intraday long trades.

This is a small backtest ledger adapter, not the production portfolio ledger.
Only complete trades are accepted; no margin, overlapping exposure or mark-to-market.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isclose, isfinite

from quant_research.backtesting.minute_trade import MinuteTrade


@dataclass(frozen=True, slots=True)
class MinutePortfolioRecord:
    entry_at: datetime
    exit_at: datetime
    entry_cash: float
    exit_cash: float
    realized_pnl: float
    ending_cash: float


class MinutePortfolioLedger:
    """Sequential, fully funded long trades in USD; cash equals equity when flat."""

    def __init__(self, *, initial_cash: float) -> None:
        if not isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self._last_exit_at: datetime | None = None
        self._records: list[MinutePortfolioRecord] = []

    @property
    def records(self) -> tuple[MinutePortfolioRecord, ...]:
        return tuple(self._records)

    @property
    def realized_pnl(self) -> float:
        return self.cash - self.initial_cash

    def book(self, trade: MinuteTrade) -> MinutePortfolioRecord:
        """Book a finished trade atomically; reject insufficient funds or overlap."""
        entry, exit_fill = trade.entry, trade.exit
        if (
            entry.filled_at.tzinfo is None
            or exit_fill.filled_at.tzinfo is None
            or entry.filled_at.utcoffset() != timedelta(0)
            or exit_fill.filled_at.utcoffset() != timedelta(0)
            or exit_fill.filled_at < entry.filled_at
        ):
            raise ValueError("Invalid trade fill timestamps")
        if self._last_exit_at is not None and entry.filled_at <= self._last_exit_at:
            raise ValueError("Trade overlaps previous position or is out of order")
        if entry.security_id != exit_fill.security_id or entry.shares != exit_fill.shares:
            raise ValueError("Entry and exit positions differ")
        if not isinstance(entry.shares, int) or entry.shares <= 0:
            raise ValueError("Shares must be positive")
        values = (entry.fill_price, exit_fill.fill_price, entry.commission, exit_fill.commission)
        if any(not isfinite(value) for value in values):
            raise ValueError("Non-finite execution value")
        if (
            entry.fill_price <= 0
            or exit_fill.fill_price <= 0
            or entry.commission < 0
            or exit_fill.commission < 0
        ):
            raise ValueError("Invalid fill price or commission")
        debit = entry.shares * entry.fill_price + entry.commission
        credit = exit_fill.shares * exit_fill.fill_price - exit_fill.commission
        if debit > self.cash + 1e-9:
            raise ValueError("Insufficient cash for entry")
        pnl = credit - debit
        if not isclose(pnl, exit_fill.net_pnl, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("Reported trade P&L does not reconcile to fills")
        new_cash = self.cash + pnl
        record = MinutePortfolioRecord(
            entry.filled_at, exit_fill.filled_at, debit, credit, pnl, new_cash
        )
        self._records.append(record)
        self.cash = new_cash
        self._last_exit_at = exit_fill.filled_at
        return record

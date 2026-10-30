"""Daily US-equity backtest orchestrator with an injectable private policy.

V1 is a DAILY CLOSE decision / NEXT BAR execution simulation. Only bars
completed at the decision timestamp are visible to the policy. Execution
fills are simulated by DailyExecutionSimulator, then booked into the ledger.
No broker access, corporate actions, short borrow, or missing-session repair.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from quant_research.backtesting.accounting import apply_simulated_fills, mark_to_market_usd
from quant_research.backtesting.execution import (
    DailyBar,
    DailyExecutionSimulator,
    HistoricalOrder,
    SimulatedFill,
)
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode
from quant_research.trading.ledger import PortfolioLedger


@dataclass(frozen=True, slots=True)
class BacktestContext:
    """Only current completed bars, current state and past fills are visible."""

    decision_at: datetime
    bars: Mapping[str, DailyBar]
    ledger: PortfolioLedger


OrderPolicy = Callable[[BacktestContext], Sequence[HistoricalOrder]]


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    equity_usd: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    ledger: PortfolioLedger
    equity_curve: tuple[EquityPoint, ...]
    fills: tuple[SimulatedFill, ...]
    open_orders: tuple[HistoricalOrder, ...]

    @property
    def total_return(self) -> Decimal | None:
        if len(self.equity_curve) < 2 or self.equity_curve[0].equity_usd <= 0:
            return None
        return self.equity_curve[-1].equity_usd / self.equity_curve[0].equity_usd - 1

    @property
    def max_drawdown(self) -> Decimal | None:
        if not self.equity_curve:
            return None
        peak = self.equity_curve[0].equity_usd
        worst = Decimal("0")
        for point in self.equity_curve:
            peak = max(peak, point.equity_usd)
            if peak > 0:
                worst = min(worst, point.equity_usd / peak - 1)
        return worst


@dataclass(frozen=True, slots=True)
class DailyBacktestEngine:
    simulator: DailyExecutionSimulator = DailyExecutionSimulator()

    def run(
        self,
        *,
        bars: Sequence[DailyBar],
        instruments: Mapping[str, Instrument],
        initial_ledger: PortfolioLedger,
        policy: OrderPolicy,
        availability: Mapping[tuple[str, datetime], datetime] | None = None,
    ) -> BacktestResult:
        """Execute pending orders on later bars and value at each bar close.

        An order persists until filled or the dataset ends; expiry/cancellation
        rules are deliberately not invented here. Pending orders are filled
        once only; duplicates and inconsistent input are rejected.
        """
        if initial_ledger.mode is Mode.MANUAL_ALPHA:
            raise ValueError("daily automated backtest requires automated portfolio")
        if initial_ledger.positions or initial_ledger.processed_fills:
            raise ValueError("v1 requires a fresh initial ledger with no positions/fills")
        if set(initial_ledger.cash) != {Currency.USD}:
            raise ValueError("v1 requires initial USD cash only")
        if initial_ledger.cash[Currency.USD] <= 0:
            raise ValueError("initial USD equity must be positive")
        if not bars:
            raise ValueError("bars cannot be empty")

        grouped: dict[datetime, dict[str, DailyBar]] = {}
        previous: datetime | None = None
        for bar in bars:
            if previous is not None and bar.timestamp < previous:
                raise ValueError("bars must be sorted chronologically")
            previous = bar.timestamp
            if bar.security_id not in instruments:
                raise ValueError(f"missing instrument for {bar.security_id}")
            instrument = instruments[bar.security_id]
            if instrument.market is not Market.US or instrument.currency != "USD":
                raise ValueError("daily automated backtest supports US USD equities only")
            if instrument.security_id.value != bar.security_id:
                raise ValueError("instrument identifier mismatch")
            same_time = grouped.setdefault(bar.timestamp, {})
            if bar.security_id in same_time:
                raise ValueError("duplicate bar for security/timestamp")
            same_time[bar.security_id] = bar

        # Default for synthetic bars: available immediately at bar end.
        # Real stored data MUST supply explicit availability for every bar.
        if availability is not None:
            expected = {(bar.security_id, bar.timestamp) for bar in bars}
            if set(availability) != expected:
                raise ValueError("availability must contain exactly one time per bar")
            for bar in bars:
                available_at = availability[(bar.security_id, bar.timestamp)]
                offset = available_at.utcoffset()
                if offset is None:
                    raise ValueError("availability times must be UTC aware")
                if offset.total_seconds() != 0:
                    raise ValueError("availability times must be UTC")
                if available_at < bar.timestamp:
                    raise ValueError("bar cannot be available before it ends")

        if availability is not None:
            timestamps = sorted(grouped)
            for idx, timestamp in enumerate(timestamps[:-1]):
                decision_time = max(availability[(sid, timestamp)] for sid in grouped[timestamp])
                if decision_time >= timestamps[idx + 1]:
                    raise ValueError("decision time must precede next daily bar end")

        ledger = initial_ledger
        marks: dict[str, Decimal] = {}
        pending: dict[str, HistoricalOrder] = {}
        seen_order_ids: set[str] = set()
        fills: list[SimulatedFill] = []
        curve: list[EquityPoint] = []

        for timestamp, current in grouped.items():
            # Orders placed on past decisions can fill against today's bar.
            # New orders are considered only after today's execution phase.
            filled_ids: list[str] = []
            for order_id, order in pending.items():
                execution_bar = current.get(order.security_id)
                if execution_bar is None:
                    continue
                fill = self.simulator.attempt_fill(order, execution_bar)
                if fill is not None:
                    updated = apply_simulated_fills(ledger, (fill,), dict(instruments))
                    ledger = updated.ledger
                    fills.append(fill)
                    filled_ids.append(order_id)
            for order_id in filled_ids:
                del pending[order_id]

            for sid, bar in current.items():
                marks[sid] = bar.close
            equity = mark_to_market_usd(ledger, marks)
            curve.append(EquityPoint(timestamp, equity))

            decision_at = max(
                (availability[(sid, timestamp)] if availability is not None else timestamp)
                for sid in current
            )
            context = BacktestContext(decision_at, dict(current), ledger)
            new_orders = policy(context)
            for order in new_orders:
                # A later day's market should never be processed before the
                # previous decision can actually be made.
                if order.order_id in seen_order_ids:
                    raise ValueError("duplicate order ID")
                if order.security_id not in instruments:
                    raise ValueError("unknown order security")
                if order.submitted_at < decision_at:
                    raise ValueError("order submitted before its decision bar finished")
                if order.submitted_at != decision_at:
                    raise ValueError("v1 orders must be submitted at decision timestamp")
                seen_order_ids.add(order.order_id)
                pending[order.order_id] = order

        return BacktestResult(ledger, tuple(curve), tuple(fills), tuple(pending.values()))

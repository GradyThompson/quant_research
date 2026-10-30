"""Explicit, journal-backed offline order simulation.

This deliberately does not emulate exchange fills, brokerage behavior, or
strategy evaluation. A caller supplies each fill price and quantity, so the
module can test the order path without falsely claiming a realistic backtest.
Never connect it directly to a broker.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from quant_research.trading.event_journal import OrderEventJournal
from quant_research.trading.order_lifecycle import (
    OrderEvent,
    OrderEventKind,
    OrderState,
    OrderStatus,
)


@dataclass(frozen=True, slots=True)
class SimulatedOrder:
    """Immutable order identity and request; no broker-side representation."""

    order_id: str
    quantity: int


class OfflineExecutionEngine:
    """Offline lifecycle driver using the same durable reducer as future adapters.

    An external caller owns market-data timing and fills. Event IDs are fresh
    for each invocation. For replaying an identical event use the journal's
    append method with its original event ID instead of calling a helper twice.
    """

    def __init__(self, journal: OrderEventJournal) -> None:
        self._journal = journal

    @classmethod
    def open_journal(cls, path: str | Path) -> OrderEventJournal:
        """Construct durable storage; caller must close the returned journal."""
        return OrderEventJournal(path)

    def create(self, order: SimulatedOrder) -> OrderState:
        return self._journal.create_order(order.order_id, order.quantity)

    def state(self, order_id: str) -> OrderState:
        return self._journal.load_order(order_id)

    def _apply(
        self,
        order_id: str,
        kind: OrderEventKind,
        at: datetime,
        *,
        fill_quantity: int = 0,
        fill_price: Decimal | None = None,
        execution_id: str | None = None,
    ) -> OrderState:
        return self._journal.append(
            OrderEvent(
                event_id=str(uuid4()),
                order_id=order_id,
                occurred_at=at,
                kind=kind,
                fill_quantity=fill_quantity,
                fill_price=fill_price,
                broker_execution_id=execution_id,
            )
        )

    def submit(self, order_id: str, at: datetime) -> OrderState:
        return self._apply(order_id, OrderEventKind.SUBMIT_REQUESTED, at)

    def acknowledge(self, order_id: str, at: datetime) -> OrderState:
        return self._apply(order_id, OrderEventKind.ACKNOWLEDGED, at)

    def fill(
        self,
        order_id: str,
        at: datetime,
        *,
        quantity: int,
        price: Decimal,
        execution_id: str,
    ) -> OrderState:
        """Record an explicitly supplied simulated execution, not a fill forecast."""
        return self._apply(
            order_id,
            OrderEventKind.FILL,
            at,
            fill_quantity=quantity,
            fill_price=price,
            execution_id=execution_id,
        )

    def cancel_confirmed(self, order_id: str, at: datetime) -> OrderState:
        """Simulate final cancellation acknowledgement (not a request)."""
        return self._apply(order_id, OrderEventKind.CANCEL_CONFIRMED, at)

    def reject(self, order_id: str, at: datetime) -> OrderState:
        return self._apply(order_id, OrderEventKind.REJECTED, at)

    def recover_active(self) -> tuple[OrderState, ...]:
        """Replay active orders from durable local storage; no broker reconciliation."""
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
        return tuple(state for state in self._journal.orders() if state.status not in terminal)

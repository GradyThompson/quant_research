"""Pure, broker-independent order lifecycle reducer.

This module does not submit, cancel, or replace orders. Events must be durably
recorded and broker state reconciled before being used for live execution.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum


class OrderStatus(StrEnum):
    CREATED = "created"
    SUBMITTED = "submitted"
    WORKING = "working"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderEventKind(StrEnum):
    SUBMIT_REQUESTED = "submit_requested"
    ACKNOWLEDGED = "acknowledged"
    FILL = "fill"
    CANCEL_CONFIRMED = "cancel_confirmed"
    REJECTED = "rejected"


_FINAL = frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED})


def _utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    event_id: str
    order_id: str
    occurred_at: datetime
    kind: OrderEventKind
    # FILL events carry incremental executed quantity (not cumulative quantity).
    fill_quantity: int = 0
    fill_price: Decimal | None = None
    broker_execution_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.order_id:
            raise ValueError("event_id and order_id are required")
        if not _utc(self.occurred_at):
            raise ValueError("occurred_at must be UTC aware")
        if self.kind is OrderEventKind.FILL:
            if self.fill_quantity <= 0 or isinstance(self.fill_quantity, bool):
                raise ValueError("fill_quantity must be positive")
            if self.fill_price is None or not self.fill_price.is_finite() or self.fill_price <= 0:
                raise ValueError("fill_price must be positive and finite")
            if not self.broker_execution_id:
                raise ValueError("fill requires broker_execution_id for deduplication")
        elif (
            self.fill_quantity != 0
            or self.fill_price is not None
            or self.broker_execution_id is not None
        ):
            raise ValueError("only fill events can carry execution data")


@dataclass(frozen=True, slots=True)
class OrderState:
    order_id: str
    requested_quantity: int
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    executed_notional: Decimal = Decimal("0")
    last_event_at: datetime | None = None
    seen_event_ids: frozenset[str] = frozenset()
    seen_execution_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if (
            not self.order_id
            or self.requested_quantity <= 0
            or isinstance(self.requested_quantity, bool)
        ):
            raise ValueError("order_id and positive requested_quantity required")
        if not 0 <= self.filled_quantity <= self.requested_quantity:
            raise ValueError("invalid filled_quantity")
        if self.executed_notional < 0 or not self.executed_notional.is_finite():
            raise ValueError("invalid executed_notional")

    @property
    def remaining_quantity(self) -> int:
        return self.requested_quantity - self.filled_quantity

    @property
    def average_fill_price(self) -> Decimal | None:
        if self.filled_quantity == 0:
            return None
        return self.executed_notional / self.filled_quantity


def apply_order_event(state: OrderState, event: OrderEvent) -> OrderState:
    """Reduce a single event. Duplicate event/execution IDs are no-ops.

    Event ordering is strictly monotonic by timestamp; same-time events are
    accepted in caller-supplied order. A fill arriving after confirmed cancel
    or reject is rejected and must be investigated via broker reconciliation.
    """
    if state.order_id != event.order_id:
        raise ValueError("order ID mismatch")
    if event.event_id in state.seen_event_ids:
        return state
    if event.kind is OrderEventKind.FILL and event.broker_execution_id in state.seen_execution_ids:
        return state
    if state.last_event_at is not None and event.occurred_at < state.last_event_at:
        raise ValueError("out-of-order event timestamp")
    if state.status in _FINAL:
        raise ValueError("terminal order cannot transition")

    new_status: OrderStatus
    new_quantity = state.filled_quantity
    new_notional = state.executed_notional
    execution_ids = state.seen_execution_ids
    if event.kind is OrderEventKind.SUBMIT_REQUESTED:
        if state.status is not OrderStatus.CREATED:
            raise ValueError("submission already requested")
        new_status = OrderStatus.SUBMITTED
    elif event.kind is OrderEventKind.ACKNOWLEDGED:
        if state.status is not OrderStatus.SUBMITTED:
            raise ValueError("acknowledgment requires submitted status")
        new_status = OrderStatus.WORKING
    elif event.kind is OrderEventKind.FILL:
        if state.status not in (
            OrderStatus.SUBMITTED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        ):
            raise ValueError("fill requires submitted or working order")
        if event.fill_quantity > state.remaining_quantity:
            raise ValueError("fill exceeds remaining quantity")
        assert event.fill_price is not None
        assert event.broker_execution_id is not None
        new_quantity += event.fill_quantity
        new_notional += event.fill_price * event.fill_quantity
        new_status = (
            OrderStatus.FILLED
            if new_quantity == state.requested_quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        execution_ids = execution_ids | {event.broker_execution_id}
    elif event.kind is OrderEventKind.CANCEL_CONFIRMED:
        if state.status not in (
            OrderStatus.SUBMITTED,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        ):
            raise ValueError("cancellation requires submitted or working order")
        new_status = OrderStatus.CANCELLED
    elif event.kind is OrderEventKind.REJECTED:
        if state.status is not OrderStatus.SUBMITTED:
            raise ValueError("rejection requires submitted order")
        new_status = OrderStatus.REJECTED
    else:
        raise ValueError("unsupported order event")
    return replace(
        state,
        status=new_status,
        filled_quantity=new_quantity,
        executed_notional=new_notional,
        last_event_at=event.occurred_at,
        seen_event_ids=state.seen_event_ids | {event.event_id},
        seen_execution_ids=frozenset(execution_ids),
    )

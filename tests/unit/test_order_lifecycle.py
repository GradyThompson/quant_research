"""Pure lifecycle reducer tests; no broker access."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.trading.order_lifecycle import (
    OrderEvent,
    OrderEventKind,
    OrderState,
    OrderStatus,
    apply_order_event,
)

NOW = datetime(2026, 1, 2, 15, tzinfo=UTC)


def event(
    kind: OrderEventKind,
    n: int,
    *,
    qty: int = 0,
    price: str | None = None,
    execution_id: str | None = None,
) -> OrderEvent:
    return OrderEvent(
        event_id=f"e-{n}",
        order_id="o-1",
        occurred_at=NOW + timedelta(seconds=n),
        kind=kind,
        fill_quantity=qty,
        fill_price=Decimal(price) if price else None,
        broker_execution_id=execution_id,
    )


def submitted() -> OrderState:
    return apply_order_event(OrderState("o-1", 10), event(OrderEventKind.SUBMIT_REQUESTED, 1))


def test_partial_fills_and_average_price():
    state = submitted()
    state = apply_order_event(state, event(OrderEventKind.ACKNOWLEDGED, 2))
    first = event(OrderEventKind.FILL, 3, qty=4, price="10.00", execution_id="x-1")
    state = apply_order_event(state, first)
    assert state.status is OrderStatus.PARTIALLY_FILLED
    assert state.remaining_quantity == 6
    assert state.average_fill_price == Decimal("10")
    assert apply_order_event(state, first) == state
    state = apply_order_event(
        state, event(OrderEventKind.FILL, 4, qty=6, price="11.00", execution_id="x-2")
    )
    assert state.status is OrderStatus.FILLED
    assert state.average_fill_price == Decimal("10.6")
    with pytest.raises(ValueError, match="terminal"):
        apply_order_event(state, event(OrderEventKind.CANCEL_CONFIRMED, 5))


def test_execution_deduplication_with_different_event_ids():
    state = submitted()
    state = apply_order_event(
        state, event(OrderEventKind.FILL, 2, qty=2, price="10", execution_id="x-1")
    )
    duplicate = event(OrderEventKind.FILL, 3, qty=2, price="10", execution_id="x-1")
    assert apply_order_event(state, duplicate) == state


def test_cancelled_partial_order_preserves_fills():
    state = submitted()
    state = apply_order_event(
        state, event(OrderEventKind.FILL, 2, qty=3, price="20", execution_id="x-1")
    )
    state = apply_order_event(state, event(OrderEventKind.CANCEL_CONFIRMED, 3))
    assert state.status is OrderStatus.CANCELLED
    assert state.filled_quantity == 3
    assert state.remaining_quantity == 7


def test_rejection():
    state = apply_order_event(submitted(), event(OrderEventKind.REJECTED, 2))
    assert state.status is OrderStatus.REJECTED


def test_invalid_transitions():
    with pytest.raises(ValueError, match="acknowledgment"):
        apply_order_event(OrderState("o-1", 10), event(OrderEventKind.ACKNOWLEDGED, 1))
    with pytest.raises(ValueError, match="exceeds"):
        apply_order_event(
            submitted(), event(OrderEventKind.FILL, 2, qty=11, price="10", execution_id="x")
        )
    with pytest.raises(ValueError, match="out-of-order"):
        apply_order_event(submitted(), event(OrderEventKind.ACKNOWLEDGED, 0))
    with pytest.raises(ValueError, match="order ID"):
        apply_order_event(OrderState("different", 10), event(OrderEventKind.SUBMIT_REQUESTED, 1))


def test_fill_event_requires_execution_identity_and_price():
    with pytest.raises(ValueError, match="broker_execution_id"):
        event(OrderEventKind.FILL, 1, qty=1, price="10")
    with pytest.raises(ValueError, match="fill_price"):
        event(OrderEventKind.FILL, 1, qty=1, execution_id="x")


def test_utc_required():
    with pytest.raises(ValueError, match="UTC"):
        OrderEvent("e", "o-1", datetime(2026, 1, 2), OrderEventKind.SUBMIT_REQUESTED)

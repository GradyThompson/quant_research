"""Journal transaction, replay, and identity guarantees."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.trading.event_journal import OrderEventJournal
from quant_research.trading.order_lifecycle import OrderEvent, OrderEventKind, OrderStatus

START = datetime(2026, 1, 2, 15, tzinfo=UTC)


def evt(
    event_id: str, kind: OrderEventKind, seconds: int, *, qty: int = 0, execution: str | None = None
) -> OrderEvent:
    return OrderEvent(
        event_id=event_id,
        order_id="order-1",
        occurred_at=START + timedelta(seconds=seconds),
        kind=kind,
        fill_quantity=qty,
        fill_price=Decimal("12.50") if qty else None,
        broker_execution_id=execution,
    )


def test_replay_survives_restart(tmp_path):
    path = tmp_path / "orders.sqlite"
    with OrderEventJournal(path) as db:
        db.create_order("order-1", 10)
        db.append(evt("submit", OrderEventKind.SUBMIT_REQUESTED, 1))
        db.append(evt("ack", OrderEventKind.ACKNOWLEDGED, 2))
        state = db.append(evt("fill", OrderEventKind.FILL, 3, qty=4, execution="exec-1"))
        assert state.status is OrderStatus.PARTIALLY_FILLED
        assert state.filled_quantity == 4
    with OrderEventJournal(path) as db:
        state = db.load_order("order-1")
        assert state.filled_quantity == 4
        assert state.average_fill_price == Decimal("12.50")
        assert len(db.orders()) == 1
        db.append(evt("cancel", OrderEventKind.CANCEL_CONFIRMED, 4))
    with OrderEventJournal(path) as db:
        assert db.load_order("order-1").status is OrderStatus.CANCELLED


def test_duplicate_event_id_idempotent_but_conflicts_rejected(tmp_path):
    with OrderEventJournal(tmp_path / "db") as db:
        db.create_order("order-1", 10)
        first = evt("submit", OrderEventKind.SUBMIT_REQUESTED, 1)
        initial = db.append(first)
        assert db.append(first) == initial
        with pytest.raises(ValueError, match="conflicting event_id"):
            db.append(evt("submit", OrderEventKind.ACKNOWLEDGED, 2))
        assert db.load_order("order-1") == initial


def test_duplicate_execution_rejected(tmp_path):
    with OrderEventJournal(tmp_path / "db") as db:
        db.create_order("order-1", 10)
        db.append(evt("submit", OrderEventKind.SUBMIT_REQUESTED, 1))
        db.append(evt("fill", OrderEventKind.FILL, 2, qty=2, execution="exec-1"))
        with pytest.raises(ValueError, match="duplicate broker_execution_id"):
            db.append(evt("fill-again", OrderEventKind.FILL, 3, qty=2, execution="exec-1"))
        assert db.load_order("order-1").filled_quantity == 2


def test_invalid_transition_does_not_persist(tmp_path):
    path = tmp_path / "db"
    with OrderEventJournal(path) as db:
        db.create_order("order-1", 10)
        with pytest.raises(ValueError, match="acknowledgment"):
            db.append(evt("ack", OrderEventKind.ACKNOWLEDGED, 1))
        assert db.load_order("order-1").status is OrderStatus.CREATED
    with OrderEventJournal(path) as db:
        assert db.load_order("order-1").seen_event_ids == frozenset()


def test_conflicting_order_and_missing_order(tmp_path):
    with OrderEventJournal(tmp_path / "db") as db:
        db.create_order("order-1", 10)
        assert db.create_order("order-1", 10).requested_quantity == 10
        with pytest.raises(ValueError, match="conflicting order"):
            db.create_order("order-1", 12)
        with pytest.raises(KeyError, match="unknown order"):
            db.load_order("missing")

"""Offline engine integration tests using a real SQLite journal."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.trading.event_journal import OrderEventJournal
from quant_research.trading.order_lifecycle import OrderStatus
from quant_research.trading.simulation import OfflineExecutionEngine, SimulatedOrder

T = datetime(2025, 2, 3, 15, tzinfo=UTC)


def test_partial_fill_restart_and_completion(tmp_path):
    path = tmp_path / "orders.sqlite"
    with OrderEventJournal(path) as journal:
        engine = OfflineExecutionEngine(journal)
        engine.create(SimulatedOrder("o1", 100))
        engine.submit("o1", T)
        engine.acknowledge("o1", T + timedelta(seconds=1))
        partial = engine.fill(
            "o1",
            T + timedelta(seconds=2),
            quantity=40,
            price=Decimal("25.10"),
            execution_id="sim-fill-1",
        )
        assert partial.status is OrderStatus.PARTIALLY_FILLED
        assert partial.remaining_quantity == 60
    with OrderEventJournal(path) as journal:
        engine = OfflineExecutionEngine(journal)
        assert [state.order_id for state in engine.recover_active()] == ["o1"]
        done = engine.fill(
            "o1",
            T + timedelta(seconds=3),
            quantity=60,
            price=Decimal("25.20"),
            execution_id="sim-fill-2",
        )
        assert done.status is OrderStatus.FILLED
        assert done.average_fill_price == Decimal("25.16")
        assert engine.recover_active() == ()


def test_cancelled_order_does_not_accept_fills(tmp_path):
    with OrderEventJournal(tmp_path / "test.sqlite") as journal:
        engine = OfflineExecutionEngine(journal)
        engine.create(SimulatedOrder("cancel", 10))
        engine.submit("cancel", T)
        engine.cancel_confirmed("cancel", T + timedelta(seconds=1))
        with pytest.raises(ValueError, match="terminal"):
            engine.fill(
                "cancel",
                T + timedelta(seconds=2),
                quantity=1,
                price=Decimal("10"),
                execution_id="late-fill",
            )
        assert engine.state("cancel").status is OrderStatus.CANCELLED


def test_duplicate_execution_is_rejected(tmp_path):
    with OrderEventJournal(tmp_path / "test.sqlite") as journal:
        engine = OfflineExecutionEngine(journal)
        engine.create(SimulatedOrder("one", 10))
        engine.submit("one", T)
        engine.fill(
            "one", T + timedelta(seconds=1), quantity=3, price=Decimal("5"), execution_id="same"
        )
        with pytest.raises(ValueError, match="duplicate broker_execution_id"):
            engine.fill(
                "one", T + timedelta(seconds=2), quantity=3, price=Decimal("5"), execution_id="same"
            )
        assert engine.state("one").filled_quantity == 3


def test_rejected_order_is_terminal(tmp_path):
    with OrderEventJournal(tmp_path / "test.sqlite") as journal:
        engine = OfflineExecutionEngine(journal)
        engine.create(SimulatedOrder("reject", 2))
        engine.submit("reject", T)
        assert engine.reject("reject", T + timedelta(seconds=1)).status is OrderStatus.REJECTED
        assert engine.recover_active() == ()


def test_invalid_quantity_is_rejected(tmp_path):
    with OrderEventJournal(tmp_path / "test.sqlite") as journal:
        engine = OfflineExecutionEngine(journal)
        with pytest.raises(ValueError, match="positive"):
            engine.create(SimulatedOrder("bad", -1))

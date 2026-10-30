from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.backtesting import (
    DailyBar,
    DailyExecutionSimulator,
    HistoricalOrder,
    OrderType,
    Side,
)

T = datetime(2025, 1, 3, 21, tzinfo=UTC)


def bar(**overrides: object) -> DailyBar:
    values = {
        "security_id": "SEC-000001",
        "timestamp": T,
        "open": Decimal("100"),
        "high": Decimal("105"),
        "low": Decimal("95"),
        "close": Decimal("102"),
        "volume": 10000,
    }
    values.update(overrides)
    return DailyBar(**values)  # type: ignore[arg-type]


def order(**overrides: object) -> HistoricalOrder:
    values = {
        "order_id": "order-1",
        "security_id": "SEC-000001",
        "side": Side.BUY,
        "kind": OrderType.MARKET,
        "quantity": 10,
        "submitted_at": T - timedelta(days=1),
    }
    values.update(overrides)
    return HistoricalOrder(**values)  # type: ignore[arg-type]


def test_market_order_fills_at_next_open_with_costs():
    fill = DailyExecutionSimulator().attempt_fill(order(), bar())
    assert fill is not None
    assert fill.price == Decimal("100.02")
    assert fill.commission == Decimal("0.050")
    assert fill.quantity == 10


def test_no_same_bar_execution_even_if_limit_touched():
    simulator = DailyExecutionSimulator()
    assert simulator.attempt_fill(order(submitted_at=T), bar()) is None
    assert simulator.attempt_fill(order(submitted_at=T + timedelta(seconds=1)), bar()) is None


def test_buy_limit_not_touched():
    proposed = order(kind=OrderType.LIMIT, limit_price=Decimal("94"))
    assert DailyExecutionSimulator().attempt_fill(proposed, bar()) is None


def test_buy_limit_slippage_cannot_violate_limit():
    proposed = order(kind=OrderType.LIMIT, limit_price=Decimal("97"))
    assert DailyExecutionSimulator().attempt_fill(proposed, bar()) is None
    fill = DailyExecutionSimulator(slippage_bps=Decimal(0)).attempt_fill(proposed, bar())
    assert fill is not None and fill.price == Decimal("97")


def test_sell_limit_gap_open_price_improvement():
    proposed = order(side=Side.SELL, kind=OrderType.LIMIT, limit_price=Decimal("98"))
    fill = DailyExecutionSimulator(slippage_bps=Decimal(0)).attempt_fill(proposed, bar())
    assert fill is not None and fill.price == Decimal("100")


def test_sell_limit_not_touched():
    proposed = order(side=Side.SELL, kind=OrderType.LIMIT, limit_price=Decimal("110"))
    assert DailyExecutionSimulator().attempt_fill(proposed, bar()) is None


def test_volume_zero_prevents_fill():
    assert DailyExecutionSimulator().attempt_fill(order(), bar(volume=0)) is None


def test_wrong_security_prevents_fill():
    assert DailyExecutionSimulator().attempt_fill(order(), bar(security_id="SEC-000002")) is None


def test_validation():
    with pytest.raises(ValueError):
        order(quantity=0)
    with pytest.raises(ValueError):
        order(kind=OrderType.LIMIT)
    with pytest.raises(ValueError):
        bar(low=Decimal("101"))
    with pytest.raises(ValueError):
        DailyExecutionSimulator(slippage_bps=Decimal("-1"))

"""Synthetic conservative minute exit tests."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.backtesting.minute_entry import MinuteEntryFill
from quant_research.backtesting.minute_exit import (
    MinuteExitConfig,
    MinuteExitReason,
    simulate_minute_exit,
)
from quant_research.core import SecurityId

START = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)


def _entry(horizon: int = 2) -> MinuteEntryFill:
    return MinuteEntryFill(
        security_id=SecurityId("SEC-000001"),
        shares=10,
        signal_at=START - timedelta(minutes=2),
        filled_at=START,
        fill_price=100.0,
        commission=0.05,
        take_profit=100.5,
        stop_loss=99.6,
        max_holding_minutes=horizon,
    )


def _bars(*values: tuple[float, float, float, float]) -> pa.Table:
    return pa.table(
        {
            "security_id": ["SEC-000001"] * len(values),
            "timestamp": [START + timedelta(minutes=i + 1) for i in range(len(values))],
            "open": [v[0] for v in values],
            "high": [v[1] for v in values],
            "low": [v[2] for v in values],
            "close": [v[3] for v in values],
        }
    )


@pytest.mark.parametrize(
    ("bar", "reason", "price", "exit_at"),
    [
        (
            (100, 100.6, 99.9, 100.2),
            MinuteExitReason.TAKE_PROFIT,
            100.5,
            START + timedelta(minutes=1),
        ),
        ((100, 100.1, 99.5, 99.8), MinuteExitReason.STOP_LOSS, 99.6, START + timedelta(minutes=1)),
        (
            (100, 100.6, 99.5, 100),
            MinuteExitReason.AMBIGUOUS_STOP,
            99.6,
            START + timedelta(minutes=1),
        ),
        ((99.0, 99.9, 98.9, 99.5), MinuteExitReason.STOP_LOSS, 99.0, START),
        ((101, 101.1, 100.9, 101.0), MinuteExitReason.TAKE_PROFIT, 100.5, START),
    ],
)
def test_exit_types(
    bar: tuple[float, float, float, float],
    reason: MinuteExitReason,
    price: float,
    exit_at: datetime,
) -> None:
    fill = simulate_minute_exit(entry=_entry(), bars=_bars(bar), config=MinuteExitConfig(0, 0))
    assert fill.reason is reason
    assert fill.fill_price == pytest.approx(price)
    assert fill.filled_at == exit_at


def test_timeout_and_net_costs() -> None:
    fill = simulate_minute_exit(
        entry=_entry(),
        bars=_bars((100, 100.2, 99.9, 100.1), (100.1, 100.3, 99.9, 100.2)),
        config=MinuteExitConfig(slippage_bps=0, commission_per_share=0.005),
    )
    assert fill.reason is MinuteExitReason.TIMEOUT
    assert fill.filled_at == START + timedelta(minutes=2)
    assert fill.net_pnl == pytest.approx(1.9)


def test_missing_minute_is_not_timeout() -> None:
    bars = _bars((100, 100.1, 99.9, 100.0)).slice(0, 0)
    with pytest.raises(ValueError, match="Insufficient"):
        simulate_minute_exit(entry=_entry(), bars=bars)


def test_invalid_bar_is_rejected() -> None:
    with pytest.raises(ValueError, match="Inconsistent"):
        simulate_minute_exit(entry=_entry(), bars=_bars((100, 99, 98, 100)))


def test_slippage_on_target() -> None:
    fill = simulate_minute_exit(
        entry=_entry(),
        bars=_bars((100, 100.6, 99.9, 100.2)),
        config=MinuteExitConfig(slippage_bps=2, commission_per_share=0.005),
    )
    assert fill.fill_price == pytest.approx(100.5 * (1 - 2 / 10000))
    assert fill.net_pnl == pytest.approx((fill.fill_price - 100) * 10 - 0.1)

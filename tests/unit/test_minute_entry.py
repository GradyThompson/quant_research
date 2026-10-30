"""Synthetic next-minute entry timing and price tests."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.backtesting.minute_entry import (
    MinuteEntryConfig,
    simulate_next_minute_entry,
)
from quant_research.core import SecurityId
from quant_research.strategy.barrier_trade_plan import BarrierTradePlan

BASE = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)
SID = SecurityId("SEC-000001")


def _plan(signal_offset_seconds: int = 0, horizon: int = 2) -> BarrierTradePlan:
    return BarrierTradePlan(
        SID, BASE + timedelta(seconds=signal_offset_seconds), 10, 0.005, 0.004, horizon
    )


def _bars() -> pa.Table:
    return pa.table(
        {
            "security_id": [str(SID)] * 5,
            "timestamp": [BASE + timedelta(minutes=i) for i in range(1, 6)],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )


def test_entry_requires_strictly_later_bar_open() -> None:
    result = simulate_next_minute_entry(
        plan=_plan(),
        bars=_bars(),
        session_close=BASE + timedelta(minutes=6),
        config=MinuteEntryConfig(slippage_bps=10, commission_per_share=0.01),
    )
    assert result is not None
    assert result.filled_at == BASE + timedelta(minutes=1)
    assert result.fill_price == pytest.approx(101.101)
    assert result.take_profit == pytest.approx(result.fill_price * 1.005)
    assert result.stop_loss == pytest.approx(result.fill_price * 0.996)
    assert result.commission == pytest.approx(0.1)


def test_signal_during_bar_cannot_fill_its_open() -> None:
    result = simulate_next_minute_entry(
        plan=_plan(signal_offset_seconds=30),
        bars=_bars(),
        session_close=BASE + timedelta(minutes=6),
    )
    assert result is not None
    assert result.filled_at == BASE + timedelta(minutes=1)


def test_horizon_beyond_close_skips() -> None:
    result = simulate_next_minute_entry(
        plan=_plan(horizon=6),
        bars=_bars(),
        session_close=BASE + timedelta(minutes=6),
    )
    assert result is None


def test_wrong_security_rejected() -> None:
    changed = _bars().set_column(0, "security_id", pa.array(["SEC-000002"] * 5))
    with pytest.raises(ValueError, match="security"):
        simulate_next_minute_entry(
            plan=_plan(),
            bars=changed,
            session_close=BASE + timedelta(minutes=6),
        )


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValueError, match="costs"):
        MinuteEntryConfig(slippage_bps=-1)

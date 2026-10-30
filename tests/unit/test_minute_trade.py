"""End-to-end single-position entry and exit using synthetic minute bars."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.backtesting.minute_entry import MinuteEntryConfig
from quant_research.backtesting.minute_exit import MinuteExitConfig, MinuteExitReason
from quant_research.backtesting.minute_trade import simulate_minute_trade
from quant_research.core import SecurityId
from quant_research.strategy.barrier_trade_plan import BarrierTradePlan

START = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)


def _plan(*, signal_at: datetime | None = None) -> BarrierTradePlan:
    return BarrierTradePlan(SecurityId("SEC-000001"), signal_at or START, 10, 0.005, 0.004, 3)


def _bars(*, profit: bool = True, missing: bool = False) -> pa.Table:
    rows = []
    for i in range(1, 7):
        if missing and i == 3:
            continue
        rows.append(
            {
                "security_id": "SEC-000001",
                "timestamp": START + timedelta(minutes=i),
                "open": 100.0,
                "high": 101.0 if i == 3 and profit else 100.1,
                "low": 99.0 if i == 3 and not profit else 99.9,
                "close": 100.0,
            }
        )
    return pa.Table.from_pylist(rows)


def _trade(bars: pa.Table, *, plan: BarrierTradePlan | None = None):
    return simulate_minute_trade(
        plan=plan or _plan(),
        bars=bars,
        session_close=START + timedelta(minutes=8),
        entry_config=MinuteEntryConfig(slippage_bps=0, commission_per_share=0),
        exit_config=MinuteExitConfig(slippage_bps=0, commission_per_share=0),
    )


def test_target_fill_and_pnl() -> None:
    trade = _trade(_bars())
    assert trade is not None
    assert trade.entry.filled_at == START + timedelta(minutes=1)
    assert trade.exit.reason is MinuteExitReason.TAKE_PROFIT
    assert trade.exit.net_pnl == pytest.approx(5.0)


def test_stop_fill_and_pnl() -> None:
    trade = _trade(_bars(profit=False))
    assert trade is not None
    assert trade.exit.reason is MinuteExitReason.STOP_LOSS
    assert trade.net_pnl == pytest.approx(-4.0)


def test_no_future_eligible_entry_returns_none() -> None:
    plan = _plan(signal_at=START + timedelta(minutes=7))
    assert _trade(_bars(), plan=plan) is None


def test_incomplete_exit_window_fails() -> None:
    with pytest.raises(ValueError, match="Missing contiguous"):
        _trade(_bars(missing=True))


def test_fees_flow_through_to_net_pnl() -> None:
    trade = simulate_minute_trade(
        plan=_plan(),
        bars=_bars(),
        session_close=START + timedelta(minutes=8),
        entry_config=MinuteEntryConfig(slippage_bps=0, commission_per_share=0.01),
        exit_config=MinuteExitConfig(slippage_bps=0, commission_per_share=0.01),
    )
    assert trade is not None
    assert trade.net_pnl == pytest.approx(4.8)

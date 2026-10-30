"""Offline end-to-end daily backtest tests; no proprietary signal required."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.backtesting.engine import DailyBacktestEngine
from quant_research.backtesting.execution import DailyBar, HistoricalOrder, OrderType, Side
from quant_research.core.identifiers import SecurityId
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId
from quant_research.trading.ledger import PortfolioLedger

START = datetime(2025, 1, 2, 21, tzinfo=UTC)
SID = "SEC-000001"


def fixtures():
    bars = [
        DailyBar(
            SID, START + timedelta(days=i), Decimal(p), Decimal(p), Decimal(p), Decimal(p), 1000
        )
        for i, p in enumerate((100, 110, 120))
    ]
    instruments = {SID: Instrument(SecurityId(SID), Market.US, "USD")}
    ledger = PortfolioLedger(
        portfolio_id=PortfolioId("test"),
        mode=Mode.AUTOMATED_ALPHA,
        cash={Currency.USD: Decimal("10000")},
    )
    return bars, instruments, ledger


def test_order_on_close_fills_only_next_bar():
    bars, instruments, ledger = fixtures()

    def policy(context):
        if context.decision_at == START:
            return [HistoricalOrder("entry", SID, Side.BUY, OrderType.MARKET, 10, START)]
        return []

    result = DailyBacktestEngine().run(
        bars=bars, instruments=instruments, initial_ledger=ledger, policy=policy
    )
    assert len(result.fills) == 1
    assert result.fills[0].bar_timestamp == START + timedelta(days=1)
    assert result.ledger.positions[SID].quantity == 10
    assert result.equity_curve[0].equity_usd == Decimal("10000")
    assert result.equity_curve[-1].equity_usd > Decimal("10000")
    assert result.total_return is not None and result.total_return > 0
    assert Decimal("-0.0001") <= result.max_drawdown <= Decimal("0")


def test_unfilled_limit_remains_pending():
    bars, instruments, ledger = fixtures()

    def policy(context):
        if context.decision_at == START:
            return [
                HistoricalOrder("entry", SID, Side.BUY, OrderType.LIMIT, 10, START, Decimal("50"))
            ]
        return []

    result = DailyBacktestEngine().run(
        bars=bars, instruments=instruments, initial_ledger=ledger, policy=policy
    )
    assert not result.fills
    assert len(result.open_orders) == 1
    assert result.equity_curve[-1].equity_usd == Decimal("10000")


def test_duplicate_order_id_rejected():
    bars, instruments, ledger = fixtures()

    def policy(context):
        return [HistoricalOrder("dup", SID, Side.BUY, OrderType.MARKET, 1, context.decision_at)]

    with pytest.raises(ValueError, match="duplicate order"):
        DailyBacktestEngine().run(
            bars=bars, instruments=instruments, initial_ledger=ledger, policy=policy
        )


def test_future_order_submission_rejected():
    bars, instruments, ledger = fixtures()

    def policy(context):
        if context.decision_at == START:
            return [
                HistoricalOrder(
                    "future", SID, Side.BUY, OrderType.MARKET, 1, START + timedelta(minutes=1)
                )
            ]
        return []

    with pytest.raises(ValueError, match="decision timestamp"):
        DailyBacktestEngine().run(
            bars=bars, instruments=instruments, initial_ledger=ledger, policy=policy
        )


def test_missing_instrument_rejected():
    bars, _, ledger = fixtures()
    with pytest.raises(ValueError, match="missing instrument"):
        DailyBacktestEngine().run(
            bars=bars, instruments={}, initial_ledger=ledger, policy=lambda _: []
        )

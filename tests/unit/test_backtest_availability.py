"""A close-only policy must not access vendor data before publication."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.backtesting.engine import DailyBacktestEngine
from quant_research.backtesting.execution import DailyBar, HistoricalOrder, OrderType, Side
from quant_research.core import SecurityId
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId
from quant_research.trading.ledger import PortfolioLedger

START = datetime(2025, 1, 2, 21, tzinfo=UTC)
SID = "SEC-000001"


def setup():
    bars = [
        DailyBar(
            SID,
            START + timedelta(days=index),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            Decimal("100"),
            100,
        )
        for index in range(3)
    ]
    instruments = {SID: Instrument(SecurityId(SID), Market.US, "USD")}
    ledger = PortfolioLedger(
        PortfolioId("backtest"), Mode.AUTOMATED_ALPHA, cash={Currency.USD: Decimal("10000")}
    )
    return bars, instruments, ledger


def test_orders_created_after_data_available_and_fill_later():
    bars, instruments, ledger = setup()
    availability = {(SID, bar.timestamp): bar.timestamp + timedelta(minutes=15) for bar in bars}
    observed = []

    def policy(context):
        observed.append(context.decision_at)
        if len(observed) == 1:
            return [
                HistoricalOrder("entry", SID, Side.BUY, OrderType.MARKET, 1, context.decision_at)
            ]
        return []

    result = DailyBacktestEngine().run(
        bars=bars,
        instruments=instruments,
        initial_ledger=ledger,
        policy=policy,
        availability=availability,
    )
    assert observed[0] == START + timedelta(minutes=15)
    assert result.fills[0].bar_timestamp == START + timedelta(days=1)


def test_missing_availability_is_error():
    bars, instruments, ledger = setup()
    with pytest.raises(ValueError, match="availability must contain"):
        DailyBacktestEngine().run(
            bars=bars,
            instruments=instruments,
            initial_ledger=ledger,
            policy=lambda _: [],
            availability={},
        )

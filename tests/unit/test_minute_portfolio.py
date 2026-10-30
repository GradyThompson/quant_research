"""Cash reconciliation tests for completed minute trades."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from quant_research.backtesting.minute_entry import MinuteEntryFill
from quant_research.backtesting.minute_exit import MinuteExitFill, MinuteExitReason
from quant_research.backtesting.minute_portfolio import MinutePortfolioLedger
from quant_research.backtesting.minute_trade import MinuteTrade
from quant_research.core import SecurityId

START = datetime(2025, 1, 13, 15, 0, tzinfo=UTC)
SID = SecurityId("SEC-000001")


def trade(
    *, at: datetime = START, shares: int = 10, buy: float = 100.0, sell: float = 101.0
) -> MinuteTrade:
    entry = MinuteEntryFill(SID, shares, at - timedelta(minutes=1), at, buy, 0.05, 100.5, 99.6, 30)
    gross = shares * (sell - buy)
    exit_fill = MinuteExitFill(
        SID,
        shares,
        at + timedelta(minutes=5),
        sell,
        0.05,
        MinuteExitReason.TIMEOUT,
        gross,
        gross - 0.10,
    )
    return MinuteTrade(entry, exit_fill)


def test_cash_and_pnl_reconcile() -> None:
    ledger = MinutePortfolioLedger(initial_cash=2000)
    first = ledger.book(trade())
    assert first.entry_cash == pytest.approx(1000.05)
    assert first.exit_cash == pytest.approx(1009.95)
    assert first.realized_pnl == pytest.approx(9.90)
    assert ledger.cash == pytest.approx(2009.90)
    ledger.book(trade(at=START + timedelta(minutes=10), sell=99))
    assert ledger.realized_pnl == pytest.approx(-0.20)
    assert len(ledger.records) == 2


def test_insufficient_cash_does_not_change_state() -> None:
    ledger = MinutePortfolioLedger(initial_cash=1000)
    with pytest.raises(ValueError, match="Insufficient cash"):
        ledger.book(trade())
    assert ledger.cash == 1000
    assert ledger.records == ()


def test_rejects_overlapping_trades() -> None:
    ledger = MinutePortfolioLedger(initial_cash=2000)
    ledger.book(trade())
    with pytest.raises(ValueError, match="overlaps"):
        ledger.book(trade(at=START + timedelta(minutes=3)))
    assert len(ledger.records) == 1


def test_rejects_inconsistent_pnl() -> None:
    ledger = MinutePortfolioLedger(initial_cash=2000)
    original = trade()
    bad = MinuteTrade(original.entry, replace(original.exit, net_pnl=123))
    with pytest.raises(ValueError, match="reconcile"):
        ledger.book(bad)
    assert ledger.records == ()


def test_rejects_security_mismatch() -> None:
    ledger = MinutePortfolioLedger(initial_cash=2000)
    original = trade()
    bad = MinuteTrade(original.entry, replace(original.exit, security_id=SecurityId("SEC-000002")))
    with pytest.raises(ValueError, match="differ"):
        ledger.book(bad)


@pytest.mark.parametrize("initial", [0, -1, float("nan"), float("inf")])
def test_invalid_initial_cash(initial: float) -> None:
    with pytest.raises(ValueError, match="initial_cash"):
        MinutePortfolioLedger(initial_cash=initial)

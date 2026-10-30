"""Tests for simulated fill -> immutable portfolio ledger integration."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.backtesting.accounting import apply_simulated_fills, mark_to_market_usd
from quant_research.backtesting.execution import Side, SimulatedFill
from quant_research.core import SecurityId
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId
from quant_research.trading.ledger import PortfolioLedger

NOW = datetime(2025, 1, 3, 21, tzinfo=UTC)
SID = "SEC-000001"
INSTRUMENT = Instrument(SecurityId(SID), Market.US, "USD")


def make_ledger() -> PortfolioLedger:
    return PortfolioLedger(
        portfolio_id=PortfolioId("backtest"),
        mode=Mode.AUTOMATED_ALPHA,
        cash={Currency.USD: Decimal("10000")},
    )


def fill(order_id: str, side: Side, quantity: int, price: str, days: int = 0) -> SimulatedFill:
    return SimulatedFill(
        order_id=order_id,
        security_id=SID,
        side=side,
        quantity=quantity,
        price=Decimal(price),
        commission=Decimal("1"),
        bar_timestamp=NOW + timedelta(days=days),
    )


def test_buys_and_sells_post_to_cash_and_positions():
    result = apply_simulated_fills(
        make_ledger(),
        (fill("buy", Side.BUY, 10, "100"), fill("sell", Side.SELL, 4, "110", 1)),
        {SID: INSTRUMENT},
    )
    assert result.ledger.positions[SID].quantity == 6
    assert result.ledger.positions[SID].realized_pnl == Decimal("40")
    assert result.ledger.cash[Currency.USD] == Decimal("9438")
    assert result.ledger.commissions(Currency.USD) == Decimal("2")
    assert mark_to_market_usd(result.ledger, {SID: Decimal("110")}) == Decimal("10098")
    assert len(result.applied) == 2


def test_same_fill_replay_is_idempotent():
    first = apply_simulated_fills(make_ledger(), (fill("a", Side.BUY, 5, "10"),), {SID: INSTRUMENT})
    repeated = apply_simulated_fills(
        first.ledger, (fill("a", Side.BUY, 5, "10"),), {SID: INSTRUMENT}
    )
    assert repeated.ledger is first.ledger
    assert repeated.applied == ()


def test_order_id_reuse_with_different_fill_rejected():
    first = apply_simulated_fills(make_ledger(), (fill("a", Side.BUY, 5, "10"),), {SID: INSTRUMENT})
    with pytest.raises(ValueError, match="conflicting execution ID"):
        apply_simulated_fills(first.ledger, (fill("a", Side.BUY, 6, "10"),), {SID: INSTRUMENT})


def test_missing_instrument_and_mismatch():
    with pytest.raises(ValueError, match="missing instrument"):
        apply_simulated_fills(make_ledger(), (fill("a", Side.BUY, 5, "10"),), {})
    other = Instrument(SecurityId("SEC-000002"), Market.US, "USD")
    with pytest.raises(ValueError, match="does not match"):
        apply_simulated_fills(make_ledger(), (fill("a", Side.BUY, 5, "10"),), {SID: other})


def test_out_of_order_fills_rejected():
    with pytest.raises(ValueError, match="out-of-order"):
        apply_simulated_fills(
            make_ledger(),
            (fill("later", Side.BUY, 5, "10", 1), fill("earlier", Side.BUY, 5, "10")),
            {SID: INSTRUMENT},
        )

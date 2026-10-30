"""Fill-driven multi-currency portfolio accounting tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.core import SecurityId
from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId
from quant_research.trading.ledger import LedgerFill, PortfolioLedger

AT = datetime(2026, 10, 30, 16, tzinfo=UTC)
US = Instrument(SecurityId("SEC-000001"), Market.US, "USD")
CA = Instrument(SecurityId("SEC-000002"), Market.CA, "CAD")


def ledger(mode: Mode = Mode.AUTOMATED_ALPHA) -> PortfolioLedger:
    return PortfolioLedger(PortfolioId("alpha"), mode, cash={Currency.USD: Decimal("10000")})


def fill(
    n: int, qty: int, price: str, *, instrument: Instrument = US, fee: str = "1"
) -> LedgerFill:
    return LedgerFill(
        f"ex-{n}",
        f"order-{n}",
        instrument,
        qty,
        Decimal(price),
        Decimal(fee),
        AT + timedelta(seconds=n),
    )


def test_buy_sell_and_realized_pnl():
    state = ledger().apply(fill(1, 10, "100")).apply(fill(2, -4, "110"))
    assert state.cash[Currency.USD] == Decimal("9438")
    position = state.positions[US.security_id.value]
    assert position.quantity == 6
    assert position.average_entry_price == Decimal("100")
    assert position.realized_pnl == Decimal("40")
    assert position.unrealized_pnl(Decimal("120")) == Decimal("120")
    assert state.equity_at_marks({US.security_id.value: Decimal("120")}, Currency.USD) == Decimal(
        "10158"
    )
    assert state.commissions(Currency.USD) == Decimal("2")


def test_short_cover_flip_and_weighted_average():
    state = ledger().apply(fill(1, -10, "50")).apply(fill(2, 4, "40"))
    position = state.positions[US.security_id.value]
    assert position.quantity == -6
    assert position.realized_pnl == Decimal("40")
    assert position.average_entry_price == Decimal("50")
    state = state.apply(fill(3, 8, "30"))
    position = state.positions[US.security_id.value]
    assert position.quantity == 2
    assert position.average_entry_price == Decimal("30")
    assert position.realized_pnl == Decimal("160")


def test_scale_into_position_and_close():
    state = ledger().apply(fill(1, 10, "10")).apply(fill(2, 10, "20"))
    assert state.positions[US.security_id.value].average_entry_price == Decimal("15")
    state = state.apply(fill(3, -20, "18"))
    assert state.positions[US.security_id.value].quantity == 0
    assert state.positions[US.security_id.value].realized_pnl == Decimal("60")


def test_fill_idempotent_but_conflicts_rejected():
    state = ledger().apply(fill(1, 10, "10"))
    assert state.apply(fill(1, 10, "10")) is state
    with pytest.raises(ValueError, match="conflicting execution"):
        state.apply(fill(1, 11, "10"))


def test_no_implicit_fx_and_mode_restriction():
    with pytest.raises(ValueError, match="USD"):
        ledger().apply(fill(1, 5, "10", instrument=CA))
    state = ledger(Mode.MANUAL_ALPHA).apply(fill(1, 5, "10", instrument=CA))
    assert state.cash[Currency.CAD] == Decimal("-51")
    assert state.cash[Currency.USD] == Decimal("10000")
    with pytest.raises(ValueError, match="missing mark"):
        state.equity_at_marks({}, Currency.CAD)
    assert state.equity_at_marks({CA.security_id.value: Decimal("12")}, Currency.CAD) == Decimal(
        "9"
    )


def test_invalid_fills_and_ordering():
    with pytest.raises(ValueError, match="quantity"):
        fill(1, 0, "10")
    with pytest.raises(ValueError, match="price"):
        fill(1, 1, "0")
    state = ledger().apply(fill(2, 1, "10"))
    with pytest.raises(ValueError, match="out-of-order"):
        state.apply(fill(1, 1, "10"))

"""Coverage of immutable portfolio capital and exposure contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from quant_research.core import SecurityId
from quant_research.trading.capital import (
    CapitalLimit,
    CashBalance,
    Currency,
    MarkedPosition,
    PortfolioCapitalSnapshot,
)
from quant_research.trading.contracts import Instrument, Market, Mode, PortfolioId

NOW = datetime(2026, 10, 30, 16, tzinfo=UTC)


def stock(number: int, market: Market, currency: str) -> Instrument:
    return Instrument(SecurityId(f"SEC-{number:06d}"), market, currency)


def snapshot(*, mode: Mode = Mode.MANUAL_ALPHA) -> PortfolioCapitalSnapshot:
    return PortfolioCapitalSnapshot(
        portfolio_id=PortfolioId("main"),
        mode=mode,
        as_of=NOW,
        base_currency=Currency.USD,
        cash=(CashBalance(Currency.USD, Decimal("10000"), Decimal("1200")),),
        limits=(CapitalLimit(Currency.USD, Decimal("4500")),),
    )


def test_available_capital_is_explicit_not_inferred():
    result = snapshot()
    assert result.cash[0].unreserved_settled == Decimal("8800")
    assert result.available_to_allocate(Currency.USD) == Decimal("4500")
    assert result.available_to_allocate(Currency.CAD) == Decimal("0")


def test_long_and_short_exposures_keep_sign():
    result = PortfolioCapitalSnapshot(
        portfolio_id=PortfolioId("alpha"),
        mode=Mode.AUTOMATED_ALPHA,
        as_of=NOW,
        base_currency=Currency.USD,
        cash=(CashBalance(Currency.USD, Decimal("1000")),),
        positions=(
            MarkedPosition(stock(1, Market.US, "USD"), 10, Decimal("100"), NOW),
            MarkedPosition(stock(2, Market.US, "USD"), -5, Decimal("40"), NOW),
        ),
    )
    assert result.exposures_by_currency()[Currency.USD] == (Decimal("800"), Decimal("1200"))


def test_manual_portfolio_may_include_cad_without_implicit_fx():
    result = PortfolioCapitalSnapshot(
        portfolio_id=PortfolioId("manual"),
        mode=Mode.MANUAL_ALPHA,
        as_of=NOW,
        base_currency=Currency.CAD,
        cash=(CashBalance(Currency.CAD, Decimal("4000")),),
        positions=(MarkedPosition(stock(3, Market.CA, "CAD"), 10, Decimal("50"), NOW),),
    )
    assert result.exposures_by_currency()[Currency.CAD] == (Decimal("500"), Decimal("500"))


def test_automated_mode_rejects_cad():
    with pytest.raises(ValueError, match=r"USD.*only"):
        PortfolioCapitalSnapshot(
            portfolio_id=PortfolioId("auto"),
            mode=Mode.SHORT_TERM,
            as_of=NOW,
            base_currency=Currency.CAD,
            cash=(CashBalance(Currency.CAD, Decimal("1")),),
        )


def test_duplicate_cash_rejected():
    with pytest.raises(ValueError, match="duplicate cash"):
        PortfolioCapitalSnapshot(
            portfolio_id=PortfolioId("test"),
            mode=Mode.MANUAL_ALPHA,
            as_of=NOW,
            base_currency=Currency.USD,
            cash=(CashBalance(Currency.USD, Decimal("1")), CashBalance(Currency.USD, Decimal("2"))),
        )


def test_future_mark_rejected():
    with pytest.raises(ValueError, match="future"):
        PortfolioCapitalSnapshot(
            portfolio_id=PortfolioId("test"),
            mode=Mode.MANUAL_ALPHA,
            as_of=NOW,
            base_currency=Currency.USD,
            cash=(CashBalance(Currency.USD, Decimal("100")),),
            positions=(
                MarkedPosition(
                    stock(4, Market.US, "USD"), 1, Decimal("10"), NOW + timedelta(seconds=1)
                ),
            ),
        )


def test_invalid_balances_and_prices():
    with pytest.raises(ValueError):
        CashBalance(Currency.USD, Decimal("0"), Decimal("-1"))
    with pytest.raises(ValueError):
        CapitalLimit(Currency.USD, Decimal("NaN"))
    with pytest.raises(ValueError):
        MarkedPosition(stock(5, Market.US, "USD"), 0, Decimal("10"), NOW)
    with pytest.raises(ValueError):
        MarkedPosition(stock(5, Market.US, "USD"), 1, Decimal("0"), NOW)

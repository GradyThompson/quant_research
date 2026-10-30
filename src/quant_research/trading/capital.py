"""Immutable portfolio capital and exposure snapshots.

Amounts are currency-native. No implicit FX conversions, broker buying-power
assumptions, mark-to-market calculation, or position-netting across portfolios.
This is an input contract for portfolio constructors, not a trading ledger.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from quant_research.trading.contracts import Instrument, Mode, PortfolioId


class Currency(StrEnum):
    USD = "USD"
    CAD = "CAD"


def _finite(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")


def _nonnegative(value: Decimal, name: str) -> None:
    _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be UTC timezone-aware")


@dataclass(frozen=True, slots=True)
class CashBalance:
    """Cash and collateral commitments in one currency.

    `settled` may be negative for margin/debit balances. `reserved` includes
    outstanding order cash commitments and must not be counted again in
    `available_to_allocate`. It does not include short-sale proceeds unless
    the broker reports them as freely available collateral.
    """

    currency: Currency
    settled: Decimal
    reserved: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        _finite(self.settled, "settled")
        _nonnegative(self.reserved, "reserved")

    @property
    def unreserved_settled(self) -> Decimal:
        """Informational only; not equivalent to broker buying power."""
        return self.settled - self.reserved


@dataclass(frozen=True, slots=True)
class MarkedPosition:
    """Signed share quantity with a contemporaneous price and currency."""

    instrument: Instrument
    quantity: int
    mark_price: Decimal
    mark_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.quantity, int)
            or isinstance(self.quantity, bool)
            or self.quantity == 0
        ):
            raise ValueError("quantity must be a nonzero integer")
        _finite(self.mark_price, "mark_price")
        if self.mark_price <= 0:
            raise ValueError("mark_price must be positive")
        _utc(self.mark_at, "mark_at")

    @property
    def signed_market_value(self) -> Decimal:
        return Decimal(self.quantity) * self.mark_price

    @property
    def gross_market_value(self) -> Decimal:
        return abs(self.signed_market_value)


@dataclass(frozen=True, slots=True)
class CapitalLimit:
    """Broker/allocator-provided usable capital, not inferred from cash."""

    currency: Currency
    available_to_allocate: Decimal

    def __post_init__(self) -> None:
        _nonnegative(self.available_to_allocate, "available_to_allocate")


@dataclass(frozen=True, slots=True)
class PortfolioCapitalSnapshot:
    portfolio_id: PortfolioId
    mode: Mode
    as_of: datetime
    base_currency: Currency
    cash: tuple[CashBalance, ...]
    positions: tuple[MarkedPosition, ...] = ()
    limits: tuple[CapitalLimit, ...] = ()

    def __post_init__(self) -> None:
        _utc(self.as_of, "as_of")
        if not self.cash:
            raise ValueError("cash must contain the base currency")
        cash_ccys = [c.currency for c in self.cash]
        if len(set(cash_ccys)) != len(cash_ccys):
            raise ValueError("duplicate cash currency")
        if self.base_currency not in cash_ccys:
            raise ValueError("base currency cash balance missing")
        limit_ccys = [c.currency for c in self.limits]
        if len(set(limit_ccys)) != len(limit_ccys):
            raise ValueError("duplicate capital limit currency")
        ids = [p.instrument.security_id for p in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate position security ID")
        for position in self.positions:
            if position.mark_at > self.as_of:
                raise ValueError("position mark is from the future")
            if self.mode is not Mode.MANUAL_ALPHA and position.instrument.currency != "USD":
                raise ValueError("automated portfolios are US-equity USD only")
        if self.mode is not Mode.MANUAL_ALPHA and (
            self.base_currency is not Currency.USD
            or any(c.currency is not Currency.USD for c in self.cash)
            or any(c.currency is not Currency.USD for c in self.limits)
        ):
            raise ValueError("automated portfolios must use USD capital only")

    def exposures_by_currency(self) -> dict[Currency, tuple[Decimal, Decimal]]:
        """Returns signed net and absolute gross market value, per currency.

        Avoid comparing exposures to equity until an explicitly timestamped
        FX/valuation service provides a common-currency equity figure.
        """
        totals: dict[Currency, tuple[Decimal, Decimal]] = {}
        for position in self.positions:
            currency = Currency(position.instrument.currency)
            net, gross = totals.get(currency, (Decimal("0"), Decimal("0")))
            totals[currency] = (
                net + position.signed_market_value,
                gross + position.gross_market_value,
            )
        return totals

    def available_to_allocate(self, currency: Currency) -> Decimal:
        """Use explicit allocated buying power; missing limit means zero."""
        return next(
            (limit.available_to_allocate for limit in self.limits if limit.currency is currency),
            Decimal("0"),
        )

"""Local, deterministic portfolio accounting from confirmed fills.

Trade-date cash is NOT settled cash or broker buying power. This reducer is
independent of broker reconciliation, borrow charges, dividends and FX.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal

from quant_research.trading.capital import Currency
from quant_research.trading.contracts import Instrument, Mode, PortfolioId


def _amount(value: Decimal, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class LedgerFill:
    """One uniquely identified, confirmed fill; signed quantity (+buy/-sell).

    Selling beyond a long position opens a short. Buying beyond a short covers
    and opens a long. The accounting convention is weighted average cost.
    """

    execution_id: str
    order_id: str
    instrument: Instrument
    quantity: int
    price: Decimal
    commission: Decimal
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not self.execution_id or not self.order_id:
            raise ValueError("execution_id and order_id are required")
        if type(self.quantity) is not int or self.quantity == 0:
            raise ValueError("quantity must be a nonzero signed integer")
        _amount(self.price, "price", positive=True)
        _amount(self.commission, "commission")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() != UTC.utcoffset(
            self.occurred_at
        ):
            raise ValueError("occurred_at must be UTC aware")


@dataclass(frozen=True, slots=True)
class LedgerPosition:
    instrument: Instrument
    quantity: int
    average_entry_price: Decimal
    realized_pnl: Decimal = Decimal("0")  # gross realized P&L, excludes commissions

    def __post_init__(self) -> None:
        if type(self.quantity) is not int:
            raise ValueError("quantity must be an integer")
        _amount(self.average_entry_price, "average_entry_price")
        if not self.realized_pnl.is_finite():
            raise ValueError("realized_pnl must be finite")

    def unrealized_pnl(self, mark: Decimal) -> Decimal:
        _amount(mark, "mark", positive=True)
        return Decimal(self.quantity) * (mark - self.average_entry_price)


@dataclass(frozen=True, slots=True)
class PortfolioLedger:
    """Immutable trade-date cash and share inventory per logical portfolio.

    `cash` represents trade-date cash movements, not available or settled cash.
    Replay all fills in a single consistent time ordering. Repeated execution
    IDs are safe no-ops; conflicting reuses are rejected.
    """

    portfolio_id: PortfolioId
    mode: Mode
    cash: dict[Currency, Decimal] = field(default_factory=dict)
    positions: dict[str, LedgerPosition] = field(default_factory=dict)
    processed_fills: dict[str, LedgerFill] = field(default_factory=dict)
    last_fill_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.mode is not Mode.MANUAL_ALPHA and any(c is not Currency.USD for c in self.cash):
            raise ValueError("automated portfolios are USD only")
        for currency, balance in self.cash.items():
            if (
                not isinstance(currency, Currency)
                or not isinstance(balance, Decimal)
                or not balance.is_finite()
            ):
                raise ValueError("invalid cash balance")
        if self.last_fill_at is not None and (
            self.last_fill_at.tzinfo is None
            or self.last_fill_at.utcoffset() != UTC.utcoffset(self.last_fill_at)
        ):
            raise ValueError("last_fill_at must be UTC aware")

    def apply(self, fill: LedgerFill) -> PortfolioLedger:
        prior = self.processed_fills.get(fill.execution_id)
        if prior is not None:
            if prior != fill:
                raise ValueError("conflicting execution ID")
            return self
        if self.last_fill_at is not None and fill.occurred_at < self.last_fill_at:
            raise ValueError("out-of-order fill: replay chronologically")
        if self.mode is not Mode.MANUAL_ALPHA and fill.instrument.currency != "USD":
            raise ValueError("automated portfolios accept USD equities only")
        currency = Currency(fill.instrument.currency)
        key = fill.instrument.security_id.value
        old = self.positions.get(key)
        if old is not None and old.instrument != fill.instrument:
            raise ValueError("instrument identity changed for an existing position")
        old_qty = old.quantity if old is not None else 0
        old_basis = old.average_entry_price if old is not None else Decimal("0")
        old_realized = old.realized_pnl if old is not None else Decimal("0")
        new_qty = old_qty + fill.quantity
        realized = old_realized
        if old_qty == 0 or (old_qty > 0) == (fill.quantity > 0):
            new_basis = (
                Decimal(abs(old_qty)) * old_basis + Decimal(abs(fill.quantity)) * fill.price
            ) / Decimal(abs(new_qty))
        else:
            closed = min(abs(old_qty), abs(fill.quantity))
            direction = Decimal(1 if old_qty > 0 else -1)
            realized += Decimal(closed) * (fill.price - old_basis) * direction
            if new_qty == 0:
                new_basis = Decimal("0")
            elif (new_qty > 0) == (old_qty > 0):
                new_basis = old_basis
            else:
                new_basis = fill.price
        positions = dict(self.positions)
        positions[key] = LedgerPosition(fill.instrument, new_qty, new_basis, realized)
        cash = dict(self.cash)
        cash[currency] = (
            cash.get(currency, Decimal("0")) - Decimal(fill.quantity) * fill.price - fill.commission
        )
        fills = dict(self.processed_fills)
        fills[fill.execution_id] = fill
        return replace(
            self,
            cash=cash,
            positions=positions,
            processed_fills=fills,
            last_fill_at=fill.occurred_at,
        )

    def gross_realized_pnl(self, currency: Currency) -> Decimal:
        return sum(
            (
                p.realized_pnl
                for p in self.positions.values()
                if p.instrument.currency == currency.value
            ),
            Decimal("0"),
        )

    def commissions(self, currency: Currency) -> Decimal:
        return sum(
            (
                f.commission
                for f in self.processed_fills.values()
                if f.instrument.currency == currency.value
            ),
            Decimal("0"),
        )

    def equity_at_marks(self, marks: dict[str, Decimal], currency: Currency) -> Decimal:
        """Value a single currency. All open positions require explicit marks.

        No CAD/USD aggregation or broker buying-power inference is performed.
        """
        value = self.cash.get(currency, Decimal("0"))
        for key, position in self.positions.items():
            if position.instrument.currency != currency.value or position.quantity == 0:
                continue
            if key not in marks:
                raise ValueError(f"missing mark for {key}")
            price = marks[key]
            _amount(price, "mark", positive=True)
            value += Decimal(position.quantity) * price
        return value

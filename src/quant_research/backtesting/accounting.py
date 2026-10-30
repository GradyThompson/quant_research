"""Pure bridge from daily simulated fills to the portfolio ledger.

This module does not decide whether orders fill; it only applies already
simulated fills to a specific logical portfolio with stable execution IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quant_research.backtesting.execution import Side, SimulatedFill
from quant_research.trading.contracts import Instrument
from quant_research.trading.ledger import LedgerFill, PortfolioLedger


@dataclass(frozen=True, slots=True)
class AccountingResult:
    ledger: PortfolioLedger
    applied: tuple[LedgerFill, ...]


def apply_simulated_fills(
    ledger: PortfolioLedger,
    fills: tuple[SimulatedFill, ...],
    instruments: dict[str, Instrument],
) -> AccountingResult:
    """Post confirmed simulated executions in provided chronological order.

    The caller owns the security -> instrument mapping. Only integer shares
    and supported currencies enter the ledger. Repeating an identical fill is
    idempotent; reusing a stable execution ID with new economics is an error.

    v1 simulates one atomic fill per historical order. `order_id` is therefore
    the stable execution identity in this integration; do not use this mapping
    with multi-fill orders without introducing per-fill IDs first.
    """
    updated = ledger
    applied: list[LedgerFill] = []
    for fill in fills:
        if fill.security_id not in instruments:
            raise ValueError(f"missing instrument for {fill.security_id}")
        instrument = instruments[fill.security_id]
        if instrument.security_id.value != fill.security_id:
            raise ValueError("instrument security ID does not match fill")
        if not fill.order_id:
            raise ValueError("fill requires order ID")
        if type(fill.quantity) is not int or fill.quantity <= 0:
            raise ValueError("fill quantity must be a positive integer")
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        ledger_fill = LedgerFill(
            execution_id=f"historical:{fill.order_id}",
            order_id=fill.order_id,
            instrument=instrument,
            quantity=signed,
            price=fill.price,
            commission=fill.commission,
            occurred_at=fill.bar_timestamp,
        )
        updated_next = updated.apply(ledger_fill)
        if updated_next is not updated:
            applied.append(ledger_fill)
        updated = updated_next
    return AccountingResult(ledger=updated, applied=tuple(applied))


def mark_to_market_usd(ledger: PortfolioLedger, marks: dict[str, Decimal]) -> Decimal:
    """USD-only valuation; never silently combine CAD and USD amounts."""
    from quant_research.trading.capital import Currency

    return ledger.equity_at_marks(marks, Currency.USD)

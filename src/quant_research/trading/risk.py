"""Conservative pre-trade risk gate for portfolio plans.

This is a *planning* gate. Broker-side buying power, borrows, market state,
price bands and order deduplication require additional checks before routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from quant_research.trading.contracts import (
    ApprovedIntent,
    Authority,
    CancelIntent,
    Market,
    Mode,
    PortfolioPlan,
    RejectedIntent,
    Side,
    TargetWeightIntent,
    TradePlanIntent,
)


@dataclass(frozen=True, slots=True)
class RiskLimits:
    max_snapshot_age: timedelta = timedelta(minutes=5)
    max_order_notional_usd: Decimal = Decimal("5000")
    max_gross_target_weight: Decimal = Decimal("1")
    max_absolute_target_weight: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        if self.max_snapshot_age <= timedelta(0):
            raise ValueError("max_snapshot_age must be positive")
        for name in (
            "max_order_notional_usd",
            "max_gross_target_weight",
            "max_absolute_target_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a positive finite Decimal")


@dataclass(frozen=True, slots=True)
class RiskContext:
    mode: Mode
    now: datetime
    portfolio_as_of: datetime
    halted: bool = False

    def __post_init__(self) -> None:
        for name, value in (("now", self.now), ("portfolio_as_of", self.portfolio_as_of)):
            if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
                raise ValueError(f"{name} must be UTC aware")
        if self.portfolio_as_of > self.now:
            raise ValueError("portfolio snapshot cannot be from future")


@dataclass(frozen=True, slots=True)
class PreTradeRiskGate:
    limits: RiskLimits = RiskLimits()

    def check(self, plan: PortfolioPlan, context: RiskContext) -> PortfolioPlan:
        """Reject risky approvals, preserve original rejections and cancel intents.

        No orders are routed. Plans with target weights are checked as a set.
        Short sales and market orders are intentionally not cleared here.
        """
        approved: list[ApprovedIntent] = []
        rejected = list(plan.rejected)
        stale = context.now - context.portfolio_as_of > self.limits.max_snapshot_age
        targets: list[TargetWeightIntent] = [
            item.intent for item in plan.approved if isinstance(item.intent, TargetWeightIntent)
        ]
        gross = sum((abs(Decimal(str(intent.weight))) for intent in targets), Decimal(0))
        gross_exceeded = gross > self.limits.max_gross_target_weight
        for item in plan.approved:
            intent = item.intent
            if isinstance(intent, CancelIntent):
                approved.append(item)  # Always permit cancellation requests, including on halt.
                continue
            reason: str | None = None
            if context.mode is Mode.MANUAL_ALPHA or plan.authority is Authority.MANUAL:
                reason = "advisory plan is not executable"
            elif context.halted:
                reason = "trading halted"
            elif stale:
                reason = "portfolio snapshot stale"
            elif isinstance(intent, TargetWeightIntent):
                if context.mode is not Mode.AUTOMATED_ALPHA:
                    reason = "target weights unsupported in this mode"
                elif intent.instrument.market is not Market.US:
                    reason = "automated orders must be US equities"
                elif gross_exceeded:
                    reason = "gross target exposure exceeds risk limit"
                elif abs(Decimal(str(intent.weight))) > self.limits.max_absolute_target_weight:
                    reason = "single-name target weight exceeds risk limit"
                else:
                    # This approval is of an allocation *intent*, not an executable order.
                    approved.append(item)
                    continue
            elif isinstance(intent, TradePlanIntent):
                entry = intent.entry
                if context.mode is not Mode.SHORT_TERM:
                    reason = "trade plans unsupported in this mode"
                elif (
                    intent.instrument.market is not Market.US or intent.instrument.currency != "USD"
                ):
                    reason = "automated trades must be US USD equities"
                elif entry.side is not Side.BUY or entry.limit_price is None:
                    reason = "only priced long entries supported"
                elif entry.expires_at is not None and entry.expires_at <= context.now:
                    reason = "trade plan expired"
                else:
                    quantity = item.approved_quantity or entry.quantity
                    notional = Decimal(quantity) * Decimal(str(entry.limit_price))
                    if notional > self.limits.max_order_notional_usd:
                        reason = "order notional exceeds risk limit"
                    else:
                        approved.append(item)
                        continue
            else:
                reason = "unsupported intent for executable risk approval"
            rejected.append(RejectedIntent(intent, reason or "risk denied"))
        return PortfolioPlan(plan.portfolio_id, plan.authority, tuple(approved), tuple(rejected))

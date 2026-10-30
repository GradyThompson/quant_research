"""Conservative portfolio constructors for three investment modes.

These are *planning* policies; nothing here places orders. Approval is not a
broker risk check. Unpriced market orders and unverified shorts are rejected.
"""

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite

from quant_research.trading.capital import Currency, PortfolioCapitalSnapshot
from quant_research.trading.contracts import (
    ApprovedIntent,
    Authority,
    CancelIntent,
    ForecastIntent,
    Market,
    Mode,
    OrderKind,
    PortfolioContext,
    PortfolioPlan,
    RejectedIntent,
    Side,
    StrategyDecision,
    TargetWeightIntent,
    TradePlanIntent,
)


def _check(context: PortfolioContext, decision: StrategyDecision, mode: Mode) -> None:
    if context.mode is not mode:
        raise ValueError(f"constructor requires {mode.value} mode")
    if decision.portfolio_id != context.snapshot.portfolio_id:
        raise ValueError("strategy decision and portfolio context ID mismatch")
    if decision.decided_at < context.snapshot.as_of:
        raise ValueError("strategy decision precedes portfolio snapshot")


def _make_plan(
    context: PortfolioContext,
    approved: list[ApprovedIntent],
    rejected: list[RejectedIntent],
) -> PortfolioPlan:
    return PortfolioPlan(
        context.snapshot.portfolio_id, context.authority, tuple(approved), tuple(rejected)
    )


@dataclass(frozen=True, slots=True)
class AutomatedAlphaConstructor:
    """Validate explicit USD long/short target weights against gross exposure.

    Forecast conversion and optimization remain a separate future component.
    Does not infer buying power or validate margin/borrow availability.
    """

    def construct(self, context: PortfolioContext, decision: StrategyDecision) -> PortfolioPlan:
        _check(context, decision, Mode.AUTOMATED_ALPHA)
        approved: list[ApprovedIntent] = []
        rejected: list[RejectedIntent] = []
        targets = [intent for intent in decision.intents if isinstance(intent, TargetWeightIntent)]
        seen: set[object] = set()
        gross = 0.0
        for target in targets:
            sid = target.instrument.security_id
            if sid in seen:
                raise ValueError("duplicate target security")
            seen.add(sid)
            if target.instrument.market is not Market.US or target.instrument.currency != "USD":
                raise ValueError("automated alpha is US-equity USD only")
            gross += abs(target.weight)
        within_limit = isfinite(gross) and gross <= context.max_gross_exposure + 1e-12
        for intent in decision.intents:
            if isinstance(intent, TargetWeightIntent):
                if within_limit:
                    approved.append(ApprovedIntent(intent))
                else:
                    rejected.append(RejectedIntent(intent, "target gross exposure exceeds limit"))
            elif isinstance(intent, CancelIntent):
                approved.append(ApprovedIntent(intent))
            else:
                rejected.append(RejectedIntent(intent, "only explicit target weights supported"))
        return _make_plan(context, approved, rejected)


@dataclass(frozen=True, slots=True)
class ManualAlphaConstructor:
    """Advisory-only forecasts and target weights, including CAD equities."""

    def construct(self, context: PortfolioContext, decision: StrategyDecision) -> PortfolioPlan:
        _check(context, decision, Mode.MANUAL_ALPHA)
        if context.authority is not Authority.MANUAL:
            raise ValueError("manual portfolio must have manual authority")
        approved: list[ApprovedIntent] = []
        rejected: list[RejectedIntent] = []
        for intent in decision.intents:
            if isinstance(intent, (ForecastIntent, TargetWeightIntent)):
                approved.append(ApprovedIntent(intent))
            else:
                rejected.append(RejectedIntent(intent, "manual mode provides recommendations only"))
        return _make_plan(context, approved, rejected)


@dataclass(frozen=True, slots=True)
class ShortTermTradeConstructor:
    """Budget-check explicit long-entry LIMIT plans; never submit broker orders.

    Capital uses the order's LIMIT price, a conservative *entry* reservation.
    Existing commitments must already be deducted by the capital provider.
    No shorts, margin sizing, fees, market entries or cross-currency offsets.
    Exit specifications are passed through unchanged and not activated here.
    """

    def construct(
        self,
        context: PortfolioContext,
        decision: StrategyDecision,
        capital: PortfolioCapitalSnapshot | None = None,
    ) -> PortfolioPlan:
        _check(context, decision, Mode.SHORT_TERM)
        if capital is None:
            raise ValueError("short-term sizing requires capital snapshot")
        if (
            capital.portfolio_id != context.snapshot.portfolio_id
            or capital.mode is not Mode.SHORT_TERM
        ):
            raise ValueError("capital snapshot portfolio/mode mismatch")
        if capital.as_of > decision.decided_at:
            raise ValueError("capital snapshot is from the future")
        budget = capital.available_to_allocate(Currency.USD)
        approved: list[ApprovedIntent] = []
        rejected: list[RejectedIntent] = []
        existing_ids: set[str] = set()
        for intent in decision.intents:
            if isinstance(intent, CancelIntent):
                approved.append(ApprovedIntent(intent))
                continue
            if not isinstance(intent, TradePlanIntent):
                rejected.append(RejectedIntent(intent, "short-term mode requires trade plans"))
                continue
            if intent.intent_id in existing_ids:
                raise ValueError("duplicate trade plan ID")
            existing_ids.add(intent.intent_id)
            if (
                intent.instrument.market is not Market.US
                or intent.instrument.currency != "USD"
                or intent.entry.side is not Side.BUY
                or intent.entry.kind is not OrderKind.LIMIT
                or intent.entry.limit_price is None
            ):
                rejected.append(RejectedIntent(intent, "requires US USD long limit entry"))
                continue
            if (
                intent.entry.expires_at is not None
                and intent.entry.expires_at <= decision.decided_at
            ):
                rejected.append(RejectedIntent(intent, "entry already expired"))
                continue
            price = Decimal(str(intent.entry.limit_price))
            affordable = int(budget // price)
            quantity = min(intent.entry.quantity, affordable)
            if quantity <= 0 or (
                quantity < intent.entry.quantity and not intent.may_reduce_quantity
            ):
                rejected.append(RejectedIntent(intent, "insufficient allocation capacity"))
                continue
            # We don't adjust exit quantities here: the order manager must
            # scale/reconcile contingent exit orders against actual entry fills.
            budget -= price * quantity
            approved.append(ApprovedIntent(intent, approved_quantity=quantity))
        return _make_plan(context, approved, rejected)

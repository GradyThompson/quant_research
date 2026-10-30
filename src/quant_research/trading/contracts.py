"""Offline contracts for multi-mode investment decisions.

No broker calls, logging, data uploads, order submission, or local plugins.
Private strategies can implement the Strategy Protocol in an ignored folder.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Protocol, TypeAlias

from quant_research.core import SecurityId


def _check_time(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must use UTC")


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


class Market(StrEnum):
    US = "US"
    CA = "CA"


class Mode(StrEnum):
    AUTOMATED_ALPHA = "automated_alpha"
    MANUAL_ALPHA = "manual_alpha"
    SHORT_TERM = "short_term"


class Authority(StrEnum):
    AUTOMATED = "automated"
    MANUAL = "manual"


class Environment(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_COVER = "buy_to_cover"


class OrderKind(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


@dataclass(frozen=True, slots=True)
class PortfolioId:
    value: str

    def __post_init__(self) -> None:
        if (
            not self.value
            or not self.value.isascii()
            or not all(c.isalnum() or c in "_-" for c in self.value)
        ):
            raise ValueError("portfolio ID must be nonempty ASCII letters/digits/_/-")


@dataclass(frozen=True, slots=True)
class Instrument:
    security_id: SecurityId
    market: Market
    currency: str

    def __post_init__(self) -> None:
        if (self.market is Market.US and self.currency != "USD") or (
            self.market is Market.CA and self.currency != "CAD"
        ):
            raise ValueError("market/currency mismatch (currency is trading currency)")


@dataclass(frozen=True, slots=True)
class Prediction:
    instrument: Instrument
    metric: str  # e.g. expected_return, relative_score, upper_first_probability
    value: float
    observed_at: datetime
    available_at: datetime
    horizon_seconds: int
    model_version: str

    def __post_init__(self) -> None:
        _check_time(self.observed_at, "observed_at")
        _check_time(self.available_at, "available_at")
        if self.available_at < self.observed_at:
            raise ValueError("prediction cannot be available before observation")
        if not self.metric or not self.model_version or self.horizon_seconds <= 0:
            raise ValueError("prediction needs metric, model version and positive horizon")
        if not isfinite(self.value):
            raise ValueError("prediction value must be finite")
        if self.metric.endswith("probability") and not 0 <= self.value <= 1:
            raise ValueError("probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Position:
    instrument: Instrument
    quantity: int  # signed, long positive / short negative


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    portfolio_id: PortfolioId
    as_of: datetime
    base_currency: str
    equity: float
    positions: tuple[Position, ...] = ()

    def __post_init__(self) -> None:
        _check_time(self.as_of, "as_of")
        _positive(self.equity, "equity")
        if self.base_currency not in ("USD", "CAD"):
            raise ValueError("base currency must be USD or CAD")
        ids = [p.instrument.security_id for p in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate positions")


@dataclass(frozen=True, slots=True)
class StrategyContext:
    portfolio: PortfolioSnapshot
    decision_at: datetime
    mode: Mode
    environment: Environment
    market_data: object  # readonly point-in-time view contract to be refined separately
    state: bytes = b""  # opaque private state; do not log

    def __post_init__(self) -> None:
        _check_time(self.decision_at, "decision_at")
        if self.portfolio.as_of > self.decision_at:
            raise ValueError("portfolio snapshot cannot be from the future")


@dataclass(frozen=True, slots=True)
class ForecastIntent:
    prediction: Prediction


@dataclass(frozen=True, slots=True)
class TargetWeightIntent:
    instrument: Instrument
    weight: float  # signed; negative = short

    def __post_init__(self) -> None:
        if not isfinite(self.weight) or abs(self.weight) > 1:
            raise ValueError("target weight must be finite within [-1,1]")


@dataclass(frozen=True, slots=True)
class OrderSpecification:
    side: Side
    kind: OrderKind
    quantity: int
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0 or isinstance(self.quantity, bool):
            raise ValueError("quantity must be a positive integer")
        if self.kind in (OrderKind.LIMIT, OrderKind.STOP_LIMIT) and self.limit_price is None:
            raise ValueError("limit price required")
        if self.kind in (OrderKind.STOP, OrderKind.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("stop price required")
        if self.kind in (OrderKind.MARKET, OrderKind.STOP) and self.limit_price is not None:
            raise ValueError("limit price not applicable")
        if self.kind in (OrderKind.MARKET, OrderKind.LIMIT) and self.stop_price is not None:
            raise ValueError("stop price not applicable")
        for name, price in (("limit_price", self.limit_price), ("stop_price", self.stop_price)):
            if price is not None:
                _positive(price, name)
        if self.expires_at is not None:
            _check_time(self.expires_at, "expires_at")


@dataclass(frozen=True, slots=True)
class ExitRule:
    """Exit intent activated after fills, NOT an immediately submitted sell."""

    kind: str  # profit_target, protective_stop, time_exit, private_condition
    order: OrderSpecification | None = None
    after_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("profit_target", "protective_stop", "time_exit", "private_condition"):
            raise ValueError("unknown exit kind")
        if self.kind in ("profit_target", "protective_stop") and self.order is None:
            raise ValueError("price exits require an order spec")
        if self.kind == "time_exit" and (self.after_seconds is None or self.after_seconds <= 0):
            raise ValueError("time exit requires positive after_seconds")


@dataclass(frozen=True, slots=True)
class TradePlanIntent:
    intent_id: str  # idempotency key scoped to strategy/portfolio
    instrument: Instrument
    entry: OrderSpecification
    exits: tuple[ExitRule, ...] = ()
    may_reduce_quantity: bool = True

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("intent_id required")


@dataclass(frozen=True, slots=True)
class CancelIntent:
    intent_id: str  # references previously issued trade plan or open order

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("cancel intent_id required")


Intent: TypeAlias = ForecastIntent | TargetWeightIntent | TradePlanIntent | CancelIntent


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    portfolio_id: PortfolioId
    decided_at: datetime
    intents: tuple[Intent, ...]
    next_state: bytes = b""

    def __post_init__(self) -> None:
        _check_time(self.decided_at, "decided_at")
        keys: set[str] = set()
        for intent in self.intents:
            if isinstance(intent, (TradePlanIntent, CancelIntent)):
                if intent.intent_id in keys:
                    raise ValueError("duplicate trade/cancel intent ID")
                keys.add(intent.intent_id)


class Strategy(Protocol):
    def evaluate(self, context: StrategyContext) -> StrategyDecision: ...


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    snapshot: PortfolioSnapshot
    authority: Authority
    mode: Mode
    max_gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.max_gross_exposure, "max_gross_exposure")
        if self.mode is Mode.MANUAL_ALPHA and self.authority is not Authority.MANUAL:
            raise ValueError("manual alpha cannot have automatic submission authority")


@dataclass(frozen=True, slots=True)
class ApprovedIntent:
    intent: Intent
    approved_quantity: int | None = None

    def __post_init__(self) -> None:
        if self.approved_quantity is not None:
            if not isinstance(self.intent, TradePlanIntent):
                raise ValueError("quantity override only applies to trade plans")
            if self.approved_quantity < 1 or self.approved_quantity > self.intent.entry.quantity:
                raise ValueError("approved quantity must be between 1 and requested quantity")
            if (
                not self.intent.may_reduce_quantity
                and self.approved_quantity != self.intent.entry.quantity
            ):
                raise ValueError("trade plan forbids resizing")


@dataclass(frozen=True, slots=True)
class RejectedIntent:
    intent: Intent
    reason: str


@dataclass(frozen=True, slots=True)
class PortfolioPlan:
    portfolio_id: PortfolioId
    authority: Authority
    approved: tuple[ApprovedIntent, ...]
    rejected: tuple[RejectedIntent, ...]


class PortfolioConstructor(Protocol):
    def construct(self, context: PortfolioContext, decision: StrategyDecision) -> PortfolioPlan: ...


class EventKind(StrEnum):
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACKNOWLEDGED = "order_acknowledged"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    MANUAL_FILL_RECORDED = "manual_fill_recorded"
    RISK_HALT = "risk_halt"


@dataclass(frozen=True, slots=True)
class TradingEvent:
    event_id: str
    portfolio_id: PortfolioId
    event_at: datetime
    kind: EventKind
    related_intent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id required")
        _check_time(self.event_at, "event_at")

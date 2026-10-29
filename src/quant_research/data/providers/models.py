"""Provider-boundary contracts; do not assume provider symbols are SecurityIds."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite

from quant_research.core import TimeRange
from quant_research.data.schemas import BarFrequency


class ProviderDataError(Exception):
    """Provider request or response is invalid (not a storage error)."""


class ProviderCapabilityError(ProviderDataError):
    """The provider does not support the requested operation."""


class InstrumentKind(StrEnum):
    TICKER = "ticker"
    FIGI = "figi"
    PROVIDER_ID = "provider_id"


def _aware(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ProviderInstrument:
    """Instrument as addressed by a particular data provider."""

    identifier: str
    identifier_kind: InstrumentKind
    mic: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier or self.identifier != self.identifier.strip():
            raise ValueError("identifier must be nonblank without surrounding whitespace")
        if self.mic is not None and (len(self.mic) != 4 or not self.mic.isupper()):
            raise ValueError("mic must be a four-letter uppercase code")


@dataclass(frozen=True, slots=True)
class BarRequest:
    instrument: ProviderInstrument
    frequency: BarFrequency
    period: TimeRange  # half-open interval, bar end timestamps in this interval


@dataclass(frozen=True, slots=True)
class ProviderBar:
    """Unadjusted vendor bar; no internal security ID assigned here.

    available_at is a provider/source availability assertion, not necessarily
    when a particular research system could have acted on the observation.
    """

    timestamp: datetime
    available_at: datetime
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: int | None
    currency: str

    def __post_init__(self) -> None:
        _aware(self.timestamp, "timestamp")
        _aware(self.available_at, "available_at")
        if self.available_at < self.timestamp:
            raise ValueError("available_at cannot precede timestamp")
        if not self.currency or len(self.currency) != 3 or not self.currency.isupper():
            raise ValueError("currency must be a three-letter uppercase code")
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.volume is not None and (isinstance(self.volume, bool) or self.volume < 0):
            raise ValueError("volume must be nonnegative")
        if self.high is not None and self.low is not None and self.high < self.low:
            raise ValueError("high cannot be below low")
        for name in ("open", "close"):
            value = getattr(self, name)
            if value is not None and self.high is not None and value > self.high:
                raise ValueError(f"{name} cannot exceed high")
            if value is not None and self.low is not None and value < self.low:
                raise ValueError(f"{name} cannot be below low")


@dataclass(frozen=True, slots=True)
class BarResponse:
    provider: str
    request: BarRequest
    retrieved_at: datetime
    bars: tuple[ProviderBar, ...]

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be nonblank")
        _aware(self.retrieved_at, "retrieved_at")
        if any(not self.request.period.contains(bar.timestamp) for bar in self.bars):
            raise ValueError("bar timestamps must be within the requested period")
        timestamps = [bar.timestamp for bar in self.bars]
        if timestamps != sorted(set(timestamps)):
            raise ValueError("bars must have unique strictly increasing timestamps")

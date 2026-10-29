"""Temporal contracts for point-in-time quantitative research."""

from dataclasses import dataclass
from datetime import UTC, datetime


def _require_aware_datetime(
    value: datetime,
    field_name: str,
) -> None:
    """Require a datetime with an explicit UTC offset."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Half-open interval: [start, end)."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        _require_aware_datetime(self.start, "start")
        _require_aware_datetime(self.end, "end")

        if self.start >= self.end:
            raise ValueError("TimeRange.start must be earlier than TimeRange.end.")

    def contains(self, timestamp: datetime) -> bool:
        """Return whether timestamp belongs to the interval."""

        _require_aware_datetime(timestamp, "timestamp")

        return self.start <= timestamp < self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Return whether two half-open intervals overlap."""

        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class TemporalMetadata:
    """Information about the timing of an observation.

    event_time:
        When the underlying event occurred.

    available_at:
        Earliest time the observation was available to the
        research or trading decision process.

    ingested_at:
        When the platform recorded the observation.
    """

    event_time: datetime
    available_at: datetime
    ingested_at: datetime

    def __post_init__(self) -> None:
        _require_aware_datetime(self.event_time, "event_time")
        _require_aware_datetime(self.available_at, "available_at")
        _require_aware_datetime(self.ingested_at, "ingested_at")

        if self.event_time > self.available_at:
            raise ValueError("event_time cannot be later than available_at.")

        if self.available_at > self.ingested_at:
            raise ValueError("available_at cannot be later than ingested_at.")

    def available_by(self, decision_time: datetime) -> bool:
        """Whether the observation was available by decision_time."""

        _require_aware_datetime(decision_time, "decision_time")

        return self.available_at <= decision_time

    def recorded_by(self, as_of: datetime) -> bool:
        """Whether the platform had recorded it by the given time."""

        _require_aware_datetime(as_of, "as_of")

        return self.ingested_at <= as_of

    def as_utc(self) -> "TemporalMetadata":
        """Return a copy normalized to UTC."""

        return TemporalMetadata(
            event_time=self.event_time.astimezone(UTC),
            available_at=self.available_at.astimezone(UTC),
            ingested_at=self.ingested_at.astimezone(UTC),
        )

"""Read-only, point-in-time OHLCV access for local private strategies.

The source is a snapshot of *already loaded* canonical OHLCV bars. Filtering
is enforced within the view; no direct access to unfiltered source is exposed.
This is a research/backtest adapter, not a real-time feed or data permission system.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import pyarrow as pa

from quant_research.core import SecurityId
from quant_research.data.schemas import BarFrequency, PriceAdjustment, VolumeUnit
from quant_research.data.validation.ohlcv import OHLCVValidator


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class MarketDataRequest:
    """Query a trailing time window, inclusive at start and cutoff at decision time."""

    security_ids: tuple[SecurityId, ...]
    start: datetime
    frequency: BarFrequency = BarFrequency.DAILY
    adjustment: PriceAdjustment = PriceAdjustment.RAW

    def __post_init__(self) -> None:
        _utc(self.start, "start")
        if len(self.security_ids) != len(set(self.security_ids)):
            raise ValueError("duplicate security IDs in request")


class MarketDataView(Protocol):
    """Read-only interface exposed to strategies, independent of storage."""

    @property
    def decision_at(self) -> datetime: ...

    def bars(self, request: MarketDataRequest) -> pa.Table: ...


class ArrowPointInTimeView:
    """In-memory historical view: only observations available by decision_at.

    The original table is validated and copied into immutable Python records.
    Arrow outputs are fresh tables, so caller modifications cannot alter the view.
    Does not use ingestion timestamp as a historical knowledge constraint because
    backfilled datasets can be ingested after the events they represent.
    """

    def __init__(
        self,
        bars: pa.Table,
        *,
        decision_at: datetime,
        frequency: BarFrequency = BarFrequency.DAILY,
        adjustment: PriceAdjustment = PriceAdjustment.RAW,
        volume_unit: VolumeUnit = VolumeUnit.SHARES,
    ) -> None:
        self._decision_at = _utc(decision_at, "decision_at")
        self._frequency = frequency
        self._adjustment = adjustment
        self._schema = bars.schema if isinstance(bars, pa.Table) else None
        OHLCVValidator(frequency, adjustment, volume_unit).validate(bars)
        # Drop future rows immediately, including rows with past event time but
        # availability after the decision timestamp.
        self._rows = tuple(
            row
            for row in bars.to_pylist()
            if row["timestamp"] <= self._decision_at and row["available_at"] <= self._decision_at
        )

    @property
    def decision_at(self) -> datetime:
        return self._decision_at

    def bars(self, request: MarketDataRequest) -> pa.Table:
        if request.frequency != self._frequency or request.adjustment != self._adjustment:
            raise ValueError("requested frequency or adjustment does not match view")
        start = _utc(request.start, "start")
        if start > self._decision_at:
            raise ValueError("request.start cannot be after decision_at")
        ids = {security_id.value for security_id in request.security_ids}
        selected = [
            row for row in self._rows if row["security_id"] in ids and start <= row["timestamp"]
        ]
        selected.sort(key=lambda row: (row["security_id"], row["timestamp"]))
        assert self._schema is not None
        return pa.Table.from_pylist(selected, schema=self._schema)

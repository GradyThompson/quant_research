"""Offline tests for Massive minute aggregates and point-in-time timestamps."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from quant_research.core import TimeRange
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.providers.models import (
    BarRequest,
    InstrumentKind,
    ProviderCapabilityError,
    ProviderDataError,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency

# pandas_market_calendars currently triggers this pandas 4 deprecation
# internally when checking exchange sessions. Keep the real calendar tests
# and narrowly filter only this upstream warning.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:DatetimeIndex\.values returning an ndarray that drops timezone "
    "information is deprecated:pandas.errors.Pandas4Warning"
)


def vendor_bar(start: datetime, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "timestamp": int(start.timestamp() * 1000),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeClient:
    def __init__(self, bars: list[SimpleNamespace]) -> None:
        self.bars = bars
        self.kwargs: dict[str, object] | None = None

    def list_aggs(self, **kwargs: object) -> list[SimpleNamespace]:
        self.kwargs = kwargs
        return self.bars


def request(
    *,
    frequency: BarFrequency = BarFrequency.MINUTE,
    kind: InstrumentKind = InstrumentKind.TICKER,
    mic: str = "XNAS",
) -> BarRequest:
    return BarRequest(
        instrument=ProviderInstrument("AAPL", kind, mic),
        frequency=frequency,
        period=TimeRange(
            datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
            datetime(2025, 1, 3, 22, tzinfo=UTC),
        ),
    )


def test_fetch_maps_start_to_end_and_uses_raw_aggs() -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    client = FakeClient([vendor_bar(start)])
    response = MassiveMinuteProvider(client=client).fetch_bars(request())
    assert len(response.bars) == 1
    bar = response.bars[0]
    assert bar.timestamp == datetime(2025, 1, 2, 14, 31, tzinfo=UTC)
    assert bar.available_at == bar.timestamp
    assert bar.volume == 1000
    assert client.kwargs is not None
    assert client.kwargs["timespan"] == "minute"
    assert client.kwargs["adjusted"] is False
    assert client.kwargs["multiplier"] == 1


def test_filters_extended_hours_and_outside_requested_end() -> None:
    client = FakeClient(
        [
            vendor_bar(datetime(2025, 1, 2, 13, 0, tzinfo=UTC)),
            vendor_bar(datetime(2025, 1, 2, 14, 30, tzinfo=UTC)),
            vendor_bar(datetime(2025, 1, 3, 22, 0, tzinfo=UTC)),
        ]
    )
    result = MassiveMinuteProvider(client=client).fetch_bars(request())
    assert len(result.bars) == 1


@pytest.mark.parametrize(
    ("frequency", "kind", "mic"),
    [
        (BarFrequency.DAILY, InstrumentKind.TICKER, "XNAS"),
        (BarFrequency.MINUTE, InstrumentKind.FIGI, "XNAS"),
        (BarFrequency.MINUTE, InstrumentKind.TICKER, "XTSE"),
    ],
)
def test_rejects_unsupported_requests(
    frequency: BarFrequency, kind: InstrumentKind, mic: str
) -> None:
    with pytest.raises(ProviderCapabilityError):
        MassiveMinuteProvider(client=FakeClient([])).fetch_bars(
            request(frequency=frequency, kind=kind, mic=mic)
        )


@pytest.mark.parametrize("volume", [-1, 3.5, float("nan")])
def test_rejects_invalid_volume(volume: float) -> None:
    bar = vendor_bar(datetime(2025, 1, 2, 14, 30, tzinfo=UTC), volume=volume)
    with pytest.raises(ProviderDataError, match="volume"):
        MassiveMinuteProvider(client=FakeClient([bar])).fetch_bars(request())


def test_rejects_misaligned_timestamp() -> None:
    bar = vendor_bar(datetime(2025, 1, 2, 14, 30, 1, tzinfo=UTC))
    with pytest.raises(ProviderDataError, match="minute-aligned"):
        MassiveMinuteProvider(client=FakeClient([bar])).fetch_bars(request())


def test_rejects_invalid_prices() -> None:
    bar = vendor_bar(datetime(2025, 1, 2, 14, 30, tzinfo=UTC), high=90.0)
    with pytest.raises(ProviderDataError, match="Invalid Massive minute aggregate"):
        MassiveMinuteProvider(client=FakeClient([bar])).fetch_bars(request())


def test_rejects_duplicate_minutes() -> None:
    bar = vendor_bar(datetime(2025, 1, 2, 14, 30, tzinfo=UTC))
    with pytest.raises(ProviderDataError, match="Invalid Massive minute aggregate response"):
        MassiveMinuteProvider(client=FakeClient([bar, bar])).fetch_bars(request())


def test_empty_result_is_valid() -> None:
    assert MassiveMinuteProvider(client=FakeClient([])).fetch_bars(request()).bars == ()


def test_wraps_vendor_errors() -> None:
    class BrokenClient:
        def list_aggs(self, **kwargs: object) -> list[SimpleNamespace]:
            raise RuntimeError("vendor unavailable")

    with pytest.raises(ProviderDataError, match="request failed"):
        MassiveMinuteProvider(client=BrokenClient()).fetch_bars(request())

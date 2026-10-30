"""Offline tests for the initial Massive minute provider scaffold."""

from datetime import UTC, datetime

import pytest

from quant_research.core import TimeRange
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.providers.models import (
    BarRequest,
    InstrumentKind,
    ProviderCapabilityError,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency


class FakeClient:
    """Raise if the scaffold accidentally performs an API request."""

    def list_aggs(self, **kwargs: object) -> None:
        raise AssertionError(f"Unexpected vendor request: {kwargs}")


def make_request(frequency: BarFrequency) -> BarRequest:
    return BarRequest(
        instrument=ProviderInstrument("AAPL", InstrumentKind.TICKER, "XNAS"),
        frequency=frequency,
        period=TimeRange(
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 3, tzinfo=UTC),
        ),
    )


def test_rejects_daily_requests() -> None:
    provider = MassiveMinuteProvider(client=FakeClient())
    with pytest.raises(ProviderCapabilityError, match="minute bars only"):
        provider.fetch_bars(make_request(BarFrequency.DAILY))


def test_minute_requests_are_not_yet_implemented() -> None:
    provider = MassiveMinuteProvider(client=FakeClient())
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        provider.fetch_bars(make_request(BarFrequency.MINUTE))

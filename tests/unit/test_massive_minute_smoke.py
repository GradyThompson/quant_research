"""Offline checks for the minute-data smoke-test invariants."""

from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.core import TimeRange
from quant_research.data.providers.models import (
    BarRequest,
    BarResponse,
    InstrumentKind,
    ProviderBar,
    ProviderInstrument,
)
from quant_research.data.schemas import BarFrequency


def test_one_minute_bar_is_valid() -> None:
    # 2025-01-08 09:30-09:31 America/New_York.
    end = datetime(2025, 1, 8, 14, 31, tzinfo=UTC)
    request = BarRequest(
        instrument=ProviderInstrument("AAPL", InstrumentKind.TICKER, "XNAS"),
        frequency=BarFrequency.MINUTE,
        period=TimeRange(end - timedelta(minutes=1), end + timedelta(minutes=1)),
    )
    response = BarResponse(
        provider="test",
        request=request,
        retrieved_at=end + timedelta(minutes=2),
        bars=(
            ProviderBar(
                timestamp=end,
                available_at=end,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=100,
                currency="USD",
            ),
        ),
    )
    # Script is imported directly from its folder; no package changes needed.
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "test_massive_minute.py"
    spec = importlib.util.spec_from_file_location("minute_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_session(response, date(2025, 1, 8))


def test_empty_response_is_rejected() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "test_massive_minute.py"
    spec = importlib.util.spec_from_file_location("minute_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    end = datetime(2025, 1, 8, 14, 31, tzinfo=UTC)
    response = BarResponse(
        provider="test",
        request=BarRequest(
            instrument=ProviderInstrument("AAPL", InstrumentKind.TICKER, "XNAS"),
            frequency=BarFrequency.MINUTE,
            period=TimeRange(end - timedelta(minutes=1), end + timedelta(minutes=1)),
        ),
        retrieved_at=end,
        bars=(),
    )
    with pytest.raises(ValueError, match="No minute bars"):
        module.verify_session(response, date(2025, 1, 8))

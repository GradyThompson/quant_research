"""Tests for the research-only first-touch barrier label."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.labels.barrier import BarrierOutcome, label_barrier_event

ENTRY = datetime(2025, 1, 10, 15, 0, tzinfo=UTC)


def _bars(*ranges: tuple[float, float], gap: bool = False) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "timestamp": ENTRY + timedelta(minutes=i + (1 if gap else 0)),
                "available_at": ENTRY + timedelta(minutes=i + (1 if gap else 0), seconds=30),
                "high": high,
                "low": low,
            }
            for i, (high, low) in enumerate(ranges, start=1)
        ]
    )


def _label(bars: pa.Table, horizon: int = 2) -> BarrierOutcome:
    return label_barrier_event(
        future_bars=bars,
        entry_at=ENTRY,
        entry_price=100.0,
        upper_barrier_pct=0.01,
        lower_barrier_pct=0.01,
        max_holding_minutes=horizon,
    ).outcome


def test_upper_first() -> None:
    assert _label(_bars((100.5, 99.5), (101.0, 99.5))) == BarrierOutcome.UPPER_FIRST


def test_lower_first() -> None:
    assert _label(_bars((100.2, 98.9))) == BarrierOutcome.LOWER_FIRST


def test_ambiguous_same_minute() -> None:
    assert _label(_bars((101.1, 98.9))) == BarrierOutcome.AMBIGUOUS


def test_timeout() -> None:
    assert _label(_bars((100.5, 99.5), (100.6, 99.4))) == BarrierOutcome.TIMEOUT


def test_incomplete_data_is_not_timeout() -> None:
    with pytest.raises(ValueError, match="Insufficient future"):
        _label(_bars((100.5, 99.5)))


def test_missing_minute_rejected() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _label(_bars((100.5, 99.5), gap=True))


def test_label_availability_is_future() -> None:
    label = label_barrier_event(
        future_bars=_bars((101.0, 99.5)),
        entry_at=ENTRY,
        entry_price=100.0,
        upper_barrier_pct=0.01,
        lower_barrier_pct=0.01,
        max_holding_minutes=2,
    )
    assert label.observed_at == ENTRY + timedelta(minutes=1)
    assert label.label_available_at == ENTRY + timedelta(minutes=1, seconds=30)

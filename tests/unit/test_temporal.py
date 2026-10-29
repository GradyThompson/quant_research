from datetime import UTC, datetime, timedelta

import pytest

from quant_research.core import (
    TemporalMetadata,
    TimeRange,
)


def dt(day: int) -> datetime:
    return datetime(2025, 1, day, tzinfo=UTC)


def test_valid_time_range():
    interval = TimeRange(
        start=dt(1),
        end=dt(10),
    )

    assert interval.start == dt(1)
    assert interval.end == dt(10)


def test_time_range_requires_positive_duration():
    with pytest.raises(ValueError):
        TimeRange(
            start=dt(10),
            end=dt(1),
        )


def test_time_range_rejects_equal_boundaries():
    with pytest.raises(ValueError):
        TimeRange(
            start=dt(1),
            end=dt(1),
        )


def test_time_range_requires_timezone():
    with pytest.raises(ValueError):
        TimeRange(
            start=datetime(2025, 1, 1),
            end=dt(10),
        )


def test_time_range_contains_start():
    interval = TimeRange(dt(1), dt(10))

    assert interval.contains(dt(1))


def test_time_range_excludes_end():
    interval = TimeRange(dt(1), dt(10))

    assert not interval.contains(dt(10))


def test_adjacent_time_ranges_do_not_overlap():
    first = TimeRange(dt(1), dt(10))
    second = TimeRange(dt(10), dt(20))

    assert not first.overlaps(second)


def test_overlapping_time_ranges():
    first = TimeRange(dt(1), dt(10))
    second = TimeRange(dt(5), dt(15))

    assert first.overlaps(second)
    assert second.overlaps(first)


def test_temporal_metadata():
    metadata = TemporalMetadata(
        event_time=dt(1),
        available_at=dt(2),
        ingested_at=dt(3),
    )

    assert metadata.event_time == dt(1)
    assert metadata.available_at == dt(2)
    assert metadata.ingested_at == dt(3)


def test_temporal_metadata_rejects_invalid_order():
    with pytest.raises(ValueError):
        TemporalMetadata(
            event_time=dt(3),
            available_at=dt(2),
            ingested_at=dt(4),
        )


def test_temporal_metadata_rejects_ingestion_before_availability():
    with pytest.raises(ValueError):
        TemporalMetadata(
            event_time=dt(1),
            available_at=dt(3),
            ingested_at=dt(2),
        )


def test_observation_not_available_before_release():
    metadata = TemporalMetadata(
        event_time=dt(1),
        available_at=dt(5),
        ingested_at=dt(6),
    )

    assert not metadata.available_by(dt(4))
    assert metadata.available_by(dt(5))


def test_observation_not_recorded_before_ingestion():
    metadata = TemporalMetadata(
        event_time=dt(1),
        available_at=dt(5),
        ingested_at=dt(6),
    )

    assert not metadata.recorded_by(dt(5))
    assert metadata.recorded_by(dt(6))


def test_as_utc():
    timezone = UTC

    metadata = TemporalMetadata(
        event_time=datetime(2025, 1, 1, tzinfo=timezone),
        available_at=datetime(2025, 1, 2, tzinfo=timezone),
        ingested_at=datetime(2025, 1, 3, tzinfo=timezone),
    )

    result = metadata.as_utc()

    assert result.event_time.utcoffset() == timedelta(0)
    assert result.available_at.utcoffset() == timedelta(0)
    assert result.ingested_at.utcoffset() == timedelta(0)

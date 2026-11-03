"""Deterministic tests for publication, same-bar ambiguity and trade arithmetic."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.model.ml_barrier import (
    FEATURE_NAMES,
    BarrierRow,
    construct_barrier_row,
    row_from_json,
    row_to_json,
)


def session(price: float = 100.0) -> list[dict[str, object]]:
    start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
    bars: list[dict[str, object]] = []
    for n in range(390):
        end = start + timedelta(minutes=n + 1)
        value = price * (1 + n * 0.00002)
        bars.append(
            {
                "timestamp": end,
                "available_at": end + timedelta(minutes=1),
                "open": value,
                "high": value * 1.00004,
                "low": value * 0.99996,
                "close": value,
                "volume": 10_000,
            }
        )
    return bars


def construct(bars: list[dict[str, object]]) -> BarrierRow | None:
    return construct_barrier_row(
        day=date(2024, 1, 3),
        ticker="FOO",
        asset=bars,
        market=session(200.0),
        opening_volume_baseline=150_000,
        prior_adv=100_000_000,
    )


def test_publication_delay_and_timeout_observable() -> None:
    row = construct(session())
    assert row is not None
    assert row.decision_at.hour == 15 and row.decision_at.minute == 1
    assert row.entry_at.minute == 2 and row.entry_at > row.decision_at
    assert row.resolved_at.minute == 2 and row.resolved_at.hour == 16
    assert row.outcome == 2
    assert len(row.features) == len(FEATURE_NAMES)
    assert sum(row.baseline) == pytest.approx(1)
    assert row_from_json(row_to_json(row)) == row


def test_stop_target_same_bar_is_loss() -> None:
    bars = session()
    entry = float(bars[32]["open"])
    bars[35]["low"] = entry * 0.96
    bars[35]["high"] = entry * 1.02
    row = construct(bars)
    assert row is not None
    assert row.ambiguous and row.outcome == 1
    assert row.gross_return_bps == pytest.approx(-200)


def test_gap_through_stop_is_worse_than_stop() -> None:
    bars = session()
    entry = float(bars[32]["open"])
    bars[36]["open"] = entry * 0.97
    bars[36]["close"] = entry * 0.97
    bars[36]["low"] = entry * 0.965
    bars[36]["high"] = entry * 0.975
    row = construct(bars)
    assert row is not None
    assert row.outcome == 1
    assert row.gross_return_bps == pytest.approx(-300)


def test_future_values_cannot_change_features_or_baseline() -> None:
    bars = session()
    original = construct(bars)
    assert original is not None
    bars[40]["high"] = 999.0
    changed = construct(bars)
    assert changed is not None
    assert changed.features == original.features
    assert changed.baseline == original.baseline
    assert changed.outcome != original.outcome


def test_missing_minute_is_rejected() -> None:
    bars = session()
    bars.pop(44)
    assert construct(bars) is None


def test_no_lookahead_with_unpublished_opening_bar() -> None:
    bars = session()
    bars[29]["available_at"] = bars[29]["available_at"] + timedelta(minutes=3)
    row = construct(bars)
    assert row is None


def test_invalid_saved_row_is_rejected() -> None:
    row = construct(session())
    assert row is not None
    bad = row_to_json(row)
    bad["features"] = [0.1]
    with pytest.raises(ValueError, match="schema"):
        row_from_json(bad)

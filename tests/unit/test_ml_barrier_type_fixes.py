"""Regression coverage for Sequence-safe rows and typed saved baselines."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.model.ml_barrier import construct_barrier_row, row_from_json, row_to_json


def test_mixed_sequence_types_and_explicit_three_part_baseline() -> None:
    start = datetime(2024, 1, 3, 14, 30, tzinfo=UTC)
    bars: list[dict[str, object]] = []
    for idx in range(95):
        end = start + timedelta(minutes=idx + 1)
        bars.append(
            {
                "timestamp": end,
                "available_at": end + timedelta(minutes=1),
                "open": 100.0,
                "close": 100.0,
                "high": 100.01,
                "low": 99.99,
                "volume": 10000,
            }
        )
    row = construct_barrier_row(
        day=date(2024, 1, 3),
        ticker="FOO",
        asset=bars,
        market=tuple(bars),
        opening_volume_baseline=150000,
        prior_adv=100000000,
    )
    assert row is not None
    assert row_from_json(row_to_json(row)) == row
    wrong_baseline = row_to_json(row)
    wrong_baseline["baseline"] = [0.1, 0.9]
    with pytest.raises(ValueError, match="schema"):
        row_from_json(wrong_baseline)

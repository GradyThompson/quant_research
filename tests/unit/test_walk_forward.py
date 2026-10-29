from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.validation import WalkForwardSplitter

BASE = datetime(2025, 1, 1, 21, tzinfo=UTC)


def labels(periods: int = 12, *, horizon: int = 1) -> pa.Table:
    rows = []
    for day in range(periods):
        for sec in ("SEC-000001", "SEC-000002"):
            end = day + horizon
            rows.append(
                {
                    "security_id": sec,
                    "timestamp": BASE + timedelta(days=day),
                    "label_end_time": BASE + timedelta(days=end) if end < periods else None,
                    "label_available_at": BASE + timedelta(days=end, minutes=15)
                    if end < periods
                    else None,
                }
            )
    return pa.Table.from_pylist(rows)


def test_expanding_purges_unavailable_training_outcomes():
    data = labels()
    folds = WalkForwardSplitter(train_periods=4, test_periods=2).split(data)
    assert len(folds) == 4
    assert folds[0].test_start == BASE + timedelta(days=4)
    assert len(folds[0].test_indices) == 4
    assert len(folds[0].train_indices) == 6  # final training date has future-unavailable labels
    rows = data.to_pylist()
    for fold in folds:
        assert max(rows[i]["label_available_at"] for i in fold.train_indices) < fold.test_start
        assert max(rows[i]["timestamp"] for i in fold.train_indices) < fold.test_start


def test_rolling_window_and_gap():
    data = labels()
    folds = WalkForwardSplitter(
        train_periods=3, test_periods=2, gap_periods=1, expanding=False
    ).split(data)
    assert folds[0].train_start == BASE
    assert folds[0].train_end == BASE + timedelta(days=2)
    assert folds[0].test_start == BASE + timedelta(days=4)
    assert folds[1].train_start == BASE + timedelta(days=2)


def test_nonoverlapping_test_steps():
    folds = WalkForwardSplitter(train_periods=3, test_periods=2).split(labels(9))
    assert [f.test_start.day for f in folds] == [4, 6, 8]


def test_missing_horizon_purged_from_train():
    folds = WalkForwardSplitter(train_periods=5, test_periods=2).split(labels(10, horizon=4))
    assert folds
    assert len(folds[0].train_indices) == 2


def test_invalid_parameters():
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_periods=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(gap_periods=-1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(step_periods=0)


def test_duplicate_rows_rejected():
    data = labels()
    with pytest.raises(ValueError, match="Duplicate"):
        WalkForwardSplitter().split(pa.concat_tables([data, data.slice(0, 1)]))


def test_short_history_returns_no_folds():
    assert WalkForwardSplitter(train_periods=20).split(labels()) == ()

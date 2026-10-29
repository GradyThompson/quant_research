"""Chronological walk-forward folds with leakage-aware training-label purging."""

from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Row indices into the input label table, with chronological boundaries."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass(frozen=True, slots=True)
class WalkForwardSplitter:
    """Generate train-before-test folds by unique observation timestamps.

    `gap` excludes full observation dates immediately before each test window.
    This pre-test gap is NOT a post-test embargo. Post-test embargo is not
    applicable when training is strictly earlier than testing.

    Training rows are purged when their outcome is unavailable at or after
    the first test decision timestamp, or when the outcome is missing.
    This assumes test decisions can occur no earlier than `test_start`.
    """

    train_periods: int = 252
    test_periods: int = 21
    step_periods: int | None = None
    gap_periods: int = 0
    expanding: bool = True

    def __post_init__(self) -> None:
        values = (self.train_periods, self.test_periods, self.gap_periods)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise ValueError("period counts must be integers")
        if self.train_periods < 1 or self.test_periods < 1 or self.gap_periods < 0:
            raise ValueError("train/test must be positive; gap nonnegative")
        if self.step_periods is not None and (
            isinstance(self.step_periods, bool)
            or not isinstance(self.step_periods, int)
            or self.step_periods < 1
        ):
            raise ValueError("step_periods must be a positive integer")

    def split(self, labels: pa.Table) -> tuple[WalkForwardFold, ...]:
        required = {"security_id", "timestamp", "label_end_time", "label_available_at"}
        if not required.issubset(labels.column_names):
            raise ValueError(f"Missing columns: {sorted(required - set(labels.column_names))}")
        rows = labels.select(sorted(required)).to_pylist()
        key_set: set[tuple[str, datetime]] = set()
        for row in rows:
            if row["security_id"] is None or row["timestamp"] is None:
                raise ValueError("Null security or timestamp")
            key = (row["security_id"], row["timestamp"])
            if key in key_set:
                raise ValueError("Duplicate security/timestamp")
            key_set.add(key)
            end, available = row["label_end_time"], row["label_available_at"]
            if (end is None) != (available is None):
                raise ValueError("Incomplete label horizon metadata")
            if end is not None and (end <= row["timestamp"] or available < end):
                raise ValueError("Invalid future label timestamps")
        dates = sorted({row["timestamp"] for row in rows})
        step = self.step_periods or self.test_periods
        folds: list[WalkForwardFold] = []
        first_test = self.train_periods + self.gap_periods
        for test_pos in range(first_test, len(dates) - self.test_periods + 1, step):
            train_stop = test_pos - self.gap_periods
            train_start = 0 if self.expanding else train_stop - self.train_periods
            if train_start < 0:
                continue
            cutoff = dates[test_pos]
            train_dates = set(dates[train_start:train_stop])
            test_dates = set(dates[test_pos : test_pos + self.test_periods])
            train_indices = tuple(
                i
                for i, row in enumerate(rows)
                if row["timestamp"] in train_dates
                and row["label_available_at"] is not None
                and row["label_available_at"] < cutoff
            )
            test_indices = tuple(i for i, row in enumerate(rows) if row["timestamp"] in test_dates)
            if not train_indices or not test_indices:
                continue
            folds.append(
                WalkForwardFold(
                    train_indices=train_indices,
                    test_indices=test_indices,
                    train_start=dates[train_start],
                    train_end=dates[train_stop - 1],
                    test_start=cutoff,
                    test_end=dates[test_pos + self.test_periods - 1],
                )
            )
        return tuple(folds)

"""Tests for strategy-independent chronological multi-stage session splitting."""

from datetime import date, timedelta

import pytest

from quant_research.validation.staged_sessions import staged_session_folds


def _days(n: int) -> list[date]:
    return [date(2025, 1, 1) + timedelta(days=index) for index in range(n)]


def test_three_folds_chronological_and_purged() -> None:
    days = _days(49)
    folds = staged_session_folds(days)
    assert len(folds) == 3
    assert [fold.validation_days[0] for fold in folds] == [days[30], days[35], days[40]]
    validation: set[date] = set()
    for fold in folds:
        assert fold.baseline_fit_days == tuple(days[:19])
        assert len(fold.alpha_fit_days) == 10
        assert len(fold.embargo_days) == 1
        assert len(fold.validation_days) == 5
        assert fold.baseline_fit_days[-1] < fold.alpha_fit_days[0]
        assert fold.alpha_fit_days[-1] < fold.embargo_days[0] < fold.validation_days[0]
        assert not set(fold.validation_days) & validation
        validation.update(fold.validation_days)
        assert not set(fold.alpha_fit_days) & set(fold.validation_days)
    # In later folds, previously evaluated days can become *past* training data.
    assert days[30] in folds[1].alpha_fit_days
    assert not validation & set(days[49:])


def test_incomplete_validation_block_is_not_scored() -> None:
    folds = staged_session_folds(_days(36))
    assert len(folds) == 1
    assert len(folds[0].validation_days) == 5


def test_rejects_invalid_partitions() -> None:
    for dates in (_days(10)[::-1], [date(2025, 1, 1)] * 20):
        with pytest.raises(ValueError, match="strictly increasing"):
            staged_session_folds(dates)
    for kwargs in (
        {"validation_sessions": 5, "step_sessions": 2},
        {"baseline_fit_sessions": 0},
        {"alpha_fit_sessions": True},
        {"embargo_sessions": -1},
        {"validation_sessions": 0},
        {"step_sessions": 0},
    ):
        with pytest.raises(ValueError, match="positive integers|must not overlap"):
            staged_session_folds(_days(40), **kwargs)

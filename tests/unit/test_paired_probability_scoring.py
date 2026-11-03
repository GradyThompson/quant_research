"""Model-independent proper-score and paired sampling regression tests."""

from datetime import UTC, date, datetime, timedelta

import pytest

from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.model.paired_probability_scoring import (
    PairedProbabilityObservation,
    score_paired_probabilities,
)


def _observation(index: int, observed: int) -> PairedProbabilityObservation:
    day = date(2025, 1, 2) + timedelta(days=index)
    return PairedProbabilityObservation(
        day=day,
        decision_at=datetime(day.year, day.month, day.day, 15, tzinfo=UTC),
        observed=observed,
        base=ThreeWayForecast(0.4, 0.4, 0.2),
        intercept=ThreeWayForecast(0.4, 0.4, 0.2),
        signal=ThreeWayForecast(0.6, 0.2, 0.2),
    )


def test_probability_scoring_matches_closed_form_brier() -> None:
    result = score_paired_probabilities(
        [_observation(0, 0), _observation(1, 0)], bootstrap_replicates=100
    )
    assert result["models"]["base"]["brier"] == pytest.approx(0.56 / 3)
    assert result["models"]["signal"]["brier"] == pytest.approx(0.24 / 3)
    assert result["intercept_minus_signal_brier"] == pytest.approx(0.32 / 3)
    assert result["first_touches"] == 2
    lower, upper = result["intercept_minus_signal_95pct_session_bootstrap"]
    assert lower == pytest.approx(0.32 / 3)
    assert upper == pytest.approx(0.32 / 3)


def test_timeout_retained_in_three_class_scores_but_not_conditional_loss() -> None:
    result = score_paired_probabilities(
        [_observation(0, 2), _observation(1, 0)], bootstrap_replicates=10
    )
    assert result["scored"] == 2
    assert result["first_touches"] == 1
    assert result["models"]["signal"]["conditional_touch_log_loss"] is not None


def test_rejects_duplicate_decisions_and_invalid_outcomes() -> None:
    obs = _observation(0, 0)
    with pytest.raises(ValueError, match="duplicate"):
        score_paired_probabilities([obs, obs])
    with pytest.raises(ValueError, match="no paired"):
        score_paired_probabilities([])
    with pytest.raises(ValueError, match="positive"):
        score_paired_probabilities([obs], bootstrap_replicates=0)
    with pytest.raises(ValueError, match="outcome"):
        _observation(0, 5)

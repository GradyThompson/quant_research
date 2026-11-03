"""Mathematical offset model and strict point-in-time tests."""

from datetime import UTC, datetime, timedelta
from math import log

import pytest

from quant_research.model.conditional_direction import (
    ConditionalDirectionModel,
    ConditionalTouchExample,
    fit_conditional_direction,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
CUTOFF = START + timedelta(days=3)


def examples() -> list[ConditionalTouchExample]:
    # A positive reversal score raises the probability of an upper first touch.
    return [
        ConditionalTouchExample(
            decision_at=START + timedelta(minutes=i * 2),
            label_available_at=START + timedelta(minutes=i * 2 + 1),
            feature=(i - 50) / 25,
            baseline_upper_given_touch=0.25 if i % 2 else 0.4,
            outcome="upper" if i > 55 else "lower",
        )
        for i in range(100)
    ]


def test_asymmetric_brownian_offset_is_recovered_at_zero_correction():
    model = ConditionalDirectionModel(0, 0, 0, 1, CUTOFF, 100, "identity", True)
    decision = CUTOFF + timedelta(days=1)
    for base in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert model.predict_upper_given_touch(base, 1, decision_at=decision) == pytest.approx(base)


def test_regularized_likelihood_learns_positive_direction_and_intercept_baseline():
    full = fit_conditional_direction(examples(), train_until=CUTOFF, version="full")
    intercept = fit_conditional_direction(
        examples(), train_until=CUTOFF, version="intercept", include_feature=False
    )
    assert full.coefficient > 0
    assert intercept.coefficient == 0
    decision = CUTOFF + timedelta(days=1)
    low = full.predict_upper_given_touch(0.25, -1.5, decision_at=decision)
    high = full.predict_upper_given_touch(0.25, 1.5, decision_at=decision)
    assert high > low
    assert 0 < high < 1
    assert 0 < intercept.predict_upper_given_touch(0.25, 99, decision_at=decision) < 1


def test_training_maturity_and_prediction_chronology_are_enforced():
    bad = examples()
    bad[-1] = ConditionalTouchExample(
        decision_at=START + timedelta(days=4),
        label_available_at=START + timedelta(days=4, minutes=5),
        feature=0,
        baseline_upper_given_touch=0.5,
        outcome="upper",
    )
    with pytest.raises(ValueError, match="not known before cutoff"):
        fit_conditional_direction(bad, train_until=CUTOFF, version="bad")
    model = fit_conditional_direction(examples(), train_until=CUTOFF, version="good")
    with pytest.raises(ValueError, match="after training cutoff"):
        model.predict_upper_given_touch(0.5, 0, decision_at=CUTOFF)
    with pytest.raises(ValueError, match="strictly within"):
        model.predict_upper_given_touch(0.0, 0, decision_at=CUTOFF + timedelta(days=1))


def test_no_future_labels_in_training_data():
    late = ConditionalTouchExample(
        decision_at=START + timedelta(days=2, hours=23),
        label_available_at=CUTOFF + timedelta(seconds=1),
        feature=2,
        baseline_upper_given_touch=0.2,
        outcome="upper",
    )
    with pytest.raises(ValueError, match="not known before cutoff"):
        fit_conditional_direction(examples() + [late], train_until=CUTOFF, version="bad")


def test_nonzero_adjustment_is_log_odds_shift():
    model = ConditionalDirectionModel(log(2), 0, 0, 1, CUTOFF, 30, "shift", False)
    p = model.predict_upper_given_touch(0.25, 0, decision_at=CUTOFF + timedelta(days=1))
    # Baseline odds=1/3; offset +log(2) doubles odds to 2/3.
    assert p == pytest.approx(0.4)

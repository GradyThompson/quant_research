from datetime import UTC, datetime, timedelta

import pytest

from quant_research.model.conditional_first_passage import (
    ThreeWayForecast,
    combine_first_passage_direction,
)
from quant_research.model.directional_alpha import (
    DirectionalExample,
    fit_directional_alpha,
)
from quant_research.model.first_passage import FirstPassageProbabilities

T0 = datetime(2026, 1, 1, tzinfo=UTC)
CUTOFF = T0 + timedelta(days=100)


def _examples(n: int = 100) -> list[DirectionalExample]:
    # Deterministic synthetic data, both touched outcomes represented.
    return [
        DirectionalExample(
            decision_at=T0 + timedelta(minutes=2 * i),
            label_available_at=T0 + timedelta(minutes=2 * i + 1),
            feature=-2 + 4 * i / n,
            outcome="lower" if i < n // 2 else "upper",
        )
        for i in range(n)
    ]


def test_m3_logistic_fit_predicts_direction_and_is_reproducible():
    model = fit_directional_alpha(_examples(), train_until=CUTOFF, version="m3-test")
    assert model.training_examples == 100
    p_low = model.predict_upper_given_touch(-1, decision_at=CUTOFF + timedelta(minutes=1))
    p_high = model.predict_upper_given_touch(1, decision_at=CUTOFF + timedelta(minutes=1))
    assert 0 < p_low < 0.5 < p_high < 1
    assert model == fit_directional_alpha(_examples(), train_until=CUTOFF, version="m3-test")


def test_m3_rejects_future_or_embargo_label_even_if_feature_is_historical():
    bad = DirectionalExample(T0 + timedelta(days=99), CUTOFF + timedelta(seconds=1), 0, "upper")
    with pytest.raises(ValueError, match="not yet available"):
        fit_directional_alpha(_examples() + [bad], train_until=CUTOFF, version="v")


def test_m3_invalid_inputs_and_insufficient_classes():
    with pytest.raises(ValueError, match="insufficient"):
        fit_directional_alpha(_examples(10), train_until=CUTOFF, version="v")
    with pytest.raises(ValueError, match="both upper and lower"):
        fit_directional_alpha(
            [
                DirectionalExample(x.decision_at, x.label_available_at, x.feature, "upper")
                for x in _examples()
            ],
            train_until=CUTOFF,
            version="v",
        )
    with pytest.raises(ValueError, match="after training"):
        model = fit_directional_alpha(_examples(), train_until=CUTOFF, version="v")
        model.predict_upper_given_touch(1, decision_at=CUTOFF)
    with pytest.raises(ValueError, match="mature after"):
        DirectionalExample(T0, T0, 1, "lower")


def test_constant_feature_model_is_intercept_only():
    samples = [
        DirectionalExample(x.decision_at, x.label_available_at, 0.0, x.outcome) for x in _examples()
    ]
    model = fit_directional_alpha(samples, train_until=CUTOFF, version="v")
    assert abs(model.coefficient) < 1e-7
    assert model.predict_upper_given_touch(
        0, decision_at=CUTOFF + timedelta(seconds=1)
    ) == pytest.approx(0.5)


def test_probability_composition_preserves_touch_odds():
    base = FirstPassageProbabilities(0.2, 0.3, 0.5)
    combined = combine_first_passage_direction(base, probability_upper_given_touch=0.8)
    assert combined.upper_first == pytest.approx(0.4)
    assert combined.lower_first == pytest.approx(0.1)
    assert combined.neither == pytest.approx(0.5)
    assert combined.touch_probability == pytest.approx(0.5)


def test_probability_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="sum to one"):
        ThreeWayForecast(0.1, 0.1, 0.1)
    with pytest.raises(ValueError, match="conditional"):
        combine_first_passage_direction(
            FirstPassageProbabilities(0.2, 0.3, 0.5), probability_upper_given_touch=float("nan")
        )

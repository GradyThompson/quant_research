"""Mathematical invariants of the driftless log-Brownian first-passage model."""

from __future__ import annotations

from math import erfc, log1p, sqrt

import pytest

from quant_research.model.first_passage import (
    _long_horizon_hits,
    _short_horizon_hit,
    first_passage_probabilities,
)


def probabilities(upper: float, lower: float, horizon: int, sigma: float):
    return first_passage_probabilities(
        upper_barrier_pct=upper,
        lower_barrier_pct=lower,
        horizon_minutes=horizon,
        volatility_per_sqrt_minute=sigma,
    )


def test_probabilities_sum_to_one_across_regimes() -> None:
    for upper, lower in ((0.005, 0.005), (0.02, 0.01), (0.1, 0.001)):
        for sigma in (0.0, 0.00001, 0.0005, 0.002, 0.02):
            for horizon in (1, 15, 60, 390):
                p = probabilities(upper, lower, horizon, sigma)
                assert 0.0 <= p.upper_first <= 1.0
                assert 0.0 <= p.lower_first <= 1.0
                assert 0.0 <= p.neither <= 1.0
                assert sum((p.upper_first, p.lower_first, p.neither)) == pytest.approx(
                    1.0, abs=1e-14
                )


def test_symmetric_log_barriers_give_equal_hitting_probabilities() -> None:
    # Equal percentage moves are not log symmetric. Pair barriers such that
    # log(1 + upper) = -log(1 - lower).
    upper = 0.02
    lower = upper / (1.0 + upper)
    for horizon in (1, 30, 60, 360):
        p = probabilities(upper, lower, horizon, 0.002)
        assert p.upper_first == pytest.approx(p.lower_first, abs=1e-13)


def test_eventual_upper_hitting_probability_matches_gambler_ruin() -> None:
    upper, lower = 0.02, 0.01
    p = probabilities(upper, lower, 100_000, 0.005)
    exact_upper = -log1p(-lower) / (log1p(upper) - log1p(-lower))
    assert p.upper_first == pytest.approx(exact_upper, abs=1e-12)
    assert p.lower_first == pytest.approx(1 - exact_upper, abs=1e-12)
    assert p.neither == pytest.approx(0.0, abs=1e-12)


def test_short_horizon_matches_one_sided_reflection_principle() -> None:
    # The opposite barrier is too remote to change first-touch probability.
    upper, lower, sigma, horizon = 0.005, 0.7, 0.001, 5
    p = probabilities(upper, lower, horizon, sigma)
    one_sided = erfc(log1p(upper) / (sigma * sqrt(2 * horizon)))
    assert p.upper_first == pytest.approx(one_sided, abs=1e-12)
    assert p.lower_first == pytest.approx(0.0, abs=1e-12)


def test_time_monotonicity() -> None:
    earlier = probabilities(0.02, 0.01, 20, 0.002)
    later = probabilities(0.02, 0.01, 120, 0.002)
    assert later.upper_first >= earlier.upper_first
    assert later.lower_first >= earlier.lower_first
    assert later.neither <= earlier.neither


def test_volatility_and_time_scale_equivalence() -> None:
    # Brownian scaling: doubling sigma equals quadrupling the time horizon.
    a = probabilities(0.02, 0.01, 60, 0.002)
    b = probabilities(0.02, 0.01, 240, 0.001)
    assert a == pytest.approx(b, abs=1e-14)


def test_short_and_long_series_agree_near_switch() -> None:
    for upper, lower in ((0.02, 0.01), (0.005, 0.005), (0.1, 0.001)):
        upper_distance = log1p(upper)
        lower_distance = -log1p(-lower)
        width = upper_distance + lower_distance
        for normalized_time in (0.03, 0.1, 0.249, 0.25, 0.4):
            t = normalized_time * width * width
            short_upper = _short_horizon_hit(upper_distance, width, t)
            short_lower = _short_horizon_hit(lower_distance, width, t)
            long_upper, long_lower = _long_horizon_hits(lower_distance, width, t)
            assert short_upper == pytest.approx(long_upper, abs=1e-12)
            assert short_lower == pytest.approx(long_lower, abs=1e-12)


def test_zero_volatility_means_neither() -> None:
    assert probabilities(0.02, 0.01, 60, 0.0).neither == 1.0


def test_extremely_small_scales_do_not_underflow_before_normalization() -> None:
    # Width and volatility both far below float sqrt(min). Rescaling by
    # barrier width must occur before squaring to retain Brownian scaling.
    p = probabilities(1e-160, 1e-160, 1, 1e-160)
    assert p.upper_first > 0.0
    assert p.lower_first > 0.0
    assert p.neither > 0.0


def test_tiny_volatility_exits_neither() -> None:
    assert probabilities(0.02, 0.01, 60, 1e-300).neither == 1.0


@pytest.mark.parametrize(
    ("upper", "lower", "horizon", "sigma"),
    [
        (0.0, 0.01, 10, 0.002),
        (0.01, 0.0, 10, 0.002),
        (-0.01, 0.01, 10, 0.002),
        (1.0, 0.01, 10, 0.002),
        (0.01, 1.0, 10, 0.002),
        (float("nan"), 0.01, 10, 0.002),
        (0.01, 0.01, 0, 0.002),
        (0.01, 0.01, True, 0.002),
        (0.01, 0.01, 10, -0.002),
        (0.01, 0.01, 10, float("inf")),
    ],
)
def test_invalid_parameters_are_rejected(
    upper: float, lower: float, horizon: int, sigma: float
) -> None:
    with pytest.raises(ValueError):
        probabilities(upper, lower, horizon, sigma)

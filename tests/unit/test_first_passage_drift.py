"""Analytical checks for the zero-*price*-drift diffusion option."""

from math import erfc, exp, log1p, sqrt

import pytest

from quant_research.model.first_passage import first_passage_probabilities


def predict(upper: float, lower: float, minutes: int, sigma: float, mode: str = "price_zero"):
    return first_passage_probabilities(
        upper_barrier_pct=upper,
        lower_barrier_pct=lower,
        horizon_minutes=minutes,
        volatility_per_sqrt_minute=sigma,
        drift_mode=mode,  # type: ignore[arg-type]
    )


def test_eventual_hitting_probability_is_price_gambler_ruin() -> None:
    for upper, lower in ((0.02, 0.01), (0.005, 0.005), (0.10, 0.01)):
        p = predict(upper, lower, 1_000_000, 0.002)
        assert p.upper_first == pytest.approx(lower / (upper + lower), abs=1e-12)
        assert p.lower_first == pytest.approx(upper / (upper + lower), abs=1e-12)
        assert p.neither == pytest.approx(0.0, abs=1e-12)


def test_one_sided_hitting_cdf_under_price_zero_drift() -> None:
    upper, lower, sigma, horizon = 0.005, 0.7, 0.002, 20
    b = log1p(upper)
    mu = -0.5 * sigma * sigma
    z = sigma * sqrt(horizon)
    # One-sided Brownian hitting distribution with general constant log drift.
    expected = 0.5 * erfc((b - mu * horizon) / (z * sqrt(2))) + (
        exp(2 * mu * b / (sigma * sigma)) * 0.5 * erfc((b + mu * horizon) / (z * sqrt(2)))
    )
    p = predict(upper, lower, horizon, sigma)
    assert p.upper_first == pytest.approx(expected, abs=1e-12)
    assert p.lower_first == pytest.approx(0.0, abs=1e-12)


def test_short_and_long_price_zero_modes_are_continuous() -> None:
    for up, down in ((0.02, 0.01), (0.005, 0.005), (0.10, 0.001)):
        width = log1p(up) - log1p(-down)
        sigma1 = width * sqrt(0.249999)
        sigma2 = width * sqrt(0.250001)
        short = predict(up, down, 1, sigma1)
        long = predict(up, down, 1, sigma2)
        assert short.upper_first == pytest.approx(long.upper_first, abs=2e-6)
        assert short.lower_first == pytest.approx(long.lower_first, abs=2e-6)


def test_price_zero_and_log_zero_are_different_but_both_normalized() -> None:
    price = predict(0.02, 0.01, 120, 0.0015)
    original = predict(0.02, 0.01, 120, 0.0015, "log_zero")
    assert price.upper_first < original.upper_first
    assert price.lower_first > original.lower_first
    for p in (price, original):
        assert 0 <= p.upper_first <= 1
        assert 0 <= p.lower_first <= 1
        assert 0 <= p.neither <= 1
        assert p.upper_first + p.lower_first + p.neither == pytest.approx(1.0, abs=1e-14)


def test_zero_volatility_is_neither_in_both_modes() -> None:
    assert predict(0.02, 0.01, 60, 0).neither == 1
    assert predict(0.02, 0.01, 60, 0, "log_zero").neither == 1


def test_invalid_drift_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="drift_mode"):
        predict(0.02, 0.01, 60, 0.001, "unexpected")

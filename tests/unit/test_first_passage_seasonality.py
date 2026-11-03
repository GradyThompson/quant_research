"""Point-in-time intraday variance seasonality for mathematical first-passage."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import exp, isfinite

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_diagnostics import analyze_first_passage
from quant_research.model.first_passage_seasonality import (
    IntradayVarianceProfile,
    fit_intraday_profile,
)
from quant_research.model.first_passage_validation import MinuteBar, Session, evaluate_sessions


def _session(day: date, *, open_sigma: float, mid_sigma: float, close_sigma: float) -> Session:
    # No boundary returns across sessions, and all session bars have valid OHLC.
    start = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)
    rows: list[MinuteBar] = []
    price = 100.0
    for i in range(390):
        sigma = open_sigma if i < 60 else mid_sigma if i < 300 else close_sigma
        price *= exp(sigma if i % 2 == 0 else -sigma)
        stamp = start + timedelta(minutes=i + 1)
        rows.append(
            MinuteBar(stamp, stamp, price, price * exp(sigma / 2), price * exp(-sigma / 2), price)
        )
    return Session(day, tuple(rows))


def test_profile_has_three_periods_and_shrinkage() -> None:
    days = [
        _session(date(2025, 1, i + 1), open_sigma=0.002, mid_sigma=0.0004, close_sigma=0.001)
        for i in range(12)
    ]
    profile = fit_intraday_profile(days, method="close")
    first, middle, last = profile.factors
    assert profile.session_count == 12
    assert first > last > middle > 0
    assert profile.observation_counts == (12 * 59, 12 * 240, 12 * 90)
    assert profile.mean_future_factor(decision_index=57, horizon=5) == pytest.approx(
        (2 * first + 3 * middle) / 5
    )
    assert profile.mean_future_factor(decision_index=298, horizon=5) == pytest.approx(
        (1 * middle + 4 * last) / 5
    )
    assert fit_intraday_profile(days, method="parkinson").factors[0] > middle


def test_profile_refuses_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="earlier historical"):
        fit_intraday_profile([], method="close")
    with pytest.raises(ValueError, match="session"):
        IntradayVarianceProfile((1.0, 1.0, 1.0), 1, (1, 1, 1)).mean_future_factor(
            decision_index=389, horizon=5
        )
    with pytest.raises(ValueError, match="shrink"):
        fit_intraday_profile([], method="close", shrink_observations=-1)


def test_seasonal_forecasts_share_labels_and_retain_baseline() -> None:
    training = [
        _session(date(2024, 12, i + 1), open_sigma=0.002, mid_sigma=0.0003, close_sigma=0.001)
        for i in range(10)
    ]
    eval_day = date(2025, 1, 15)
    eval_session = _session(eval_day, open_sigma=0.002, mid_sigma=0.0003, close_sigma=0.001)
    scenario = BarrierScenario(0.02, 0.02, 5)
    kwargs = dict(
        sessions=[*training, eval_session],
        evaluation_days={eval_day},
        scenarios=[scenario],
        drift_modes=("price_zero",),
        per_scenario_stride=True,
        lookback_returns=120,
        min_returns=20,
    )
    baseline, _ = evaluate_sessions(volatility_methods=("close",), **kwargs)
    both, _ = evaluate_sessions(volatility_methods=("close", "seasonal_close"), **kwargs)
    unchanged = [r for r in both if r.volatility_method == "close"]
    adjusted = [r for r in both if r.volatility_method == "seasonal_close"]
    assert baseline == unchanged
    assert len(unchanged) == len(adjusted) > 10
    assert [r.observed for r in unchanged] == [r.observed for r in adjusted]
    assert all(isfinite(r.volatility) for r in adjusted)
    early_pairs = [
        (a, b)
        for a, b in zip(unchanged, adjusted, strict=True)
        if a.decision_bar_index is not None and a.decision_bar_index < 30
    ]
    mid_pairs = [
        (a, b)
        for a, b in zip(unchanged, adjusted, strict=True)
        if a.decision_bar_index is not None and 150 < a.decision_bar_index < 250
    ]
    assert early_pairs and mid_pairs
    assert any(b.volatility > a.volatility for a, b in early_pairs)
    assert any(b.volatility < a.volatility for a, b in mid_pairs)
    comparisons = analyze_first_passage(both, bootstrap_replicates=100)
    assert len(comparisons["paired_seasonality"]) == 1
    assert comparisons["paired_seasonality"][0]["scored_pairs"] == len(baseline)


def test_seasonal_predictions_do_not_fit_on_validation_data() -> None:
    training = [
        _session(date(2024, 12, i + 1), open_sigma=0.002, mid_sigma=0.0003, close_sigma=0.001)
        for i in range(5)
    ]
    day = date(2025, 1, 15)
    original = _session(day, open_sigma=0.002, mid_sigma=0.0003, close_sigma=0.001)
    altered = _session(day, open_sigma=0.002, mid_sigma=0.004, close_sigma=0.001)
    kwargs = dict(
        evaluation_days={day},
        scenarios=[BarrierScenario(0.02, 0.02, 5)],
        volatility_methods=("seasonal_close",),
        drift_modes=("price_zero",),
        per_scenario_stride=True,
        lookback_returns=60,
        min_returns=10,
    )
    before, _ = evaluate_sessions(sessions=[*training, original], **kwargs)
    after, _ = evaluate_sessions(sessions=[*training, altered], **kwargs)
    first = [r for r in before if r.decision_bar_index is not None and r.decision_bar_index < 30]
    second = [r for r in after if r.decision_bar_index is not None and r.decision_bar_index < 30]
    assert first and len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.volatility == pytest.approx(b.volatility)
        assert a.upper_probability == pytest.approx(b.upper_probability)
        assert a.lower_probability == pytest.approx(b.lower_probability)
    with pytest.raises(ValueError, match="earlier historical"):
        evaluate_sessions(sessions=[original], **kwargs)

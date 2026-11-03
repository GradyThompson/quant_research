"""Causal volatility estimators and non-overlapping scenario decision grids."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import exp, log, sqrt

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import (
    MinuteBar,
    RollingMinuteVolatility,
    Session,
    evaluate_sessions,
    summarize_forecasts,
)

START = datetime(2025, 1, 10, 15, tzinfo=UTC)
DAY = date(2025, 1, 10)


def bar(index: int, *, close: float = 100.0, excursion: float = 0.0) -> MinuteBar:
    stamp = START + timedelta(minutes=index)
    return MinuteBar(
        timestamp=stamp,
        available_at=stamp,
        open=close,
        high=close * exp(excursion),
        low=close * exp(-excursion),
        close=close,
    )


def test_parkinson_uses_completed_intraminute_range_not_close_returns() -> None:
    close = RollingMinuteVolatility(lookback_returns=4, min_returns=2)
    range_based = RollingMinuteVolatility(lookback_returns=4, min_returns=2, method="parkinson")
    excursion = 0.004
    for index in range(1, 6):
        observation = bar(index, excursion=excursion)
        close.observe(observation)
        range_based.observe(observation)
    assert close.estimate() == 0.0
    assert range_based.estimate() == pytest.approx(excursion / sqrt(log(2.0)))


def test_parkinson_excludes_overnight_ranges() -> None:
    estimator = RollingMinuteVolatility(lookback_returns=3, min_returns=2, method="parkinson")
    for index in (1, 2, 3):
        estimator.observe(bar(index, excursion=0.003))
    before = estimator.estimate()
    assert before is not None
    overnight = MinuteBar(
        timestamp=START + timedelta(days=1),
        available_at=START + timedelta(days=1),
        open=100.0,
        high=1000.0,
        low=50.0,
        close=100.0,
    )
    estimator.observe(overnight)
    assert estimator.estimate() == pytest.approx(before)


def test_scenario_stride_increases_short_horizon_coverage_without_overlap() -> None:
    session = Session(DAY, tuple(bar(index) for index in range(1, 48)))
    short = BarrierScenario(0.005, 0.005, 3)
    long = BarrierScenario(0.02, 0.01, 8)
    common = dict(
        sessions=[session],
        evaluation_days={DAY},
        scenarios=[short, long],
        drift_modes=("log_zero",),
        lookback_returns=10,
        min_returns=2,
        decision_stride=12,
    )
    original, _ = evaluate_sessions(**common)
    per_scenario, _ = evaluate_sessions(**common, per_scenario_stride=True)
    original_short = [row for row in original if row.scenario == short]
    short_rows = [row for row in per_scenario if row.scenario == short]
    long_rows = [row for row in per_scenario if row.scenario == long]
    assert len(short_rows) > len(original_short)
    for rows, horizon in ((short_rows, 3), (long_rows, 8)):
        assert all(
            (b.decision_at - a.decision_at).total_seconds() >= 60 * horizon
            for a, b in zip(rows, rows[1:], strict=False)
        )
    assert len({row.decision_at for row in short_rows}) == len(short_rows)


def test_volatility_methods_compare_identical_outcomes_without_future_leakage() -> None:
    sample = tuple(
        bar(index, close=100 * exp(0.002 * (index % 2)), excursion=0.003) for index in range(1, 40)
    )
    scenario = BarrierScenario(0.005, 0.005, 4)
    rows, _ = evaluate_sessions(
        sessions=[Session(DAY, sample)],
        evaluation_days={DAY},
        scenarios=[scenario],
        drift_modes=("log_zero",),
        volatility_methods=("close", "parkinson"),
        decision_stride=7,
        lookback_returns=10,
        min_returns=2,
    )
    close_rows = [row for row in rows if row.volatility_method == "close"]
    range_rows = [row for row in rows if row.volatility_method == "parkinson"]
    assert close_rows and len(close_rows) == len(range_rows)
    assert [(row.decision_at, row.observed) for row in close_rows] == [
        (row.decision_at, row.observed) for row in range_rows
    ]
    assert any(a.volatility != b.volatility for a, b in zip(close_rows, range_rows, strict=True))
    summaries = summarize_forecasts(rows)
    assert {s["volatility_method"] for s in summaries} == {"close", "parkinson"}
    assert all(s["scored"] == len(close_rows) for s in summaries)


def test_volatility_method_validation() -> None:
    with pytest.raises(ValueError, match="Unknown volatility"):
        RollingMinuteVolatility(method="not_a_method")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_sessions(
            sessions=[],
            evaluation_days={DAY},
            scenarios=[BarrierScenario(0.01, 0.01, 5)],
            volatility_methods=("close", "close"),
        )

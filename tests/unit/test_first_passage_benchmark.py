"""Training-only frequency benchmark, class calibration, and embargo isolation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import exp

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_benchmark import (
    benchmark_scores,
    fit_historical_frequency_benchmark,
)
from quant_research.model.first_passage_diagnostics import analyze_first_passage
from quant_research.model.first_passage_validation import (
    ForecastResult,
    MinuteBar,
    Session,
    evaluate_sessions,
    label_first_touch,
)

SCENARIO = BarrierScenario(0.01, 0.01, 2)


def _session(day: date, *, move: float = 0.001, opening_spike: bool = False) -> Session:
    start = datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC)
    bars: list[MinuteBar] = []
    for index in range(390):
        price = 100.0 * exp(move * (index % 2))
        stamp = start + timedelta(minutes=index + 1)
        upper = price * (1.015 if opening_spike and index < 60 else 1.001)
        lower = price * (0.985 if opening_spike and index < 60 else 0.999)
        bars.append(MinuteBar(stamp, stamp, price, upper, lower, price))
    return Session(day, tuple(bars))


def test_training_frequency_benchmark_has_normalized_nonzero_probabilities() -> None:
    train = [_session(date(2025, 1, i + 1)) for i in range(3)]
    baseline = fit_historical_frequency_benchmark(
        sessions=train,
        scenarios=[SCENARIO],
        decision_stride=4,
        per_scenario_stride=False,
    )
    for index in (0, 62, 300, 389):
        probs = baseline.probabilities(SCENARIO, index)
        assert sum(probs) == pytest.approx(1.0)
        assert all(0 < p < 1 for p in probs)
    assert sum(baseline.pooled_counts[SCENARIO]) > 0
    assert baseline.training_days == {day.day for day in train}


def test_training_baseline_does_not_depend_on_validation_or_embargo() -> None:
    training = [_session(date(2024, 12, 1)), _session(date(2024, 12, 2))]
    baseline = fit_historical_frequency_benchmark(
        sessions=training,
        scenarios=[SCENARIO],
        decision_stride=3,
        per_scenario_stride=True,
    )
    embargo = _session(date(2024, 12, 3), opening_spike=True)
    val = _session(date(2024, 12, 4), opening_spike=True)
    results, _ = evaluate_sessions(
        sessions=[*training, embargo, val],
        evaluation_days={val.day},
        scenarios=[SCENARIO],
        drift_modes=("price_zero",),
        volatility_methods=("close",),
        lookback_returns=30,
        min_returns=5,
        per_scenario_stride=True,
    )
    scores = benchmark_scores(results, baseline)
    assert scores[0]["training_sessions"] == 2
    assert scores[0]["training_observations"] == sum(baseline.pooled_counts[SCENARIO])
    assert scores[0]["frequency_brier"] >= 0
    assert scores[0]["model_brier"] >= 0
    assert scores[0]["brier_skill_vs_frequency"] is not None
    # Same historical training set => same fitted probabilities regardless
    # of what happened in embargo and validation.
    assert baseline.probabilities(SCENARIO, 0) == baseline.probabilities(SCENARIO, 0)
    with pytest.raises(ValueError, match="Training decisions"):
        benchmark_scores(
            [
                ForecastResult(
                    train_day,
                    SCENARIO,
                    "price_zero",
                    training[0].bars[0].available_at,
                    0.001,
                    0.2,
                    0.3,
                    0.5,
                    2,
                    decision_bar_index=0,
                )
                for train_day in (training[0].day,)
            ],
            baseline,
        )


def test_embargo_is_not_used_to_fit_seasonality() -> None:
    training = [_session(date(2024, 12, i + 1), move=0.001, opening_spike=True) for i in range(3)]
    embargo_day = date(2024, 12, 4)
    evaluation_day = date(2024, 12, 5)
    kwargs = dict(
        evaluation_days={evaluation_day},
        scenarios=[SCENARIO],
        drift_modes=("price_zero",),
        volatility_methods=("seasonal_close",),
        lookback_returns=30,
        min_returns=5,
        decision_stride=61,
        seasonality_training_days={row.day for row in training},
    )
    normal, _ = evaluate_sessions(
        sessions=[
            *training,
            _session(embargo_day, move=0.001),
            _session(evaluation_day, move=0.001),
        ],
        **kwargs,
    )
    distorted, _ = evaluate_sessions(
        sessions=[
            *training,
            _session(embargo_day, move=0.02, opening_spike=True),
            _session(evaluation_day, move=0.001),
        ],
        **kwargs,
    )

    def late(rows: list[ForecastResult]) -> list[ForecastResult]:
        return [
            row
            for row in rows
            if row.decision_bar_index is not None and row.decision_bar_index >= 61
        ]

    assert late(normal)
    for a, b in zip(late(normal), late(distorted), strict=True):
        assert a.volatility == pytest.approx(b.volatility)
        assert a.upper_probability == pytest.approx(b.upper_probability)
    with pytest.raises(ValueError, match="strictly pre-validation"):
        evaluate_sessions(
            sessions=[*training, _session(evaluation_day)],
            seasonality_training_days={evaluation_day},
            **{key: value for key, value in kwargs.items() if key != "seasonality_training_days"},
        )


def test_three_class_calibration_and_session_intervals() -> None:
    day1, day2 = date(2025, 1, 1), date(2025, 1, 2)
    rows = [
        ForecastResult(
            day1,
            SCENARIO,
            "price_zero",
            datetime(2025, 1, 1, 15, tzinfo=UTC),
            0.01,
            0.2,
            0.3,
            0.5,
            0,
            decision_bar_index=6,
        ),
        ForecastResult(
            day2,
            SCENARIO,
            "price_zero",
            datetime(2025, 1, 2, 15, tzinfo=UTC),
            0.01,
            0.2,
            0.3,
            0.5,
            2,
            decision_bar_index=6,
        ),
    ]
    analysis = analyze_first_passage(rows, bootstrap_replicates=100)
    classes = {entry["outcome"]: entry for entry in analysis["class_calibration"]}
    assert set(classes) == {"upper", "lower", "neither"}
    assert classes["upper"]["observed_fraction"] == pytest.approx(0.5)
    assert classes["upper"]["predicted_fraction"] == pytest.approx(0.2)
    assert classes["lower"]["observed_fraction"] == pytest.approx(0)
    assert classes["lower"]["predicted_fraction"] == pytest.approx(0.3)
    assert classes["neither"]["observed_fraction"] == pytest.approx(0.5)
    assert all(entry["gap_95pct_session_bootstrap"] is not None for entry in classes.values())


def test_first_touch_explicit_opening_order_and_ambiguous_range() -> None:
    day = date(2025, 1, 10)
    s = _session(day)
    decision = s.bars[0].available_at
    future = list(s.bars)
    stamp = future[1].timestamp
    future[1] = MinuteBar(stamp, stamp, 102, 104, 98, 100)
    assert label_first_touch(
        future, reference_price=100, decision_at=decision, scenario=SCENARIO
    ) == (0, None)
    future[1] = MinuteBar(stamp, stamp, 98, 104, 97, 100)
    assert label_first_touch(
        future, reference_price=100, decision_at=decision, scenario=SCENARIO
    ) == (1, None)
    future[1] = MinuteBar(stamp, stamp, 100, 104, 97, 100)
    assert label_first_touch(
        future, reference_price=100, decision_at=decision, scenario=SCENARIO
    ) == (None, "ambiguous_barriers")


def test_baseline_rejects_invalid_settings() -> None:
    training = [_session(date(2025, 1, 1))]
    args = dict(
        sessions=training, scenarios=[SCENARIO], decision_stride=3, per_scenario_stride=False
    )
    with pytest.raises(ValueError, match="finite"):
        fit_historical_frequency_benchmark(**args, period_prior_strength=float("nan"))
    with pytest.raises(ValueError, match="Scenarios"):
        fit_historical_frequency_benchmark(**{**args, "scenarios": []})
    with pytest.raises(ValueError, match="increasing"):
        fit_historical_frequency_benchmark(**{**args, "sessions": [*training, *training]})


def test_frequency_benchmark_rejects_truly_missing_training_outcomes() -> None:
    """A missing fit must not silently become a made-up frequency forecast."""
    session = _session(date(2025, 1, 6))
    with pytest.raises(ValueError, match="No unambiguous training outcomes"):
        fit_historical_frequency_benchmark(
            sessions=[Session(session.day, session.bars[:1])],
            scenarios=[SCENARIO],
            decision_stride=3,
            per_scenario_stride=False,
        )

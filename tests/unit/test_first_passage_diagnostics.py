"""Diagnostics for mathematical first-passage forecasts; no market data needed."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_diagnostics import analyze_first_passage
from quant_research.model.first_passage_validation import ForecastResult

SCENARIO = BarrierScenario(0.005, 0.005, 15)
START = datetime(2024, 12, 1, 15, 0, tzinfo=timezone.utc)


def forecast(
    *,
    day: int,
    minute: int,
    label: int,
    method: str = "close",
    upper: float = 0.15,
    lower: float = 0.15,
) -> ForecastResult:
    return ForecastResult(
        day=date(2024, 12, day),
        scenario=SCENARIO,
        drift_mode="price_zero",
        decision_at=START + timedelta(days=day - 1, minutes=minute),
        volatility=0.001,
        upper_probability=upper,
        lower_probability=lower,
        neither_probability=1.0 - upper - lower,
        observed=label,
        volatility_method=method,  # type: ignore[arg-type]
    )


def test_calibration_expected_touch_counts_and_neither_baseline() -> None:
    rows = [
        forecast(day=1, minute=0, label=0),
        forecast(day=1, minute=20, label=2),
        forecast(day=2, minute=0, label=1),
        forecast(day=2, minute=20, label=2),
    ]
    report = analyze_first_passage(rows, bootstrap_replicates=100)
    result = report["calibration"][0]
    assert result["scored"] == 4
    assert result["scored_sessions"] == 2
    assert result["observed_touches"] == 2
    assert result["expected_touches"] == pytest.approx(1.2)
    assert result["observed_minus_expected_touch_rate"] == pytest.approx(0.2)
    assert result["always_neither_brier"] == pytest.approx(1.0 / 3.0)
    assert result["brier_skill_vs_neither"] is not None
    assert len(result["touches_by_session"]) == 2
    assert result["touch_rate_gap_95pct_session_bootstrap"] == pytest.approx([0.2, 0.2])


def test_neither_baseline_undefined_when_no_touches() -> None:
    rows = [forecast(day=1, minute=0, label=2), forecast(day=2, minute=0, label=2)]
    report = analyze_first_passage(rows, bootstrap_replicates=100)
    assert report["calibration"][0]["brier_skill_vs_neither"] is None


def test_paired_comparison_prefers_more_accurate_probabilities() -> None:
    rows = [
        forecast(day=1, minute=0, label=0, method="close", upper=0.15),
        forecast(day=1, minute=0, label=0, method="parkinson", upper=0.60, lower=0.10),
        forecast(day=2, minute=0, label=1, method="close", lower=0.15),
        forecast(day=2, minute=0, label=1, method="parkinson", lower=0.60, upper=0.10),
    ]
    report = analyze_first_passage(rows, bootstrap_replicates=100)
    paired = report["paired_volatility"]
    assert len(paired) == 1
    assert paired[0]["scored_pairs"] == 2
    assert paired[0]["close_minus_parkinson_brier"] > 0
    interval = paired[0]["close_minus_parkinson_brier_95pct_session_bootstrap"]
    assert interval is not None
    assert interval[0] > 0
    assert report == analyze_first_passage(rows, bootstrap_replicates=100)


def test_mismatched_decisions_rejected() -> None:
    rows = [
        forecast(day=1, minute=0, label=2, method="close"),
        forecast(day=1, minute=20, label=2, method="parkinson"),
    ]
    with pytest.raises(ValueError, match="identical decisions"):
        analyze_first_passage(rows, bootstrap_replicates=100)


def test_inconsistent_labels_rejected() -> None:
    rows = [
        forecast(day=1, minute=0, label=2, method="close"),
        forecast(day=1, minute=0, label=1, method="parkinson"),
    ]
    with pytest.raises(ValueError, match="Different observed labels"):
        analyze_first_passage(rows, bootstrap_replicates=100)


def test_duplicate_decisions_rejected() -> None:
    rows = [
        forecast(day=1, minute=0, label=2, method="close"),
        forecast(day=1, minute=0, label=2, method="close"),
        forecast(day=1, minute=0, label=2, method="parkinson"),
    ]
    with pytest.raises(ValueError, match="Duplicate decision"):
        analyze_first_passage(rows, bootstrap_replicates=100)


def test_one_session_has_no_resampling_interval() -> None:
    report = analyze_first_passage([forecast(day=1, minute=0, label=2)], bootstrap_replicates=100)
    assert report["calibration"][0]["touch_rate_gap_95pct_session_bootstrap"] is None


def test_missing_comparison_method_is_allowed() -> None:
    report = analyze_first_passage([forecast(day=1, minute=0, label=2)], bootstrap_replicates=100)
    assert report["paired_volatility"] == []
    assert analyze_first_passage([], bootstrap_replicates=100)["calibration"] == []


def test_bootstrap_configuration_validation() -> None:
    with pytest.raises(ValueError, match="replicates"):
        analyze_first_passage([], bootstrap_replicates=10)
    with pytest.raises(ValueError, match="seed"):
        analyze_first_passage([], bootstrap_seed=1.2)  # type: ignore[arg-type]

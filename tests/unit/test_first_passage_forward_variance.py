"""Ex-post volatility diagnostics without changes to first-passage forecasts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import exp

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_diagnostics import analyze_first_passage
from quant_research.model.first_passage_validation import (
    ForecastResult,
    MinuteBar,
    Session,
    _forward_variances,
    evaluate_sessions,
)

DAY = date(2025, 1, 10)
START = datetime(2025, 1, 10, 15, tzinfo=UTC)
SCENARIO = BarrierScenario(0.05, 0.05, 3)


def _bar(index: int, close: float) -> MinuteBar:
    stamp = START + timedelta(minutes=index)
    return MinuteBar(
        stamp,
        stamp,
        close,
        close * exp(0.001),
        close * exp(-0.001),
        close,
    )


def test_forward_variances_are_expected_square_returns() -> None:
    rows = tuple(_bar(i, 100.0 * exp(0.001 * i)) for i in range(12))
    close_variance, range_variance = _forward_variances(
        rows, decision_index=3, horizon_minutes=3
    ) or (None, None)
    assert close_variance == pytest.approx(0.001**2)
    assert range_variance is not None
    assert range_variance > 0.0


def test_forward_variances_refuse_missing_minute_and_late_publication() -> None:
    rows = tuple(_bar(i, 100.0) for i in (0, 1, 2, 4, 5, 6))
    assert _forward_variances(rows, decision_index=1, horizon_minutes=3) is None
    late = list(_bar(i, 100.0) for i in range(6))
    original = late[1]
    late[1] = MinuteBar(
        original.timestamp,
        original.available_at + timedelta(seconds=1),
        original.open,
        original.high,
        original.low,
        original.close,
    )
    assert _forward_variances(late, decision_index=1, horizon_minutes=3) is None


def test_future_volatility_never_influences_forecasts() -> None:
    before = [_bar(i, 100.0 * exp(0.001 * (i % 2))) for i in range(12)]
    after_a = [_bar(i, 100.0) for i in range(12, 32)]
    after_b = [_bar(i, 100.0 * exp(0.006 * (i % 2))) for i in range(12, 32)]

    def evaluate(after: list[MinuteBar]) -> list[ForecastResult]:
        forecasts, _ = evaluate_sessions(
            sessions=[Session(DAY, tuple(before + after))],
            evaluation_days={DAY},
            scenarios=[SCENARIO],
            drift_modes=("price_zero",),
            decision_stride=10,
            lookback_returns=5,
            min_returns=2,
        )
        return forecasts

    first = evaluate(after_a)
    second = evaluate(after_b)
    # Decision index 10 is before the modified future bars.
    a = next(row for row in first if row.decision_bar_index == 10)
    b = next(row for row in second if row.decision_bar_index == 10)
    assert a.upper_probability == pytest.approx(b.upper_probability)
    assert a.lower_probability == pytest.approx(b.lower_probability)
    assert a.neither_probability == pytest.approx(b.neither_probability)
    assert a.forward_close_variance != b.forward_close_variance


def test_variance_diagnostics_match_predicted_and_future_variance() -> None:
    scenario = BarrierScenario(0.002, 0.002, 5)
    sample = []
    for index, day in ((5, date(2025, 1, 6)), (101, date(2025, 1, 7)), (312, date(2025, 1, 8))):
        sample.append(
            ForecastResult(
                day=day,
                scenario=scenario,
                drift_mode="price_zero",
                decision_at=START + timedelta(minutes=index),
                volatility=0.002,
                upper_probability=0.15,
                lower_probability=0.15,
                neither_probability=0.70,
                observed=2,
                volatility_method="close",
                forward_close_variance=0.001**2,
                forward_range_variance=0.0015**2,
                decision_bar_index=index,
            )
        )
    result = analyze_first_passage(sample, bootstrap_replicates=100)
    variance = result["forward_variance"][0]
    assert variance["without_forward_variance"] == 0
    overall = variance["overall"]
    assert overall["scored"] == 3
    assert overall["predicted_sigma"] == pytest.approx(0.002)
    assert overall["realized_sigma"] == pytest.approx(0.001)
    assert overall["predicted_to_realized_variance"] == pytest.approx(4.0)
    assert len(variance["by_session_period"]) == 3


def test_legacy_rows_do_not_require_realized_variance() -> None:
    result = analyze_first_passage(
        [
            ForecastResult(
                day=DAY,
                scenario=SCENARIO,
                drift_mode="price_zero",
                decision_at=START,
                volatility=0.001,
                upper_probability=0.1,
                lower_probability=0.2,
                neither_probability=0.7,
                observed=2,
            )
        ],
        bootstrap_replicates=100,
    )
    variance = result["forward_variance"][0]
    assert variance["without_forward_variance"] == 1
    assert variance["overall"]["scored"] == 0


def test_forward_variances_align_with_delayed_publication() -> None:
    """A published bar can lag its end; use the decision-aligned window."""
    base = [_bar(i, 100.0 * exp(0.001 * i)) for i in range(14)]
    rows = tuple(
        MinuteBar(
            b.timestamp,
            b.available_at + timedelta(minutes=1),
            b.open,
            b.high,
            b.low,
            b.close,
        )
        for b in base
    )
    # Index 3 has decision time START+4m; its first holding bar ends +5m.
    variances = _forward_variances(rows, decision_index=3, horizon_minutes=3)
    assert variances is not None
    close_variance, range_variance = variances
    # First future minute contributes an open-to-close move (zero here);
    # subsequent two minutes contribute 0.1% closing returns.
    assert close_variance == pytest.approx((2 / 3) * 0.001**2)
    assert range_variance > 0.0


def test_delayed_publication_produces_forward_variance_diagnostics() -> None:
    bars = tuple(
        MinuteBar(
            b.timestamp,
            b.timestamp + timedelta(minutes=1),
            b.open,
            b.high,
            b.low,
            b.close,
        )
        for b in (_bar(i, 100.0 * exp(0.001 * (i % 2))) for i in range(32))
    )
    forecasts, _ = evaluate_sessions(
        sessions=[Session(DAY, bars)],
        evaluation_days={DAY},
        scenarios=[SCENARIO],
        drift_modes=("price_zero",),
        decision_stride=10,
        lookback_returns=5,
        min_returns=2,
    )
    assert forecasts
    assert all(row.forward_close_variance is not None for row in forecasts)
    diagnostics = analyze_first_passage(forecasts, bootstrap_replicates=100)
    assert diagnostics["forward_variance"][0]["overall"]["scored"] == len(forecasts)

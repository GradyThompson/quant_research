"""Historical first-touch labels, point-in-time volatility, and holdout isolation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from math import exp

import pytest

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import (
    ForecastResult,
    MinuteBar,
    RollingMinuteVolatility,
    Session,
    evaluate_sessions,
    label_first_touch,
    select_receipt_window,
    summarize_forecasts,
)

BASE = datetime(2025, 1, 10, 15, tzinfo=UTC)
SCENARIO = BarrierScenario(0.02, 0.01, 3)


def bar(
    i: int,
    *,
    open: float = 100,
    high: float = 100,
    low: float = 100,
    close: float = 100,
    delay: int = 0,
) -> MinuteBar:
    stamp = BASE + timedelta(minutes=i)
    return MinuteBar(stamp, stamp + timedelta(minutes=delay), open, high, low, close)


def test_upper_first_and_lower_first_and_neither() -> None:
    upper = [bar(i) for i in range(1, 8)]
    upper[2] = bar(3, high=103)
    assert label_first_touch(
        upper, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (0, None)
    lower = [bar(i) for i in range(1, 8)]
    lower[2] = bar(3, low=98)
    assert label_first_touch(
        lower, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (1, None)
    flat = [bar(i) for i in range(1, 8)]
    assert label_first_touch(
        flat, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (2, None)


def test_opening_gap_order_precedes_intrabar_extremes() -> None:
    rows = [bar(i) for i in range(1, 8)]
    rows[2] = bar(3, open=103, high=104, low=98, close=100)
    assert label_first_touch(
        rows, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (0, None)


def test_double_touch_is_ambiguous_without_opening_gap() -> None:
    rows = [bar(i) for i in range(1, 8)]
    rows[2] = bar(3, high=103, low=98)
    assert label_first_touch(
        rows, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (None, "ambiguous_barriers")


def test_missing_minutes_and_partial_horizon_are_excluded() -> None:
    rows = [bar(i) for i in (1, 2, 3, 5, 6, 7)]
    assert label_first_touch(
        rows, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (None, "missing_minutes")
    rows = [bar(i) for i in range(1, 4)]
    assert label_first_touch(
        rows, reference_price=100, decision_at=BASE + timedelta(minutes=2), scenario=SCENARIO
    ) == (None, "incomplete_horizon")


def test_non_aligned_publication_time_is_excluded() -> None:
    rows = [bar(i) for i in range(1, 8)]
    stamp = BASE + timedelta(minutes=2, seconds=15)
    assert label_first_touch(rows, reference_price=100, decision_at=stamp, scenario=SCENARIO) == (
        None,
        "non_aligned_start",
    )


def test_rolling_volatility_uses_only_adjacent_minutes() -> None:
    estimator = RollingMinuteVolatility(lookback_returns=4, min_returns=2)
    for i in range(5):
        close = 100 * exp(0.01 * (i % 2))
        estimator.observe(bar(i + 1, open=close, high=close, low=close, close=close))
    assert estimator.estimate() == pytest.approx(0.01, abs=1e-12)
    # Ignore a large overnight jump and drop the oldest return only on new adjacent data.
    estimator.observe(
        MinuteBar(BASE + timedelta(days=1), BASE + timedelta(days=1), 1000, 1000, 1000, 1000)
    )
    assert estimator.estimate() == pytest.approx(0.01, abs=1e-12)


def test_receipt_window_never_returns_holdout_or_evaluation_session() -> None:
    entries = {
        str(i): {
            "ticker": "AAPL",
            "session": (date(2024, 10, 1) + timedelta(days=i)).isoformat(),
            "security_id": "SEC-1",
            "dataset_id": str(i),
        }
        for i in range(60)
    }
    entries["evaluation"] = {"ticker": "AAPL", "session": "2025-01-13", "security_id": "SEC-1"}
    loaded, days, withheld = select_receipt_window(
        {"schema_version": 1, "entries": entries}, ticker="AAPL", evaluation_start=date(2025, 1, 13)
    )
    assert len(loaded) == 49
    assert len(days) == 10
    assert len(withheld) == 11
    assert all(row["session"] not in withheld for row in loaded)
    assert all(row["session"] != "2025-01-13" for row in loaded)
    assert max(days) < date.fromisoformat(withheld[0])


def test_evaluator_prediction_is_independent_of_future_outcomes() -> None:
    start = datetime(2025, 1, 10, 15, tzinfo=UTC)

    def make_bars(future_multiplier: float) -> tuple[MinuteBar, ...]:
        result = []
        for i in range(20):
            stamp = start + timedelta(minutes=i)
            price = 100 * exp(0.005 * (i % 2))
            if i >= 10:
                price *= future_multiplier
            result.append(MinuteBar(stamp, stamp, price, price, price, price))
        return tuple(result)

    day = date(2025, 1, 10)
    common = dict(
        evaluation_days={day},
        scenarios=[BarrierScenario(0.1, 0.1, 3)],
        decision_stride=5,
        lookback_returns=6,
        min_returns=3,
    )
    original, _ = evaluate_sessions(sessions=[Session(day, make_bars(1.0))], **common)
    changed, _ = evaluate_sessions(sessions=[Session(day, make_bars(1.02))], **common)

    def early(rows: list[ForecastResult]) -> list[ForecastResult]:
        return [r for r in rows if r.decision_at < start + timedelta(minutes=10)]

    assert early(original)
    for a, b in zip(early(original), early(changed), strict=True):
        assert (a.upper_probability, a.lower_probability, a.neither_probability) == (
            b.upper_probability,
            b.lower_probability,
            b.neither_probability,
        )


def test_summary_scores_and_counts_are_correct() -> None:
    day = date(2025, 1, 10)
    bars = tuple(
        bar(
            i,
            close=100 + 0.01 * (i % 2),
            high=100 + 0.01 * (i % 2),
            open=100 + 0.01 * (i % 2),
            low=100 + 0.01 * (i % 2),
        )
        for i in range(1, 25)
    )
    results, excluded = evaluate_sessions(
        sessions=[Session(day, bars)],
        evaluation_days={day},
        scenarios=[BarrierScenario(0.02, 0.01, 3)],
        min_returns=3,
        lookback_returns=6,
        decision_stride=5,
    )
    assert results and excluded
    summaries = summarize_forecasts(results)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["scored"] > 0
        assert summary["log_loss"] >= 0
        assert summary["brier"] >= 0
        assert sum(summary["observed_counts"].values()) == summary["scored"]

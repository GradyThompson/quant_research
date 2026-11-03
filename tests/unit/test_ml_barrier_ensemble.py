"""Test distinct models, cold-start fallback, economy and chronological folds."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from quant_research.model.ml_barrier import FEATURE_NAMES, BarrierRow
from quant_research.model.ml_barrier_ensemble import Predictions, forecast_all, score_predictions


def row(index: int, outcome: int, *, day_index: int | None = None) -> BarrierRow:
    d = date(2024, 1, 1) + timedelta(days=index if day_index is None else day_index)
    morning = datetime(d.year, d.month, d.day, 15, tzinfo=UTC)
    return BarrierRow(
        day=d,
        ticker=f"S{index}",
        decision_at=morning,
        entry_at=morning + timedelta(minutes=1),
        resolved_at=morning + timedelta(hours=1),
        features=tuple(0.01 * (index % 7 + j) for j in range(len(FEATURE_NAMES))),
        baseline=(0.7, 0.05, 0.25),
        outcome=outcome,
        gross_return_bps=(50 if outcome == 0 else (-200 if outcome == 1 else 10)),
        timeout_return_bps=(10 if outcome == 2 else 0),
        entry_price=100,
        ambiguous=False,
    )


def test_ensemble_distinct_components_and_probability_validity() -> None:
    training = [row(i, i % 3) for i in range(120)]
    scored = [row(150 + i, i % 3) for i in range(12)]
    forecasts = forecast_all(training, scored)
    assert set(forecasts.probabilities) == {
        "math",
        "logistic",
        "direct",
        "decomposed",
        "correction",
        "ensemble",
    }
    for p in forecasts.probabilities.values():
        assert p.shape == (12, 3)
        assert np.all((0 <= p) & (p <= 1))
        assert np.allclose(p.sum(axis=1), 1.0)
    assert forecasts.timeout_bps.shape == (12,)


def test_one_class_training_falls_back_to_smoothed_prior() -> None:
    forecasts = forecast_all([row(i, 2) for i in range(30)], [row(40, 0)])
    for p in forecasts.probabilities.values():
        assert np.all(np.isfinite(p))
        assert p.sum() == pytest.approx(1)


def test_select_by_expected_payoff_not_highest_win_probability() -> None:
    candidates = [row(0, 0, day_index=0), row(1, 0, day_index=0)]
    predictions = Predictions(
        probabilities={"test": np.asarray([[0.88, 0.12, 0], [0.82, 0.01, 0.17]])},
        timeout_bps=np.array([0.0, 50.0]),
        empirical_lower_bps=-200.0,
    )
    result = score_predictions(
        rows=candidates, pred=predictions, cost_bps=(4, 12), positions=1, edge_margin_bps=0
    )["test"]["cost_scenarios"]
    assert result["4"]["trades"] == 1
    assert result["4"]["trades_detail"][0]["ticker"] == "S1"
    assert result["12"]["trades_detail"][0]["ticker"] == "S1"
    assert result["12"]["net_bps_per_session"] == pytest.approx(
        result["4"]["net_bps_per_session"] - 8
    )
    assert result["4"]["trades_detail"][0]["commission_bps"] == pytest.approx(0.7)


def test_no_trade_when_expected_net_edge_not_positive() -> None:
    candidate = row(0, 2)
    result = score_predictions(
        rows=[candidate],
        pred=Predictions(
            probabilities={"math": np.asarray([[0.8, 0.2, 0.0]])},
            timeout_bps=np.zeros(1),
            empirical_lower_bps=-200,
        ),
        cost_bps=(4,),
        positions=3,
        edge_margin_bps=2,
    )["math"]["cost_scenarios"]["4"]
    assert result["trades"] == 0
    assert result["net_bps_per_session"] == 0

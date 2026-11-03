"""Temporal, missing-data and paired proper-score invariants for M3 research."""

from datetime import UTC, datetime, timedelta
from math import exp, sin

import pytest

from quant_research.model.conditional_direction import ConditionalDirectionModel
from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.model.directional_research import (
    DirectionalResearchRow,
    build_directional_rows,
    score_directional_models,
)
from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import ForecastResult, MinuteBar, Session

START = datetime(2025, 1, 1, 15, tzinfo=UTC)


def session(start: datetime, *, factor: bool) -> Session:
    bars = []
    previous = 100.0
    for i in range(180):
        # Factor and asset vary nontrivially; all prices use last close.
        change = 0.00015 * sin(i * 0.39) + (0 if factor else 0.0001 * sin(i * 0.9))
        close = previous * exp(change)
        stamp = start + timedelta(minutes=i + 1)
        bars.append(
            MinuteBar(stamp, stamp, previous, max(previous, close), min(previous, close), close)
        )
        previous = close
    return Session(start.date(), tuple(bars))


def forecast(sess: Session, decision_index: int = 130) -> ForecastResult:
    bar = sess.bars[decision_index]
    return ForecastResult(
        day=sess.day,
        scenario=BarrierScenario(0.002, 0.003, 5),
        drift_mode="price_zero",
        decision_at=bar.available_at,
        volatility=0.001,
        upper_probability=0.2,
        lower_probability=0.3,
        neither_probability=0.5,
        observed=1,
        volatility_method="seasonal_close",
        decision_bar_index=decision_index,
    )


def test_feature_uses_published_past_and_future_prices_do_not_affect_it():
    asset = session(START, factor=False)
    factor = session(START, factor=True)
    f = forecast(asset)
    rows = build_directional_rows(
        asset_sessions=[asset],
        factor_sessions=[factor],
        forecasts=[f],
        beta_lookback=60,
        min_beta_observations=30,
    )
    assert len(rows) == 1
    assert rows[0].label_available_at == asset.bars[135].available_at
    distorted = Session(
        asset.day,
        asset.bars[:136]
        + tuple(
            MinuteBar(
                bar.timestamp, bar.available_at, bar.open, bar.high * 2, bar.low, bar.close * 2
            )
            for bar in asset.bars[136:]
        ),
    )
    altered = build_directional_rows(
        asset_sessions=[distorted],
        factor_sessions=[factor],
        forecasts=[f],
        beta_lookback=60,
        min_beta_observations=30,
    )
    assert altered[0].feature == pytest.approx(rows[0].feature)


def test_stale_factor_not_forward_filled():
    asset = session(START, factor=False)
    factor = session(START, factor=True)
    result = build_directional_rows(
        asset_sessions=[asset],
        factor_sessions=[Session(factor.day, factor.bars[:125])],
        forecasts=[forecast(asset)],
        beta_lookback=60,
        min_beta_observations=30,
    )
    assert result == ()


def test_three_class_scoring_and_timeout_is_unchanged():
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    base = ThreeWayForecast(0.2, 0.3, 0.5)
    intercept = ConditionalDirectionModel(0, 0, 0, 1, cutoff, 30, "zero", False)
    signal = ConditionalDirectionModel(0, 0, 0, 1, cutoff, 30, "zero", True)
    rows = [
        DirectionalResearchRow(
            day=START.date() + timedelta(days=i),
            decision_at=START + timedelta(days=i),
            label_available_at=START + timedelta(days=i, minutes=5),
            feature=float(i),
            baseline=base,
            observed=i % 3,
        )
        for i in range(3)
    ]
    scores = score_directional_models(
        rows, intercept_only=intercept, with_signal=signal, bootstrap_replicates=50
    )
    assert scores["scored"] == 3
    assert scores["first_touches"] == 2
    assert scores["models"]["base"]["brier"] == pytest.approx(scores["models"]["signal"]["brier"])
    assert scores["base_minus_signal_brier"] == pytest.approx(0)
    assert scores["intercept_minus_signal_brier"] == pytest.approx(0)
    assert scores["intercept_minus_signal_95pct_session_bootstrap"] == pytest.approx([0, 0])


def test_timeout_is_not_a_direction_training_label():
    row = DirectionalResearchRow(
        START.date(),
        START,
        START + timedelta(minutes=5),
        1,
        ThreeWayForecast(0.25, 0.25, 0.5),
        2,
    )
    assert row.training_touch() is None

"""Integration and point-in-time checks for the first-passage model adapter."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import exp

import pyarrow as pa
import pytest

from quant_research.core import SecurityId
from quant_research.model.barrier import BarrierModelInput
from quant_research.model.barrier_runner import run_barrier_model
from quant_research.model.first_passage_model import BarrierScenario, FirstPassageModel
from quant_research.model.volatility import estimate_minute_volatility


def bars(*, count: int = 75, gap_at: int | None = None) -> pa.Table:
    start = datetime(2025, 1, 10, 14, 31, tzinfo=UTC)
    times = [
        start + timedelta(minutes=i + int(gap_at is not None and i >= gap_at)) for i in range(count)
    ]
    # Exactly +/- 1% log returns so RMS volatility is known.
    closes = [100 * exp(0.01 * (i % 2)) for i in range(count)]
    return pa.table({"timestamp": times, "available_at": times, "close": closes})


def input_data(table: pa.Table) -> BarrierModelInput:
    return BarrierModelInput(
        security_id=SecurityId("SEC-000001"),
        decision_at=datetime(2025, 1, 10, 21, tzinfo=UTC),
        minute_bars=table,
        upper_barrier_pct=0.02,
        lower_barrier_pct=0.01,
        max_holding_minutes=60,
    )


def test_volatility_is_rms_of_contiguous_minute_log_returns() -> None:
    data = input_data(bars())
    sigma = estimate_minute_volatility(data.minute_bars, decision_at=data.decision_at)
    assert sigma == pytest.approx(0.01, abs=1e-14)


def test_minute_gap_is_excluded_from_volatility() -> None:
    data = input_data(bars(gap_at=35))
    sigma = estimate_minute_volatility(
        data.minute_bars, decision_at=data.decision_at, min_returns=73
    )
    assert sigma == pytest.approx(0.01, abs=1e-14)
    with pytest.raises(ValueError, match="Insufficient"):
        estimate_minute_volatility(data.minute_bars, decision_at=data.decision_at, min_returns=74)


def test_runner_accepts_first_passage_forecast() -> None:
    data = input_data(bars())
    prediction = run_barrier_model(FirstPassageModel(), data)
    assert prediction.generated_at == data.decision_at
    assert prediction.security_id == data.security_id
    assert prediction.confidence is None
    assert sum(
        (
            prediction.probability_upper_first,
            prediction.probability_lower_first,
            prediction.probability_neither,
        )
    ) == pytest.approx(1.0)


def test_grid_matches_individual_forecasts() -> None:
    data = input_data(bars())
    model = FirstPassageModel()
    configs = [
        BarrierScenario(0.005, 0.005, 15),
        BarrierScenario(0.02, 0.01, 60),
        BarrierScenario(0.02, 0.01, 120),
    ]
    grid = model.predict_many(data, configs)
    individually = [
        model.predict(
            replace(
                data,
                upper_barrier_pct=config.upper_barrier_pct,
                lower_barrier_pct=config.lower_barrier_pct,
                max_holding_minutes=config.horizon_minutes,
            )
        )
        for config in configs
    ]
    assert grid == individually
    assert model.predict_many(data, []) == []


def test_rejects_future_or_unavailable_bar() -> None:
    data = input_data(bars())
    for column in ("available_at", "timestamp"):
        values = data.minute_bars.column(column).to_pylist()
        values[-1] = data.decision_at + timedelta(minutes=1)
        bad = data.minute_bars.set_column(
            data.minute_bars.column_names.index(column), column, pa.array(values)
        )
        with pytest.raises(ValueError, match="available by decision_at"):
            run_barrier_model(FirstPassageModel(), replace(data, minute_bars=bad))


def test_insufficient_return_history_rejected() -> None:
    data = input_data(bars(count=30))
    with pytest.raises(ValueError, match="Insufficient"):
        run_barrier_model(FirstPassageModel(), data)


def test_volatility_lookback_not_recent_data_length() -> None:
    table = bars(count=75)
    data = input_data(table)
    sigma = estimate_minute_volatility(
        data.minute_bars, decision_at=data.decision_at, lookback_returns=10, min_returns=10
    )
    assert sigma == pytest.approx(0.01, abs=1e-14)

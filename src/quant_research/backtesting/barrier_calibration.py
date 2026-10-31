"""Evaluate barrier forecasts using explicit minute-reference conventions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from math import isfinite, log
from statistics import mean

import pyarrow as pa

from quant_research.backtesting.calibration_benchmarks import (
    BenchmarkScores,
    score_retrospective_frequency_baseline,
    score_uniform_baseline,
)
from quant_research.labels.barrier import BarrierOutcome, label_barrier_event
from quant_research.model.barrier import BarrierForecast


class ReferenceConvention(StrEnum):
    EXECUTABLE = "executable"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    total_forecasts: int
    scored: int
    ambiguous: int
    incomplete: int
    brier_score: float | None
    log_loss: float | None
    observed: dict[str, int]
    confidence_buckets: dict[str, tuple[int, float]]
    upper_probability_buckets: dict[str, tuple[int, float, float]]
    uniform_baseline: BenchmarkScores
    retrospective_frequency_baseline: BenchmarkScores


def _bucket(value: float) -> str:
    index = min(int(value * 5), 4)
    return f"{index * 20:02d}-{(index + 1) * 20:02d}%"


def evaluate_barrier_calibration(
    *,
    forecasts: list[BarrierForecast],
    bars: pa.Table,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    max_holding_minutes: int,
    reference_convention: ReferenceConvention = ReferenceConvention.EXECUTABLE,
) -> CalibrationResult:
    """Evaluate outcome frequencies without changing the trading engine.

    Minute bar timestamps mark bar *ends*. The first reference bar is the
    first minute open strictly later than forecast time for EXECUTABLE, or
    greater than/equal to forecast time for LEGACY. The assumed entry open
    does not include slippage or fees; only label scoring uses it.
    """
    if max_holding_minutes < 1:
        raise ValueError("max_holding_minutes must be positive")
    required = {"timestamp", "available_at", "open", "high", "low"}
    if not required.issubset(bars.column_names):
        raise ValueError("Missing minute bars columns")
    rows = bars.to_pylist()
    stamps = [row["timestamp"] for row in rows]
    if stamps != sorted(set(stamps)):
        raise ValueError("Bars must be sorted with unique timestamps")

    outcomes = (BarrierOutcome.UPPER_FIRST, BarrierOutcome.LOWER_FIRST, BarrierOutcome.TIMEOUT)
    observed: Counter[str] = Counter()
    confidence: dict[str, list[int]] = {}
    upper_groups: dict[str, list[tuple[float, int]]] = {}
    briers: list[float] = []
    losses: list[float] = []
    actual_classes: list[int] = []
    ambiguous = incomplete = 0

    for forecast in forecasts:
        index = next(
            (
                i
                for i, timestamp in enumerate(stamps)
                if (
                    timestamp - timedelta(minutes=1) > forecast.generated_at
                    if reference_convention is ReferenceConvention.EXECUTABLE
                    else timestamp - timedelta(minutes=1) >= forecast.generated_at
                )
            ),
            None,
        )
        if index is None:
            incomplete += 1
            continue
        first = rows[index]
        price = first["open"]
        if not isinstance(price, (int, float)) or not isfinite(price) or price <= 0:
            raise ValueError("Invalid reference open")
        # An incomplete horizon can still be scored when a barrier is reached
        # unambiguously before the data ends.
        entry_at = stamps[index] - timedelta(minutes=1)
        window = bars.slice(index, max_holding_minutes)
        try:
            label = label_barrier_event(
                future_bars=window,
                entry_at=entry_at,
                entry_price=float(price),
                upper_barrier_pct=upper_barrier_pct,
                lower_barrier_pct=lower_barrier_pct,
                max_holding_minutes=max_holding_minutes,
            )
        except ValueError as exc:
            if "Insufficient future minute bars" not in str(exc):
                raise
            incomplete += 1
            continue
        if label.outcome is BarrierOutcome.AMBIGUOUS:
            ambiguous += 1
            continue
        probabilities = (
            forecast.probability_upper_first,
            forecast.probability_lower_first,
            forecast.probability_neither,
        )
        true_index = outcomes.index(label.outcome)
        observed[label.outcome.value] += 1
        actual_classes.append(true_index)
        briers.append(
            sum((p - float(i == true_index)) ** 2 for i, p in enumerate(probabilities)) / 3
        )
        losses.append(-log(max(probabilities[true_index], 1e-15)))
        if forecast.confidence is not None:
            key = _bucket(forecast.confidence)
            confidence.setdefault(key, []).append(
                int(probabilities.index(max(probabilities)) == true_index)
            )
        upper_key = _bucket(probabilities[0])
        upper_groups.setdefault(upper_key, []).append((probabilities[0], int(true_index == 0)))

    return CalibrationResult(
        total_forecasts=len(forecasts),
        uniform_baseline=score_uniform_baseline(actual_classes),
        retrospective_frequency_baseline=score_retrospective_frequency_baseline(actual_classes),
        scored=len(briers),
        ambiguous=ambiguous,
        incomplete=incomplete,
        brier_score=mean(briers) if briers else None,
        log_loss=mean(losses) if losses else None,
        observed=dict(observed),
        confidence_buckets={k: (len(v), mean(v)) for k, v in sorted(confidence.items())},
        upper_probability_buckets={
            k: (len(v), mean(p for p, _ in v), mean(y for _, y in v))
            for k, v in sorted(upper_groups.items())
        },
    )

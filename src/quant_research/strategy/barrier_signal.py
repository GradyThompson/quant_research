"""Run a private barrier model and apply the public entry policy."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research.model.barrier import BarrierForecast, BarrierModel, BarrierModelInput
from quant_research.model.barrier_runner import run_barrier_model
from quant_research.strategy.barrier_decision import (
    BarrierDecisionConfig,
    BarrierEntryDecision,
    decide_barrier_entry,
)


@dataclass(frozen=True, slots=True)
class BarrierSignalResult:
    """Validated forecast paired with a research-only entry decision."""

    forecast: BarrierForecast
    decision: BarrierEntryDecision


def evaluate_barrier_signal(
    model: BarrierModel,
    data: BarrierModelInput,
    config: BarrierDecisionConfig,
    *,
    has_open_position: bool = False,
    has_pending_entry: bool = False,
) -> BarrierSignalResult:
    """Evaluate one point-in-time window without placing any orders."""
    forecast = run_barrier_model(model, data)
    decision = decide_barrier_entry(
        forecast,
        config,
        has_open_position=has_open_position,
        has_pending_entry=has_pending_entry,
    )
    return BarrierSignalResult(forecast=forecast, decision=decision)

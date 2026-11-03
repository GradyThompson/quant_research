"""Reusable, proper three-class forecast scoring on paired chronological decisions.

No fitting, I/O or model assumptions. Session-block intervals preserve within-day
correlation and never resample individual one-minute observations independently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import log
from typing import Sequence

from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.model.first_passage_diagnostics import session_block_interval


@dataclass(frozen=True, slots=True)
class PairedProbabilityObservation:
    day: date
    decision_at: datetime
    observed: int  # upper=0, lower=1, neither=2
    base: ThreeWayForecast
    intercept: ThreeWayForecast
    signal: ThreeWayForecast

    def __post_init__(self) -> None:
        if self.observed not in (0, 1, 2):
            raise ValueError("outcome must be upper, lower, or neither")
        if self.decision_at.tzinfo is None or self.decision_at.utcoffset() != timedelta(0):
            raise ValueError("decision timestamp must be UTC-aware")


def _brier(p: ThreeWayForecast, outcome: int) -> float:
    return (
        sum(
            (value - float(i == outcome)) ** 2
            for i, value in enumerate((p.upper_first, p.lower_first, p.neither))
        )
        / 3.0
    )


def _log_loss(p: ThreeWayForecast, outcome: int) -> float:
    return -log(max(1e-15, (p.upper_first, p.lower_first, p.neither)[outcome]))


def _conditional_log_loss(p: ThreeWayForecast, outcome: int) -> float:
    conditional_up = p.upper_first / p.touch_probability
    return -log(max(1e-15, conditional_up if outcome == 0 else 1.0 - conditional_up))


def score_paired_probabilities(
    observations: Sequence[PairedProbabilityObservation],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20250113,
) -> dict[str, object]:
    """Score identical decisions for all alternatives, with session-block CIs.

    Deltas are baseline minus candidate, so positive means candidate improves.
    The conditional loss is evaluated only for first-touch (upper/lower) labels.
    """
    if not observations:
        raise ValueError("no paired validation observations")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    identifiers = [(obs.day, obs.decision_at) for obs in observations]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate scored decisions")
    by_day: dict[str, dict[date, list[float]]] = {
        "base_minus_signal": defaultdict(list),
        "intercept_minus_signal": defaultdict(list),
    }
    loss: dict[str, list[float]] = {key: [] for key in ("base", "intercept", "signal")}
    log_losses: dict[str, list[float]] = {key: [] for key in loss}
    conditional_losses: dict[str, list[float]] = {key: [] for key in loss}
    for obs in observations:
        alternatives = {"base": obs.base, "intercept": obs.intercept, "signal": obs.signal}
        for key, probabilities in alternatives.items():
            loss[key].append(_brier(probabilities, obs.observed))
            log_losses[key].append(_log_loss(probabilities, obs.observed))
            if obs.observed != 2:
                conditional_losses[key].append(_conditional_log_loss(probabilities, obs.observed))
        by_day["base_minus_signal"][obs.day].append(loss["base"][-1] - loss["signal"][-1])
        by_day["intercept_minus_signal"][obs.day].append(loss["intercept"][-1] - loss["signal"][-1])

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "scored": len(observations),
        "scored_sessions": len({obs.day for obs in observations}),
        "first_touches": len(conditional_losses["base"]),
        "models": {
            key: {
                "brier": mean(loss[key]),
                "log_loss": mean(log_losses[key]),
                "conditional_touch_log_loss": mean(conditional_losses[key]),
            }
            for key in loss
        },
        "base_minus_signal_brier": mean(
            [value for values in by_day["base_minus_signal"].values() for value in values]
        ),
        "intercept_minus_signal_brier": mean(
            [value for values in by_day["intercept_minus_signal"].values() for value in values]
        ),
        "base_minus_signal_95pct_session_bootstrap": session_block_interval(
            by_day["base_minus_signal"], replications=bootstrap_replicates, seed=bootstrap_seed
        ),
        "intercept_minus_signal_95pct_session_bootstrap": session_block_interval(
            by_day["intercept_minus_signal"], replications=bootstrap_replicates, seed=bootstrap_seed
        ),
    }

"""Training-only categorical frequency benchmark for barrier first touches.

A separate unconditional distribution is fitted for each barrier configuration.
The three regular-session periods are shrunk toward their scenario's pooled
training frequencies. No validation (or embargo) observations are fitted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import log
from typing import Sequence, cast

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_diagnostics import session_block_interval
from quant_research.model.first_passage_validation import (
    ForecastResult,
    Session,
    label_first_touch,
)


@dataclass(frozen=True, slots=True)
class HistoricalFrequencyBenchmark:
    """Fixed training-period distributions for upper/lower/neither outcomes."""

    training_days: frozenset[date]
    pooled_counts: dict[BarrierScenario, tuple[int, int, int]]
    period_counts: dict[tuple[BarrierScenario, int], tuple[int, int, int]]
    period_prior_strength: float

    def probabilities(
        self, scenario: BarrierScenario, bar_index: int
    ) -> tuple[float, float, float]:
        """Predict using training-only counts and a pooled Dirichlet prior."""
        pooled = self.pooled_counts[scenario]
        pooled_n = sum(pooled)
        if pooled_n == 0:
            raise ValueError("No training outcomes for scenario")
        # Add one observation of each class to prevent exact-zero probabilities.
        prior = tuple((value + 1.0) / (pooled_n + 3.0) for value in pooled)
        period = _period(bar_index)
        counts = self.period_counts.get((scenario, period), (0, 0, 0))
        denominator = sum(counts) + self.period_prior_strength
        return (
            (counts[0] + self.period_prior_strength * prior[0]) / denominator,
            (counts[1] + self.period_prior_strength * prior[1]) / denominator,
            (counts[2] + self.period_prior_strength * prior[2]) / denominator,
        )


def _period(bar_index: int) -> int:
    if bar_index < 0 or bar_index >= 390:
        raise ValueError("bar_index must be in [0, 390)")
    return 0 if bar_index < 60 else 1 if bar_index < 300 else 2


def fit_historical_frequency_benchmark(
    *,
    sessions: Sequence[Session],
    scenarios: Sequence[BarrierScenario],
    decision_stride: int,
    per_scenario_stride: bool,
    period_prior_strength: float = 30.0,
) -> HistoricalFrequencyBenchmark:
    """Fit from designated training sessions, never an evaluation split.

    Only first touches with unambiguous, fully aligned holding windows are
    counted. This is a benchmark, NOT an input to Brownian probabilities.
    """
    if not sessions:
        raise ValueError("Training sessions are required")
    if not scenarios:
        raise ValueError("Scenarios are required")
    if type(decision_stride) is not int or decision_stride < 1:
        raise ValueError("decision_stride must be positive")
    if type(per_scenario_stride) is not bool:
        raise ValueError("per_scenario_stride must be boolean")
    if not 0 < period_prior_strength < float("inf"):
        raise ValueError("period_prior_strength must be finite and positive")
    days = [session.day for session in sessions]
    if days != sorted(set(days)):
        raise ValueError("Training sessions must have increasing unique days")
    totals: dict[BarrierScenario, list[int]] = {scenario: [0, 0, 0] for scenario in scenarios}
    periods: dict[tuple[BarrierScenario, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for session in sessions:
        for i, bar in enumerate(session.bars):
            for scenario in scenarios:
                stride = scenario.horizon_minutes + 1 if per_scenario_stride else decision_stride
                if i % stride != 0:
                    continue
                outcome, _ = label_first_touch(
                    session.bars,
                    reference_price=bar.close,
                    decision_at=bar.available_at,
                    scenario=scenario,
                )
                if outcome is None:
                    continue
                totals[scenario][outcome] += 1
                periods[(scenario, _period(i))][outcome] += 1
    if any(sum(counts) == 0 for counts in totals.values()):
        raise ValueError("No unambiguous training outcomes for a scenario")
    return HistoricalFrequencyBenchmark(
        training_days=frozenset(days),
        pooled_counts={
            scenario: cast(tuple[int, int, int], tuple(counts))
            for scenario, counts in totals.items()
        },
        period_counts={
            key: cast(tuple[int, int, int], tuple(counts)) for key, counts in periods.items()
        },
        period_prior_strength=period_prior_strength,
    )


def benchmark_scores(
    rows: Sequence[ForecastResult],
    benchmark: HistoricalFrequencyBenchmark,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20250113,
) -> list[dict[str, object]]:
    """Paired multiclass scores for held-out DEVELOPMENT validation forecasts.

    Higher ``brier_skill_vs_frequency`` is better. Observations are never
    used to fit the stored historical-frequency probabilities.
    """
    groups: dict[tuple[BarrierScenario, str, str], list[ForecastResult]] = defaultdict(list)
    for row in rows:
        if row.day in benchmark.training_days:
            raise ValueError("Training decisions cannot be included in benchmark scoring")
        groups[(row.scenario, row.drift_mode, row.volatility_method)].append(row)
    result: list[dict[str, object]] = []
    for (scenario, drift, method), group in groups.items():
        baseline_loss = 0.0
        model_loss = 0.0
        baseline_log_loss = 0.0
        model_log_loss = 0.0
        observed = [0, 0, 0]
        predicted = [0.0, 0.0, 0.0]
        paired_by_day: dict[date, list[float]] = defaultdict(list)
        for row in group:
            if row.decision_bar_index is None:
                raise ValueError("Benchmark scoring requires decision_bar_index")
            baseline = benchmark.probabilities(scenario, row.decision_bar_index)
            model = (row.upper_probability, row.lower_probability, row.neither_probability)
            observed[row.observed] += 1
            for k in range(3):
                predicted[k] += baseline[k]
            baseline_row_brier = (
                sum((p - float(row.observed == k)) ** 2 for k, p in enumerate(baseline)) / 3
            )
            model_row_brier = (
                sum((p - float(row.observed == k)) ** 2 for k, p in enumerate(model)) / 3
            )
            baseline_loss += baseline_row_brier
            model_loss += model_row_brier
            paired_by_day[row.day].append(baseline_row_brier - model_row_brier)
            baseline_log_loss -= log(max(1e-15, baseline[row.observed]))
            model_log_loss -= log(max(1e-15, model[row.observed]))
        n = len(group)
        baseline_brier = baseline_loss / n
        model_brier = model_loss / n
        result.append(
            {
                "upper_barrier_pct": scenario.upper_barrier_pct,
                "lower_barrier_pct": scenario.lower_barrier_pct,
                "horizon_minutes": scenario.horizon_minutes,
                "drift_mode": drift,
                "volatility_method": method,
                "scored": n,
                "training_observations": sum(benchmark.pooled_counts[scenario]),
                "training_sessions": len(benchmark.training_days),
                "model_brier": model_brier,
                "frequency_brier": baseline_brier,
                "scored_sessions": len(paired_by_day),
                "frequency_minus_model_brier": baseline_brier - model_brier,
                "frequency_minus_model_brier_95pct_session_bootstrap": session_block_interval(
                    paired_by_day,
                    replications=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
                "model_log_loss": model_log_loss / n,
                "frequency_log_loss": baseline_log_loss / n,
                "brier_skill_vs_frequency": (
                    1.0 - model_brier / baseline_brier if baseline_brier > 0 else None
                ),
                "observed_rates": {
                    "upper": observed[0] / n,
                    "lower": observed[1] / n,
                    "neither": observed[2] / n,
                },
                "frequency_predicted_rates": {
                    "upper": predicted[0] / n,
                    "lower": predicted[1] / n,
                    "neither": predicted[2] / n,
                },
            }
        )
    return sorted(
        result,
        key=lambda item: (
            cast(float, item["upper_barrier_pct"]),
            cast(float, item["lower_barrier_pct"]),
            cast(int, item["horizon_minutes"]),
            str(item["drift_mode"]),
            str(item["volatility_method"]),
        ),
    )

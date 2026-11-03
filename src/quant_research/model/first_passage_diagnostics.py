"""Research-only first-passage calibration and paired volatility comparisons.

Uncertainty resamples *sessions*, not individual intraday forecasts. It is
exploratory with few sessions, and neither corrects model misspecification nor
substitutes for untouched chronological testing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from math import sqrt
from random import Random
from typing import Sequence, cast

from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import ForecastResult

GroupKey = tuple[BarrierScenario, str, str]
PairKey = tuple[date, datetime]


def _brier(row: ForecastResult) -> float:
    probabilities = (row.upper_probability, row.lower_probability, row.neither_probability)
    return sum((p - float(row.observed == k)) ** 2 for k, p in enumerate(probabilities)) / 3.0


def _touch_probability(row: ForecastResult) -> float:
    return row.upper_probability + row.lower_probability


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def session_block_interval(
    day_values: dict[date, list[float]],
    *,
    replications: int,
    seed: int,
) -> list[float] | None:
    """Equal-weight session resampling, with forecast-weighted point estimates.

    Returns no interval for fewer than two sessions. Each selected session
    contributes all its forecast values, preserving within-session clustering.
    """
    days = sorted(day_values)
    if len(days) < 2:
        return None
    rng = Random(seed)
    summaries = [(sum(day_values[day]), len(day_values[day])) for day in days]
    samples: list[float] = []
    for _ in range(replications):
        total = 0.0
        count = 0
        for _ in days:
            day_sum, day_count = summaries[rng.randrange(len(days))]
            total += day_sum
            count += day_count
        samples.append(total / count)
    return [_percentile(samples, 0.025), _percentile(samples, 0.975)]


def _group_description(scenario: BarrierScenario, drift: str) -> dict[str, str | float | int]:
    return {
        "upper_barrier_pct": scenario.upper_barrier_pct,
        "lower_barrier_pct": scenario.lower_barrier_pct,
        "horizon_minutes": scenario.horizon_minutes,
        "drift_mode": drift,
    }


def _variance_row_summary(rows: Sequence[ForecastResult]) -> dict[str, float | int | None]:
    """Compare forecast one-minute variance with realized forward variance.

    The comparison is descriptive and uses future data *only* for diagnostics.
    """
    selected = [
        row
        for row in rows
        if row.forward_close_variance is not None and row.forward_range_variance is not None
    ]
    if not selected:
        return {
            "scored": 0,
            "predicted_sigma": None,
            "realized_sigma": None,
            "predicted_to_realized_variance": None,
            "observed_touch_rate": None,
            "predicted_touch_rate": None,
        }

    n = len(selected)
    predicted_variance = sum(row.volatility**2 for row in selected) / n
    realized_variance = (
        sum(
            (
                row.forward_close_variance
                if row.volatility_method in ("close", "seasonal_close")
                else row.forward_range_variance
            )
            or 0.0
            for row in selected
        )
        / n
    )
    return {
        "scored": n,
        "predicted_sigma": sqrt(predicted_variance),
        "realized_sigma": sqrt(realized_variance),
        "predicted_to_realized_variance": (
            predicted_variance / realized_variance if realized_variance > 0.0 else None
        ),
        "observed_touch_rate": sum(row.observed != 2 for row in selected) / n,
        "predicted_touch_rate": sum(_touch_probability(row) for row in selected) / n,
    }


def _session_period(index: int) -> str:
    """Approximate RTH segment using position in the pinned session."""
    if index < 60:
        return "first_60_bars"
    if index < 300:
        return "middle_240_bars"
    return "last_90_bars"


def _variance_diagnostics(
    groups: dict[GroupKey, list[ForecastResult]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for (scenario, drift, method), group in groups.items():
        # Drift assumptions have identical variance inputs; don't double report.
        if drift != "price_zero":
            continue
        available = [row for row in group if row.decision_bar_index is not None]
        periods: list[dict[str, object]] = []
        for period in ("first_60_bars", "middle_240_bars", "last_90_bars"):
            subset = [
                row
                for row in available
                if row.decision_bar_index is not None
                and _session_period(row.decision_bar_index) == period
            ]
            if subset:
                periods.append({"period": period, **_variance_row_summary(subset)})
        results.append(
            {
                **_group_description(scenario, drift),
                "volatility_method": method,
                "without_forward_variance": len(group)
                - sum(
                    row.forward_close_variance is not None
                    and row.forward_range_variance is not None
                    for row in group
                ),
                "overall": _variance_row_summary(group),
                "by_session_period": periods,
            }
        )
    return results


def analyze_first_passage(
    rows: Sequence[ForecastResult],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20250113,
) -> dict[str, object]:
    """Summarize calibration uncertainty and paired close/Parkinson scores.

    No future outcomes enter the mathematical forecasts. The historical
    observations here are used only for development diagnostics.
    """
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be an integer >= 100")
    if type(bootstrap_seed) is not int:
        raise ValueError("bootstrap_seed must be an integer")

    groups: dict[GroupKey, list[ForecastResult]] = defaultdict(list)
    for row in rows:
        groups[(row.scenario, row.drift_mode, row.volatility_method)].append(row)

    calibration: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    class_calibration: list[dict[str, object]] = []

    for (scenario, drift, method), group in groups.items():
        n = len(group)
        touch_count = sum(row.observed != 2 for row in group)
        predicted_touches = sum(_touch_probability(row) for row in group)
        model_brier = sum(_brier(row) for row in group) / n
        # Always-neither probabilities: (0,0,1). Its Brier is 0 for a
        # timeout and 2/3 for a touch, using the project's /3 convention.
        neither_brier = (2.0 / 3.0) * touch_count / n
        touch_difference_by_day: dict[date, list[float]] = defaultdict(list)
        session_counts: dict[date, list[float]] = defaultdict(list)
        for row in group:
            touch_difference_by_day[row.day].append(
                float(row.observed != 2) - _touch_probability(row)
            )
            session_counts[row.day].append(float(row.observed != 2))
        for k, outcome_name in enumerate(("upper", "lower", "neither")):
            by_day: dict[date, list[float]] = defaultdict(list)
            total_prediction = 0.0
            count_observed = 0
            for row in group:
                probability = (
                    row.upper_probability,
                    row.lower_probability,
                    row.neither_probability,
                )[k]
                total_prediction += probability
                count_observed += row.observed == k
                by_day[row.day].append(float(row.observed == k) - probability)
            class_calibration.append(
                {
                    **_group_description(scenario, drift),
                    "volatility_method": method,
                    "outcome": outcome_name,
                    "scored": n,
                    "observed_fraction": count_observed / n,
                    "predicted_fraction": total_prediction / n,
                    "observed_minus_expected_rate": (count_observed - total_prediction) / n,
                    "gap_95pct_session_bootstrap": session_block_interval(
                        by_day,
                        replications=bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                }
            )
        calibration.append(
            {
                **_group_description(scenario, drift),
                "volatility_method": method,
                "scored": n,
                "scored_sessions": len(touch_difference_by_day),
                "observed_touches": touch_count,
                "expected_touches": predicted_touches,
                "observed_minus_expected_touch_rate": (touch_count - predicted_touches) / n,
                "touch_rate_gap_95pct_session_bootstrap": session_block_interval(
                    touch_difference_by_day,
                    replications=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
                "model_brier": model_brier,
                "always_neither_brier": neither_brier,
                "brier_skill_vs_neither": (
                    1.0 - model_brier / neither_brier if neither_brier > 0.0 else None
                ),
                "touches_by_session": [
                    {"day": day.isoformat(), "scored": len(values), "touches": int(sum(values))}
                    for day, values in sorted(session_counts.items())
                ],
            }
        )

    scenario_drifts = {(scenario, drift) for scenario, drift, _ in groups}
    for scenario, drift in scenario_drifts:
        left_rows = groups.get((scenario, drift, "close"))
        right_rows = groups.get((scenario, drift, "parkinson"))
        if not left_rows or not right_rows:
            continue

        def index_by_decision(observations: list[ForecastResult]) -> dict[PairKey, ForecastResult]:
            result: dict[PairKey, ForecastResult] = {}
            for row in observations:
                key = (row.day, row.decision_at)
                if key in result:
                    raise ValueError("Duplicate decision in volatility comparison")
                result[key] = row
            return result

        left = index_by_decision(left_rows)
        right = index_by_decision(right_rows)
        if left.keys() != right.keys():
            raise ValueError("Volatility methods must score identical decisions")
        differences: dict[date, list[float]] = defaultdict(list)
        for key, a in left.items():
            b = right[key]
            if a.observed != b.observed:
                raise ValueError("Different observed labels for the same decision")
            differences[a.day].append(_brier(a) - _brier(b))
        delta = sum(sum(values) for values in differences.values()) / len(left)
        paired.append(
            {
                **_group_description(scenario, drift),
                "scored_pairs": len(left),
                "scored_sessions": len(differences),
                "close_minus_parkinson_brier": delta,
                "close_minus_parkinson_brier_95pct_session_bootstrap": session_block_interval(
                    differences,
                    replications=bootstrap_replicates,
                    seed=bootstrap_seed,
                ),
                "positive_means_parkinson_better": True,
            }
        )

    seasonal_pairs: list[dict[str, object]] = []
    for (scenario, drift, method), group in groups.items():
        if method not in ("close", "parkinson"):
            continue
        seasonal_group = groups.get((scenario, drift, f"seasonal_{method}"))
        if not seasonal_group:
            continue
        baseline = {(row.day, row.decision_at): row for row in group}
        seasonal = {(row.day, row.decision_at): row for row in seasonal_group}
        if (
            baseline.keys() != seasonal.keys()
            or len(baseline) != len(group)
            or len(seasonal) != len(seasonal_group)
        ):
            raise ValueError("Seasonal and nonseasonal decisions must match exactly")
        seasonal_differences: dict[date, list[float]] = defaultdict(list)
        for key, normal in baseline.items():
            adjusted = seasonal[key]
            if normal.observed != adjusted.observed:
                raise ValueError("Seasonal and baseline labels disagree")
            seasonal_differences[normal.day].append(_brier(normal) - _brier(adjusted))
        seasonal_pairs.append(
            {
                **_group_description(scenario, drift),
                "volatility_method": method,
                "scored_pairs": len(baseline),
                "baseline_minus_seasonal_brier": sum(map(sum, seasonal_differences.values()))
                / len(baseline),
                "baseline_minus_seasonal_brier_95pct_session_bootstrap": session_block_interval(
                    seasonal_differences, replications=bootstrap_replicates, seed=bootstrap_seed
                ),
            }
        )

    def order(value: dict[str, object]) -> tuple[float, float, int, str, str]:
        return (
            cast(float, value["upper_barrier_pct"]),
            cast(float, value["lower_barrier_pct"]),
            cast(int, value["horizon_minutes"]),
            str(value["drift_mode"]),
            str(value.get("volatility_method", "")),
        )

    return {
        "calibration": sorted(calibration, key=order),
        "class_calibration": sorted(
            class_calibration, key=lambda row: (*order(row), str(row["outcome"]))
        ),
        "paired_volatility": sorted(paired, key=order),
        "paired_seasonality": sorted(seasonal_pairs, key=order),
        "forward_variance": sorted(_variance_diagnostics(groups), key=order),
        "uncertainty": {
            "method": "percentile bootstrap resampling complete sessions",
            "replications": bootstrap_replicates,
            "seed": bootstrap_seed,
            "warning": "Small numbers of days and rare touches make these intervals unstable; forecasts within a session are dependent.",
        },
    }

"""Point-in-time historical calibration for the mathematical first-passage model.

Minute timestamps mark bar END; available_at marks when a completed bar
becomes usable. A forecast references the last published close. Barriers
are fixed to that known reference, NOT to an unknown future entry price.
This module does not execute trades or read datasets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite, log, sqrt
from typing import Literal, Sequence

from quant_research.model.first_passage import BarrierScenario, first_passage_probabilities
from quant_research.model.first_passage_seasonality import (
    IntradayVarianceProfile,
    VolatilityBase,
    fit_intraday_profile,
)

MINUTE = timedelta(minutes=1)
DriftMode = Literal["log_zero", "price_zero"]
VolatilityMethod = Literal["close", "parkinson", "seasonal_close", "seasonal_parkinson"]


@dataclass(frozen=True, slots=True)
class MinuteBar:
    """A regular-session OHLCV minute bar, timestamped at the minute end."""

    timestamp: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    # Optional for legacy synthetic callers; actual minute OHLCV supplies it.
    volume: float | None = None


@dataclass(frozen=True, slots=True)
class Session:
    day: date
    bars: tuple[MinuteBar, ...]


@dataclass(frozen=True, slots=True)
class ForecastResult:
    day: date
    scenario: BarrierScenario
    drift_mode: DriftMode
    decision_at: datetime
    volatility: float
    upper_probability: float
    lower_probability: float
    neither_probability: float
    observed: int  # 0=upper, 1=lower, 2=neither
    volatility_method: VolatilityMethod = "close"
    # Outcome diagnostics only: these MUST NOT be inputs to the forecast.
    forward_close_variance: float | None = None
    forward_range_variance: float | None = None
    decision_bar_index: int | None = None


@dataclass(frozen=True, slots=True)
class Exclusion:
    day: date
    scenario: BarrierScenario
    reason: str
    volatility_method: VolatilityMethod = "close"


class RollingMinuteVolatility:
    """O(1) online volatility from published adjacent-minute price bars.

    `close`: root mean squared one-minute log returns (original M2).
    `parkinson`: square root of the mean one-minute Parkinson variance,
    log(high/low)**2 / (4*log(2)). It uses the *completed* bar's extremes.

    Only adjacent minutes enter the window, so overnight gaps are excluded.
    Neither method uses future bars; both assume continuous intraminute paths.
    Parkinson is an alternative estimator, not a calibrated correction.
    """

    def __init__(
        self,
        *,
        lookback_returns: int = 390,
        min_returns: int = 60,
        method: VolatilityMethod = "close",
    ) -> None:
        if type(lookback_returns) is not int or lookback_returns < 1:
            raise ValueError("lookback_returns must be a positive integer")
        if type(min_returns) is not int or not 1 <= min_returns <= lookback_returns:
            raise ValueError("min_returns must be between 1 and lookback_returns")
        if method not in ("close", "parkinson"):
            raise ValueError("Unknown volatility method")
        self.method = method
        self.lookback_returns = lookback_returns
        self.min_returns = min_returns
        self._squares: deque[float] = deque()
        self._sum_squares = 0.0
        self._last_stamp: datetime | None = None
        self._last_available: datetime | None = None
        self._last_close: float | None = None

    def observe(self, bar: MinuteBar) -> None:
        _validate_bar(bar)
        if self._last_stamp is not None and bar.timestamp <= self._last_stamp:
            raise ValueError("Bars must be strictly ordered")
        if self._last_available is not None and bar.available_at < self._last_available:
            raise ValueError("Bars must be in publication order")
        if self._last_stamp is not None and bar.timestamp - self._last_stamp == MINUTE:
            assert self._last_close is not None
            if self.method == "close":
                ret = log(bar.close) - log(self._last_close)
                square = ret * ret
            else:
                # Use range only after this completed minute bar is published.
                log_range = log(bar.high) - log(bar.low)
                square = log_range * log_range / (4.0 * log(2.0))
            if len(self._squares) == self.lookback_returns:
                self._sum_squares -= self._squares.popleft()
            self._squares.append(square)
            self._sum_squares += square
        self._last_stamp = bar.timestamp
        self._last_available = bar.available_at
        self._last_close = bar.close

    def estimate(self) -> float | None:
        if len(self._squares) < self.min_returns:
            return None
        return sqrt(max(0.0, self._sum_squares) / len(self._squares))


def _validate_bar(bar: MinuteBar) -> None:
    for stamp in (bar.timestamp, bar.available_at):
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Bar timestamps must be UTC-aware")
    if bar.available_at < bar.timestamp:
        raise ValueError("Bar must not be available before its end")
    values = (bar.open, bar.high, bar.low, bar.close)
    if any(not isfinite(v) or v <= 0 for v in values):
        raise ValueError("OHLC prices must be finite and positive")
    if bar.low > min(bar.open, bar.close) or bar.high < max(bar.open, bar.close):
        raise ValueError("Inconsistent OHLC")


def label_first_touch(
    session_bars: Sequence[MinuteBar],
    *,
    reference_price: float,
    decision_at: datetime,
    scenario: BarrierScenario,
) -> tuple[int | None, str | None]:
    """Label barriers measured from last-known close, not future entry open.

    Only bars whose opens are at/after decision_at may contribute. Reject
    any delay to first bar open or missing minute, so the outcome horizon
    corresponds exactly to the mathematical forecast horizon. If both
    barriers touch in the same bar without an opening gap, order is unknown.
    """
    if not isfinite(reference_price) or reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if decision_at.tzinfo is None or decision_at.utcoffset() != timedelta(0):
        raise ValueError("decision_at must be UTC-aware")
    if scenario.horizon_minutes < 1:
        raise ValueError("horizon_minutes must be positive")
    upper = reference_price * (1.0 + scenario.upper_barrier_pct)
    lower = reference_price * (1.0 - scenario.lower_barrier_pct)
    # Strictly greater than the previous bar's END is not necessary here:
    # this is a probability label, not an executable trade-fill convention.
    entry = next(
        (i for i, bar in enumerate(session_bars) if bar.timestamp - MINUTE >= decision_at),
        None,
    )
    if entry is None:
        return None, "incomplete_horizon"
    first_start = session_bars[entry].timestamp - MINUTE
    if first_start != decision_at:
        return None, "non_aligned_start"
    window = session_bars[entry : entry + scenario.horizon_minutes]
    if len(window) != scenario.horizon_minutes:
        return None, "incomplete_horizon"
    for j, bar in enumerate(window):
        if bar.timestamp != first_start + (j + 1) * MINUTE:
            return None, "missing_minutes"
        # The opening price is chronologically before the high and low.
        if bar.open >= upper:
            return 0, None
        if bar.open <= lower:
            return 1, None
        if bar.high >= upper and bar.low <= lower:
            return None, "ambiguous_barriers"
        if bar.high >= upper:
            return 0, None
        if bar.low <= lower:
            return 1, None
    return 2, None


def _forward_variances(
    bars: Sequence[MinuteBar],
    *,
    decision_index: int,
    horizon_minutes: int,
) -> tuple[float, float] | None:
    """Measure EX-POST minute variance over the *forecasted* holding window.

    The first bar starts at ``reference.available_at``, not necessarily at
    ``reference.timestamp``: historical bars can be published after their end.
    Never include the intervening bars in the forward holding window.
    This diagnostic does not influence any forecast or training calculation.
    """
    reference = bars[decision_index]
    decision_at = reference.available_at
    first_index = next(
        (
            i
            for i in range(decision_index + 1, len(bars))
            if bars[i].timestamp - MINUTE >= decision_at
        ),
        None,
    )
    if first_index is None:
        return None
    if bars[first_index].timestamp - MINUTE != decision_at:
        return None
    future = bars[first_index : first_index + horizon_minutes]
    if len(future) != horizon_minutes:
        return None

    # At zero publication delay the reference close is known at the decision.
    # With publication delay, omit the unobserved interval before the first
    # forward bar by starting at its open instead of the stale reference close.
    prev_close = reference.close if decision_at == reference.timestamp else future[0].open
    close_squares = 0.0
    range_squares = 0.0
    for minute, bar in enumerate(future, start=1):
        if bar.timestamp != decision_at + minute * MINUTE:
            return None
        ret = log(bar.close) - log(prev_close)
        close_squares += ret * ret
        log_range = log(bar.high) - log(bar.low)
        range_squares += log_range * log_range / (4.0 * log(2.0))
        prev_close = bar.close
    return close_squares / horizon_minutes, range_squares / horizon_minutes


def evaluate_sessions(
    *,
    sessions: Sequence[Session],
    evaluation_days: set[date],
    scenarios: Sequence[BarrierScenario],
    drift_modes: Sequence[DriftMode] = ("log_zero", "price_zero"),
    decision_stride: int = 62,
    lookback_returns: int = 390,
    min_returns: int = 60,
    volatility_methods: Sequence[VolatilityMethod] = ("close",),
    per_scenario_stride: bool = False,
    seasonality_training_days: set[date] | None = None,
) -> tuple[list[ForecastResult], list[Exclusion]]:
    """Stream historical bars in publication order, never using future prices.

    Scoring uses all earlier sessions as volatility warm-up. Decision dates
    are explicitly supplied by the caller, which must enforce the holdout
    split before loading any session tables.
    """
    if type(decision_stride) is not int or decision_stride < 1:
        raise ValueError("decision_stride must be positive")
    if not scenarios or not drift_modes or not volatility_methods:
        raise ValueError("Scenarios, drift modes, and volatility methods are required")
    if any(
        method not in ("close", "parkinson", "seasonal_close", "seasonal_parkinson")
        for method in volatility_methods
    ):
        raise ValueError("Unknown volatility method")
    if len(set(volatility_methods)) != len(volatility_methods):
        raise ValueError("Duplicate volatility methods")
    if type(per_scenario_stride) is not bool:
        raise ValueError("per_scenario_stride must be boolean")
    if any(mode not in ("log_zero", "price_zero") for mode in drift_modes):
        raise ValueError("Unsupported drift mode")
    if not evaluation_days:
        raise ValueError("No evaluation days")
    if any(sessions[i].day >= sessions[i + 1].day for i in range(len(sessions) - 1)):
        raise ValueError("Sessions must be ordered and unique")

    # The CLI specifies the designated training days to exclude the preceding
    # embargo from parameter fitting. The fallback preserves direct API usage
    # for callers without an explicit training/embargo split.
    seasonal_methods = [m for m in volatility_methods if m.startswith("seasonal_")]
    pre_validation_days = {
        session.day for session in sessions if session.day < min(evaluation_days)
    }
    if seasonality_training_days is not None:
        if not seasonality_training_days or not seasonality_training_days <= pre_validation_days:
            raise ValueError(
                "Seasonality training days must be nonempty and strictly pre-validation"
            )
        training_days = seasonality_training_days
    else:
        training_days = pre_validation_days
    training_sessions = [session for session in sessions if session.day in training_days]
    profiles: dict[VolatilityMethod, IntradayVarianceProfile] = {}
    for method in seasonal_methods:
        profile_base_method: VolatilityBase = "close" if method == "seasonal_close" else "parkinson"
        profiles[method] = fit_intraday_profile(training_sessions, method=profile_base_method)

    def base_method(method: VolatilityMethod) -> VolatilityBase:
        return "close" if method in ("close", "seasonal_close") else "parkinson"

    # Share the base rolling estimators between constant and seasonal models.
    estimators = {
        method: RollingMinuteVolatility(
            lookback_returns=lookback_returns,
            min_returns=min_returns,
            method=method,
        )
        for method in {base_method(selected) for selected in volatility_methods}
    }
    # The mean past seasonal weight corresponds exactly to the adjacent bars
    # in each rolling estimator's squared-return window.
    weight_windows: dict[VolatilityMethod, deque[float]] = {
        method: deque() for method in seasonal_methods
    }
    weight_sums: dict[VolatilityMethod, float] = {method: 0.0 for method in seasonal_methods}
    last_seen_timestamp: datetime | None = None
    forecasts: list[ForecastResult] = []
    exclusions: list[Exclusion] = []
    for session in sessions:
        for i, bar in enumerate(session.bars):
            adjacent = (
                last_seen_timestamp is not None and bar.timestamp - last_seen_timestamp == MINUTE
            )
            for estimator in estimators.values():
                estimator.observe(bar)
            if adjacent:
                for method, profile in profiles.items():
                    weights = weight_windows[method]
                    if len(weights) == lookback_returns:
                        weight_sums[method] -= weights.popleft()
                    factor = profile.factor(i)
                    weights.append(factor)
                    weight_sums[method] += factor
            last_seen_timestamp = bar.timestamp
            if session.day not in evaluation_days:
                continue
            for scenario in scenarios:
                stride = scenario.horizon_minutes + 1 if per_scenario_stride else decision_stride
                if i % stride != 0:
                    continue
                outcome, reason = label_first_touch(
                    session.bars,
                    reference_price=bar.close,
                    decision_at=bar.available_at,
                    scenario=scenario,
                )
                # Identify the first forecasted minute by its scheduled start.
                # This handles publication delays without inspecting future prices.
                first_future_index = (
                    next(
                        (
                            j
                            for j in range(i + 1, len(session.bars))
                            if session.bars[j].timestamp - MINUTE >= bar.available_at
                        ),
                        None,
                    )
                    if outcome is not None
                    else None
                )
                # Ex-post diagnostics are never used in the probability model.
                future_variances = (
                    _forward_variances(
                        session.bars, decision_index=i, horizon_minutes=scenario.horizon_minutes
                    )
                    if outcome is not None
                    else None
                )
                for method in volatility_methods:
                    sigma = estimators[base_method(method)].estimate()
                    if sigma is None:
                        exclusions.append(
                            Exclusion(session.day, scenario, "insufficient_history", method)
                        )
                        continue
                    if outcome is None:
                        assert reason is not None
                        exclusions.append(Exclusion(session.day, scenario, reason, method))
                        continue
                    if method in profiles:
                        weights = weight_windows[method]
                        if not weights:
                            raise AssertionError("Missing rolling variance weights")
                        past_factor = weight_sums[method] / len(weights)
                        future_factor = profiles[method].mean_future_factor(
                            decision_index=(first_future_index - 1)
                            if first_future_index is not None
                            else i,
                            horizon=scenario.horizon_minutes,
                        )
                        # Deterministic variance time-change: aggregate future
                        # variance maps exactly to constant effective sigma.
                        sigma *= sqrt(future_factor / past_factor)
                    for mode in drift_modes:
                        p = first_passage_probabilities(
                            upper_barrier_pct=scenario.upper_barrier_pct,
                            lower_barrier_pct=scenario.lower_barrier_pct,
                            horizon_minutes=scenario.horizon_minutes,
                            volatility_per_sqrt_minute=sigma,
                            drift_mode=mode,
                        )
                        forecasts.append(
                            ForecastResult(
                                session.day,
                                scenario,
                                mode,
                                bar.available_at,
                                sigma,
                                p.upper_first,
                                p.lower_first,
                                p.neither,
                                outcome,
                                method,
                                forward_close_variance=(
                                    future_variances[0] if future_variances else None
                                ),
                                forward_range_variance=(
                                    future_variances[1] if future_variances else None
                                ),
                                decision_bar_index=i,
                            )
                        )
    return forecasts, exclusions


def summarize_forecasts(forecasts: Sequence[ForecastResult]) -> list[dict[str, object]]:
    """Grouped proper scores and 10%-wide upper-probability calibration bins."""
    groups: dict[tuple[BarrierScenario, DriftMode, VolatilityMethod], list[ForecastResult]] = {}
    for row in forecasts:
        groups.setdefault((row.scenario, row.drift_mode, row.volatility_method), []).append(row)
    summaries: list[dict[str, object]] = []
    for (scenario, mode, method), rows in groups.items():
        n = len(rows)
        counts = [sum(row.observed == k for row in rows) for k in range(3)]
        predicted = [
            sum(
                (row.upper_probability, row.lower_probability, row.neither_probability)[k]
                for row in rows
            )
            / n
            for k in range(3)
        ]
        brier = (
            sum(
                sum(
                    (p - float(row.observed == k)) ** 2
                    for k, p in enumerate(
                        (row.upper_probability, row.lower_probability, row.neither_probability)
                    )
                )
                / 3.0
                for row in rows
            )
            / n
        )
        log_loss = (
            -sum(
                log(
                    max(
                        1e-15,
                        (row.upper_probability, row.lower_probability, row.neither_probability)[
                            row.observed
                        ],
                    )
                )
                for row in rows
            )
            / n
        )
        calibration: list[dict[str, float | int]] = []
        for k in range(10):
            bucket = [row for row in rows if min(9, int(row.upper_probability * 10)) == k]
            if bucket:
                calibration.append(
                    {
                        "range_start": k / 10.0,
                        "count": len(bucket),
                        "mean_predicted": sum(row.upper_probability for row in bucket)
                        / len(bucket),
                        "observed_upper_fraction": sum(row.observed == 0 for row in bucket)
                        / len(bucket),
                    }
                )
        # Descriptive volatility strata from the evaluation sample only.
        # They do not affect the point-in-time forecasts or select a policy.
        ordered = sorted(rows, key=lambda row: row.volatility)
        regimes: list[dict[str, object]] = []
        for i, label in enumerate(("low", "middle", "high")):
            subset = ordered[i * n // 3 : (i + 1) * n // 3]
            if not subset:
                continue
            regimes.append(
                {
                    "name": label,
                    "count": len(subset),
                    "volatility_min": subset[0].volatility,
                    "volatility_max": subset[-1].volatility,
                    "observed_upper_fraction": sum(row.observed == 0 for row in subset)
                    / len(subset),
                    "observed_lower_fraction": sum(row.observed == 1 for row in subset)
                    / len(subset),
                    "mean_predicted_upper": sum(row.upper_probability for row in subset)
                    / len(subset),
                    "mean_predicted_lower": sum(row.lower_probability for row in subset)
                    / len(subset),
                }
            )
        summaries.append(
            {
                "upper_barrier_pct": scenario.upper_barrier_pct,
                "lower_barrier_pct": scenario.lower_barrier_pct,
                "horizon_minutes": scenario.horizon_minutes,
                "drift_mode": mode,
                "volatility_method": method,
                "scored": n,
                "scored_sessions": len({row.day for row in rows}),
                "sessions_with_barrier_touch": len({row.day for row in rows if row.observed != 2}),
                "volatility_mean": sum(row.volatility for row in rows) / n,
                "volatility_min": min(row.volatility for row in rows),
                "volatility_max": max(row.volatility for row in rows),
                "observed_any_touch_fraction": (counts[0] + counts[1]) / n,
                "predicted_any_touch_fraction": predicted[0] + predicted[1],
                "observed_counts": {"upper": counts[0], "lower": counts[1], "neither": counts[2]},
                "predicted_mean": {
                    "upper": predicted[0],
                    "lower": predicted[1],
                    "neither": predicted[2],
                },
                "brier": brier,
                "log_loss": log_loss,
                "upper_calibration": calibration,
                "volatility_regimes": regimes,
            }
        )
    return summaries


def select_receipt_window(
    receipt: dict[str, object],
    *,
    ticker: str,
    evaluation_start: date,
    training_sessions: int = 38,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
) -> tuple[list[dict[str, object]], set[date], list[str]]:
    """Return ONLY permitted pinned entries, validation days and withheld dates.

    The last `holdout_sessions` dates and the preceding embargo are never
    included in the returned loadable entries. The evaluation_start date
    itself is also excluded. Callers fit model parameters on only the
    designated training prefix; the pre-validation embargo can warm up
    point-in-time volatility but may not fit seasonal parameters.
    """
    if any(
        type(v) is not int or v < 1
        for v in (training_sessions, validation_sessions, holdout_sessions, embargo_sessions)
    ):
        raise ValueError("All split sizes must be positive integers")
    if receipt.get("schema_version") != 1 or not isinstance(receipt.get("entries"), dict):
        raise ValueError("Expected a schema_version=1 pinned minute receipt")
    entries = receipt["entries"]
    assert isinstance(entries, dict)
    candidates: list[dict[str, object]] = []
    for raw in entries.values():
        if not isinstance(raw, dict):
            raise ValueError("Invalid receipt entry")
        if raw.get("ticker") != ticker:
            continue
        session = date.fromisoformat(str(raw["session"]))
        if session < evaluation_start:
            candidates.append(raw)
    candidates.sort(key=lambda e: str(e["session"]))
    if len({str(e["session"]) for e in candidates}) != len(candidates):
        raise ValueError(f"Duplicate sessions for {ticker}")
    if len({str(e["security_id"]) for e in candidates}) != 1:
        raise ValueError(f"Multiple security IDs for {ticker}")
    required = training_sessions + validation_sessions + holdout_sessions + 2 * embargo_sessions
    if len(candidates) < required:
        raise ValueError(
            f"Need {required} pinned sessions before {evaluation_start}, got {len(candidates)}"
        )
    chosen = candidates[-required:]
    validation_begin = training_sessions + embargo_sessions
    validation_end = validation_begin + validation_sessions
    retained = chosen[:validation_end]
    days = {date.fromisoformat(str(e["session"])) for e in chosen[validation_begin:validation_end]}
    withheld = [str(e["session"]) for e in chosen[validation_end:]]
    return retained, days, withheld

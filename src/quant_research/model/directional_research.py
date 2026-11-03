"""Reusable point-in-time construction and paired scoring of M2/M3 forecasts.

No data loading, broker access, forward-return feature engineering or fitting
on evaluation labels occurs here. Session-block intervals retain correlation
between decisions within the same session.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Sequence

from quant_research.features.market_residual import compute_market_residual
from quant_research.model.conditional_direction import (
    ConditionalDirectionModel,
    ConditionalTouchExample,
)
from quant_research.model.conditional_first_passage import (
    ThreeWayForecast,
    combine_first_passage_direction,
)
from quant_research.model.first_passage_validation import ForecastResult, MinuteBar, Session
from quant_research.model.paired_probability_scoring import (
    PairedProbabilityObservation,
    score_paired_probabilities,
)


@dataclass(frozen=True, slots=True)
class DirectionalResearchRow:
    day: date
    decision_at: datetime
    label_available_at: datetime
    feature: float
    baseline: ThreeWayForecast
    observed: int

    def __post_init__(self) -> None:
        if self.decision_at.tzinfo is None or self.decision_at.utcoffset() != timedelta(0):
            raise ValueError("decision_at must be UTC-aware")
        if (
            self.label_available_at.tzinfo is None
            or self.label_available_at.utcoffset() != timedelta(0)
            or self.label_available_at <= self.decision_at
        ):
            raise ValueError("label must be published after the UTC decision")
        if self.observed not in (0, 1, 2) or not isfinite(self.feature):
            raise ValueError("invalid class or feature")
        if self.baseline.upper_first <= 0 or self.baseline.lower_first <= 0:
            raise ValueError("baseline must assign positive probability to both first touches")

    @property
    def baseline_conditional_upper(self) -> float:
        return self.baseline.upper_first / self.baseline.touch_probability

    def training_touch(self) -> ConditionalTouchExample | None:
        if self.observed == 2:
            return None
        return ConditionalTouchExample(
            decision_at=self.decision_at,
            label_available_at=self.label_available_at,
            feature=self.feature,
            baseline_upper_given_touch=self.baseline_conditional_upper,
            outcome="upper" if self.observed == 0 else "lower",
        )


def build_directional_rows(
    *,
    asset_sessions: Sequence[Session],
    factor_sessions: Sequence[Session],
    forecasts: Sequence[ForecastResult],
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
) -> tuple[DirectionalResearchRow, ...]:
    """Align historical M2 decisions with published asset and market closes.

    At each decision the feature receives only bars whose *timestamp* is at
    or before that decision; compute_market_residual additionally enforces
    available_at. Labels mature only when the final horizon bar is published.
    Every missing/stale/misaligned feature is skipped, not forward-filled.
    """
    if not forecasts:
        return ()
    if not asset_sessions or not factor_sessions:
        raise ValueError("asset and factor sessions are required")
    for sessions in (asset_sessions, factor_sessions):
        days = [session.day for session in sessions]
        if days != sorted(set(days)):
            raise ValueError("sessions must be strictly chronological")
    if {s.day for s in asset_sessions} != {s.day for s in factor_sessions}:
        raise ValueError("factor and asset session days must match")
    asset_days = {session.day for session in asset_sessions}
    sessions_by_day = {session.day: session for session in asset_sessions}
    asset_bars = [bar for session in asset_sessions for bar in session.bars]
    factor_bars = [bar for session in factor_sessions for bar in session.bars]
    asset_times = [bar.timestamp for bar in asset_bars]
    factor_times = [bar.timestamp for bar in factor_bars]
    if asset_times != sorted(set(asset_times)) or factor_times != sorted(set(factor_times)):
        raise ValueError("bars must be strictly ordered")
    final_bars: dict[tuple[date, datetime], MinuteBar] = {
        (session.day, bar.timestamp): bar for session in asset_sessions for bar in session.bars
    }
    results: list[DirectionalResearchRow] = []
    signals: dict[datetime, float | None] = {}
    for forecast in forecasts:
        if forecast.day not in asset_days:
            raise ValueError("forecast session not loaded")
        if forecast.decision_bar_index is None:
            raise ValueError("M3 requires a source decision bar index")
        source_bars = sessions_by_day[forecast.day].bars
        if not 0 <= forecast.decision_bar_index < len(source_bars):
            raise ValueError("invalid source decision bar index")
        if source_bars[forecast.decision_bar_index].available_at != forecast.decision_at:
            raise ValueError("M2 decision is not aligned with source publication")
        if forecast.upper_probability <= 0 or forecast.lower_probability <= 0:
            continue  # A numerically saturated Brownian offset has undefined log-odds.
        last = final_bars.get(
            (
                forecast.day,
                forecast.decision_at + timedelta(minutes=forecast.scenario.horizon_minutes),
            )
        )
        if last is None:
            continue
        if forecast.decision_at not in signals:
            a_stop = bisect_right(asset_times, forecast.decision_at)
            f_stop = bisect_right(factor_times, forecast.decision_at)
            tail = 2 * beta_lookback + return_minutes + 3
            signal = compute_market_residual(
                asset_bars[max(0, a_stop - tail) : a_stop],
                factor_bars[max(0, f_stop - tail) : f_stop],
                decision_at=forecast.decision_at,
                return_minutes=return_minutes,
                beta_lookback=beta_lookback,
                min_beta_observations=min_beta_observations,
            )
            signals[forecast.decision_at] = signal.reversal_score if signal else None
        feature = signals[forecast.decision_at]
        if feature is None:
            continue
        if last.available_at <= forecast.decision_at:
            raise ValueError("label maturity must follow decision")
        results.append(
            DirectionalResearchRow(
                day=forecast.day,
                decision_at=forecast.decision_at,
                label_available_at=last.available_at,
                feature=feature,
                baseline=ThreeWayForecast(
                    forecast.upper_probability,
                    forecast.lower_probability,
                    forecast.neither_probability,
                ),
                observed=forecast.observed,
            )
        )
    return tuple(results)


def predict_paired_directional_rows(
    rows: Sequence[DirectionalResearchRow],
    *,
    intercept_only: ConditionalDirectionModel,
    with_signal: ConditionalDirectionModel,
) -> tuple[PairedProbabilityObservation, ...]:
    """Create side-by-side predictions without accessing or fitting future labels."""
    if intercept_only.uses_feature or not with_signal.uses_feature:
        raise ValueError("expected intercept-only and feature correction models")
    predictions: list[PairedProbabilityObservation] = []
    for row in rows:
        p0 = row.baseline_conditional_upper
        predictions.append(
            PairedProbabilityObservation(
                day=row.day,
                decision_at=row.decision_at,
                observed=row.observed,
                base=row.baseline,
                intercept=combine_first_passage_direction(
                    row.baseline,
                    probability_upper_given_touch=intercept_only.predict_upper_given_touch(
                        p0, row.feature, decision_at=row.decision_at
                    ),
                ),
                signal=combine_first_passage_direction(
                    row.baseline,
                    probability_upper_given_touch=with_signal.predict_upper_given_touch(
                        p0, row.feature, decision_at=row.decision_at
                    ),
                ),
            )
        )
    return tuple(predictions)


def score_directional_models(
    rows: Sequence[DirectionalResearchRow],
    *,
    intercept_only: ConditionalDirectionModel,
    with_signal: ConditionalDirectionModel,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20250113,
) -> dict[str, object]:
    """Score M2.5/intercept/M3 using the model-independent paired scorer."""
    return score_paired_probabilities(
        predict_paired_directional_rows(
            rows, intercept_only=intercept_only, with_signal=with_signal
        ),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )

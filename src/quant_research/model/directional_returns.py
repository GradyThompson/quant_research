"""Development-only directional return research, independent of barrier choice.

Labels use the first minute open STRICTLY after the decision through the
close of the horizon's last minute. These are research price proxies, NOT executable fill estimates.
All features are point-in-time and labels carry an explicit maturity time.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import isfinite, log
from typing import Literal, Sequence

import numpy as np

from quant_research.features.market_residual import compute_market_residual
from quant_research.model.first_passage_validation import Session

MINUTE = timedelta(minutes=1)
ReturnModelKind = Literal["intercept", "factor", "residual", "combined"]
MODEL_KINDS: tuple[ReturnModelKind, ...] = ("intercept", "factor", "residual", "combined")


def _utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


@dataclass(frozen=True, slots=True)
class ReturnResearchRow:
    day: date
    decision_at: datetime
    label_available_at: datetime
    horizon_minutes: int
    realized_log_return: float
    factor_return: float
    residual_z: float

    def __post_init__(self) -> None:
        if not _utc(self.decision_at) or not _utc(self.label_available_at):
            raise ValueError("timestamps must be UTC-aware")
        if self.label_available_at <= self.decision_at:
            raise ValueError("outcome maturity must follow decision")
        if type(self.horizon_minutes) is not int or self.horizon_minutes < 1:
            raise ValueError("invalid horizon")
        if not all(
            isfinite(x) for x in (self.realized_log_return, self.factor_return, self.residual_z)
        ):
            raise ValueError("return and features must be finite")


def build_return_rows(
    *,
    asset_sessions: Sequence[Session],
    factor_sessions: Sequence[Session],
    horizon_minutes: int,
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
    diagnostics: dict[str, int] | None = None,
) -> tuple[ReturnResearchRow, ...]:
    """Construct labeled candidate decisions using an h-minute nonoverlapping grid.

    The decision grid is anchored to each session's first observed bar. A bar's
    timestamp is its END; its interval opens one minute earlier. For positive
    publication lag, the next indexed bar may have already started *before*
    decision_at. Find the first future bar whose START is STRICTLY AFTER decision_at,
    consistent with the existing next-minute entry simulator. Require an
    opening no more than one minute after the decision and contiguous bars.
    No future bar is used in any point-in-time feature. Model fitting must
    separately enforce maturity and training partitions.
    """
    if type(horizon_minutes) is not int or horizon_minutes < 1:
        raise ValueError("horizon_minutes must be positive")
    if not asset_sessions or not factor_sessions:
        raise ValueError("asset and factor sessions required")
    for sessions in (asset_sessions, factor_sessions):
        days = [session.day for session in sessions]
        if days != sorted(set(days)):
            raise ValueError("sessions must be ordered and unique")
    if [s.day for s in asset_sessions] != [s.day for s in factor_sessions]:
        raise ValueError("factor and asset dates must match")

    asset_bars = [b for s in asset_sessions for b in s.bars]
    factor_bars = [b for s in factor_sessions for b in s.bars]
    asset_times = [b.timestamp for b in asset_bars]
    factor_times = [b.timestamp for b in factor_bars]
    if asset_times != sorted(set(asset_times)) or factor_times != sorted(set(factor_times)):
        raise ValueError("bars must be strictly chronological")
    tail = 2 * beta_lookback + return_minutes + 3
    counts: dict[str, int] = {
        "sampled_decisions": 0,
        "insufficient_future_bars": 0,
        "unaligned_future_start": 0,
        "non_contiguous_future": 0,
        "overlapping_future": 0,
        "missing_factor_or_residual_feature": 0,
        "eligible_rows": 0,
    }
    rows: list[ReturnResearchRow] = []
    for session in asset_sessions:
        bars = session.bars
        stamps = [bar.timestamp for bar in bars]
        latest_end: datetime | None = None
        for index in range(0, len(bars), horizon_minutes):
            counts["sampled_decisions"] += 1
            current = bars[index]
            decision_at = current.available_at
            if not _utc(decision_at):
                raise ValueError("minute timestamps must be UTC-aware")

            # The price at a bar OPEN is only eligible if its start is
            # STRICTLY after the signal is available. Equality assumes an
            # impossible zero-latency order at the decision instant. This
            # matches backtesting.minute_entry.simulate_next_minute_entry.
            first = bisect_right(stamps, decision_at + MINUTE, lo=index + 1)
            future = bars[first : first + horizon_minutes]
            if len(future) != horizon_minutes:
                counts["insufficient_future_bars"] += 1
                continue
            first_start = future[0].timestamp - MINUTE
            if not decision_at < first_start <= decision_at + MINUTE:
                counts["unaligned_future_start"] += 1
                continue
            if any(
                b.timestamp - a.timestamp != MINUTE
                for a, b in zip(future, future[1:], strict=False)
            ):
                counts["non_contiguous_future"] += 1
                continue
            if latest_end is not None and future[0].timestamp <= latest_end:
                counts["overlapping_future"] += 1
                continue
            if not _utc(future[-1].available_at):
                raise ValueError("minute timestamps must be UTC-aware")
            if future[-1].available_at <= decision_at:
                raise ValueError("label maturity must follow decision")

            a_stop = bisect_right(asset_times, decision_at)
            f_stop = bisect_right(factor_times, decision_at)
            signal = compute_market_residual(
                asset_bars[max(0, a_stop - tail) : a_stop],
                factor_bars[max(0, f_stop - tail) : f_stop],
                decision_at=decision_at,
                return_minutes=return_minutes,
                beta_lookback=beta_lookback,
                min_beta_observations=min_beta_observations,
            )
            if signal is None:
                counts["missing_factor_or_residual_feature"] += 1
                continue
            if future[0].open <= 0 or future[-1].close <= 0:
                raise ValueError("future prices must be positive")
            rows.append(
                ReturnResearchRow(
                    day=session.day,
                    decision_at=decision_at,
                    label_available_at=future[-1].available_at,
                    horizon_minutes=horizon_minutes,
                    realized_log_return=log(future[-1].close / future[0].open),
                    factor_return=signal.factor_return,
                    residual_z=signal.z_score,
                )
            )
            latest_end = future[-1].timestamp
            counts["eligible_rows"] += 1
    if diagnostics is not None:
        diagnostics.update(counts)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class DirectionalReturnModel:
    kind: ReturnModelKind
    intercept: float
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    train_until: datetime
    training_rows: int
    horizon_minutes: int

    def predict(self, row: ReturnResearchRow) -> float:
        if row.decision_at <= self.train_until or not _utc(row.decision_at):
            raise ValueError("predictions require a decision after training cutoff")
        if row.horizon_minutes != self.horizon_minutes:
            raise ValueError("prediction horizon differs from fitted horizon")
        values = _features(row, self.kind)
        return self.intercept + sum(
            coef * (value - mean) / scale
            for coef, value, mean, scale in zip(
                self.coefficients, values, self.feature_means, self.feature_scales, strict=True
            )
        )


def _features(row: ReturnResearchRow, kind: ReturnModelKind) -> tuple[float, ...]:
    if kind == "intercept":
        return ()
    if kind == "factor":
        return (row.factor_return,)
    if kind == "residual":
        return (row.residual_z,)
    if kind == "combined":
        return (row.factor_return, row.residual_z)
    raise ValueError(f"Unknown model kind: {kind}")


def fit_directional_return_model(
    rows: Sequence[ReturnResearchRow],
    *,
    kind: ReturnModelKind,
    train_until: datetime,
    min_rows: int = 30,
    regularization: float = 10.0,
) -> DirectionalReturnModel:
    """Fit ridge on training-only standardized features; never penalize intercept."""
    if not _utc(train_until) or type(min_rows) is not int or min_rows < 2:
        raise ValueError("invalid training configuration")
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be nonnegative")
    if len(rows) < min_rows:
        raise ValueError("insufficient training rows")
    if any(row.decision_at >= train_until or row.label_available_at > train_until for row in rows):
        raise ValueError("unmatured or out-of-partition training row")
    if len({row.horizon_minutes for row in rows}) != 1:
        raise ValueError("mixed horizons")
    y = np.asarray([row.realized_log_return for row in rows], dtype=np.float64)
    matrix = np.asarray([_features(row, kind) for row in rows], dtype=np.float64)
    if kind == "intercept":
        return DirectionalReturnModel(
            kind, float(y.mean()), (), (), (), train_until, len(rows), rows[0].horizon_minutes
        )
    # NumPy represents the empty feature case differently; the branch above
    # avoids it. This matrix always has shape (n, 1) or (n, 2).
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    centered_y = y - y.mean()
    ridge = standardized.T @ standardized + regularization * np.eye(standardized.shape[1])
    coefficients = np.linalg.solve(ridge, standardized.T @ centered_y)
    return DirectionalReturnModel(
        kind=kind,
        intercept=float(y.mean()),
        coefficients=tuple(float(x) for x in coefficients),
        feature_means=tuple(float(x) for x in means),
        feature_scales=tuple(float(x) for x in scales),
        train_until=train_until,
        training_rows=len(rows),
        horizon_minutes=rows[0].horizon_minutes,
    )

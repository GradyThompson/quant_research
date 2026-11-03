"""Simple, past-only intraday variance seasonality for first-passage forecasts.

Three fixed regular-session periods (minute indices 0:60, 60:300, 300:390).
Profiles are estimated only from sessions BEFORE the first evaluation day.
No price-direction or barrier-outcome parameters are fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import log
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:
    from quant_research.model.first_passage_validation import Session

VolatilityBase = Literal["close", "parkinson"]
MINUTE = timedelta(minutes=1)
PERIOD_EDGES = (0, 60, 300, 390)


def _period(index: int) -> int:
    if index < 60:
        return 0
    if index < 300:
        return 1
    return 2


@dataclass(frozen=True, slots=True)
class IntradayVarianceProfile:
    """Relative variance by RTH period, shrinkage toward flat factor 1."""

    factors: tuple[float, float, float]
    session_count: int
    observation_counts: tuple[int, int, int]

    def factor(self, index: int) -> float:
        return self.factors[_period(index)]

    def mean_future_factor(self, *, decision_index: int, horizon: int) -> float:
        """Mean variance factor for the next `horizon` one-minute bars."""
        if horizon < 1:
            raise ValueError("horizon must be positive")
        start, end = decision_index + 1, decision_index + horizon + 1
        total = 0.0
        for period, factor in enumerate(self.factors):
            left = max(start, PERIOD_EDGES[period])
            right = min(end, PERIOD_EDGES[period + 1])
            if right > left:
                total += (right - left) * factor
        # The validator excludes horizon windows past the actual session end.
        if end > PERIOD_EDGES[-1] or start < 0:
            raise ValueError("Future horizon must stay within the regular session")
        return total / horizon


def fit_intraday_profile(
    sessions: Sequence[Session],
    *,
    method: VolatilityBase,
    shrink_observations: int = 390,
) -> IntradayVarianceProfile:
    """Fit fixed three-bucket volatility seasonality on PRE-evaluation sessions.

    Caller is responsible for passing ONLY historical training sessions. This
    function is deliberately agnostic about validation and holdout dates.
    """
    if method not in ("close", "parkinson"):
        raise ValueError("Unsupported volatility method")
    if type(shrink_observations) is not int or shrink_observations < 0:
        raise ValueError("shrink_observations must be a nonnegative integer")
    if not sessions:
        raise ValueError("Seasonality requires earlier historical sessions")
    totals = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for session in sessions:
        for i in range(1, len(session.bars)):
            previous, current = session.bars[i - 1], session.bars[i]
            if current.timestamp - previous.timestamp != MINUTE:
                continue
            if method == "close":
                value = log(current.close / previous.close) ** 2
            else:
                value = log(current.high / current.low) ** 2 / (4.0 * log(2.0))
            bucket = _period(i)
            totals[bucket] += value
            counts[bucket] += 1
    n = sum(counts)
    if n == 0:
        raise ValueError("No adjacent minute returns in historical sessions")
    mean = sum(totals) / n
    if mean <= 0.0:
        # A flat-price training sample contains no information about seasonality.
        factors = (1.0, 1.0, 1.0)
    else:
        estimated = [
            (total + shrink_observations * mean) / (count + shrink_observations) / mean
            for total, count in zip(totals, counts, strict=True)
        ]
        factors = (estimated[0], estimated[1], estimated[2])
    return IntradayVarianceProfile(
        factors=factors,
        session_count=len(sessions),
        observation_counts=(counts[0], counts[1], counts[2]),
    )

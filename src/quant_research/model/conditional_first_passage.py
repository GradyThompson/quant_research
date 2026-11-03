"""Compose a first-touch probability forecast with a conditional direction model.

Pure probability arithmetic, usable with any first-passage and directional
model. This is NOT a trading/executable-fill probability or a calibration claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from quant_research.model.first_passage import FirstPassageProbabilities


@dataclass(frozen=True, slots=True)
class ThreeWayForecast:
    upper_first: float
    lower_first: float
    neither: float

    def __post_init__(self) -> None:
        probabilities = (self.upper_first, self.lower_first, self.neither)
        if any(not isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("probabilities must be finite and within [0, 1]")
        if not isclose(sum(probabilities), 1.0, abs_tol=1e-10, rel_tol=1e-10):
            raise ValueError("first-touch probabilities must sum to one")

    @property
    def touch_probability(self) -> float:
        return self.upper_first + self.lower_first


def combine_first_passage_direction(
    base: FirstPassageProbabilities | ThreeWayForecast,
    *,
    probability_upper_given_touch: float,
) -> ThreeWayForecast:
    """Keep M2.5's total touch probability, replace its conditional split.

    Assumption to test: the direction model does not change touch/timeout odds.
    Conditioning on a touch is important: P(upper FIRST) is not P(price up).
    """
    if not isfinite(probability_upper_given_touch) or not 0 <= probability_upper_given_touch <= 1:
        raise ValueError("conditional upper probability must be in [0,1]")
    validated = ThreeWayForecast(base.upper_first, base.lower_first, base.neither)
    touch = validated.touch_probability
    upper = touch * probability_upper_given_touch
    return ThreeWayForecast(upper, touch - upper, validated.neither)

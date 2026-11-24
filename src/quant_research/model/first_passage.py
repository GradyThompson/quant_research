"""Finite-horizon first-passage probabilities for log-price Brownian motion.

Let X(t) = log(P(t) / P(0)) = mu*t + sigma*W(t), with sigma measured
per square-root minute. The log barriers are log(1 + upper_pct) and
log(1 - lower_pct). The log_zero convention sets mu = 0; price_zero
sets mu = -sigma**2 / 2, making the price a martingale. No model
parameters are fitted here.

The solution uses the method of images for short normalized horizons and
an absorbing-boundary Fourier series for longer horizons, with drift
corrections for price_zero. Both solve the continuous-path diffusion
problem; they are not approximations from sampling minute closes.

Limitations: constant volatility, continuous paths, and no jumps. Zero
log drift is NOT the same as zero expected price return.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, exp, isfinite, log1p, pi, sin, sqrt
from typing import Literal


@dataclass(frozen=True, slots=True)
class BarrierScenario:
    """Positive percentage barrier distances and a horizon in minutes."""

    upper_barrier_pct: float
    lower_barrier_pct: float
    horizon_minutes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("upper_barrier_pct", self.upper_barrier_pct),
            ("lower_barrier_pct", self.lower_barrier_pct),
        ):
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1")
        if type(self.horizon_minutes) is not int or self.horizon_minutes < 1:
            raise ValueError("horizon_minutes must be a positive integer")


@dataclass(frozen=True, slots=True)
class FirstPassageProbabilities:
    """Mutually exclusive barrier outcomes up to the given horizon."""

    upper_first: float
    lower_first: float
    neither: float


def _short_horizon_hit(distance: float, width: float, scaled_time: float) -> float:
    """Absorbing-interval hitting CDF from reflected Gaussian images.

    `distance` is the positive distance from the starting point to the
    barrier being evaluated; `width` is the log-barrier interval.
    """
    scale = sqrt(2.0 * scaled_time)
    result = 0.0
    # For scaled_time / width**2 < 1/4, images with |k| >= 6 are
    # negligible (< 1e-25). Early exit avoids unnecessary erfc calls.
    for k in range(-6, 7):
        signed_distance = distance + 2.0 * k * width
        term = erfc(abs(signed_distance) / scale)
        result += term if signed_distance > 0.0 else -term
    return result


def _long_horizon_hits(
    lower_distance: float, width: float, scaled_time: float
) -> tuple[float, float]:
    """Integrate boundary flux using the absorbing heat-kernel eigenseries."""
    ratio = lower_distance / width
    alpha = pi * pi * scaled_time / (2.0 * width * width)
    upper_tail = 0.0
    lower_tail = 0.0
    for n in range(1, 1000):
        term = (2.0 / pi) * sin(n * pi * ratio) * exp(-n * n * alpha) / n
        lower_tail += term
        upper_tail += term if n % 2 else -term
        if exp(-n * n * alpha) < 1e-16:
            break
    return ratio - upper_tail, 1.0 - ratio - lower_tail


def _tilted_image_integral(distance: float, scaled_time: float, drift: float) -> float:
    """Integrate exp(-drift**2*t/2) times the one-sided zero-drift flux.

    All quantities are dimensionless: distance is in barrier widths,
    scaled_time is sigma**2 * T / width**2, drift is mu * width / sigma**2.
    """
    positive = abs(distance)
    speed = abs(drift)
    scale = sqrt(2.0 * scaled_time)
    first = exp(-positive * speed) * erfc((positive - speed * scaled_time) / scale)
    second = exp(positive * speed) * erfc((positive + speed * scaled_time) / scale)
    return 0.5 * (first + second)


def _price_neutral_hits(
    *,
    upper_distance: float,
    lower_distance: float,
    scaled_time: float,
    eventual_upper: float,
) -> tuple[float, float]:
    """First-passage CDFs for dS/S = sigma*dW (zero price drift).

    An exponential change of measure multiplies the driftless first-hit
    density at barrier b by exp(mu*b/sigma**2 - mu**2*t/(2*sigma**2)).
    Here mu=-sigma**2/2 in log space; no fitted drift is needed.
    """
    width = upper_distance + lower_distance
    r = lower_distance / width
    scaled_drift = -width / 2.0
    if scaled_time < 0.25:

        def hitting(distance: float, target: float) -> float:
            result = 0.0
            for k in range(-6, 7):
                signed = distance + 2.0 * k
                integral = _tilted_image_integral(signed, scaled_time, scaled_drift)
                result += integral if signed > 0.0 else -integral
            return exp(scaled_drift * target) * result

        return hitting(1.0 - r, 1.0 - r), hitting(r, -r)

    # Finite-horizon CDF = infinite-horizon hitting probability minus
    # the remaining survival-weighted boundary-flux tail.
    decay_scale = pi * pi * scaled_time / 2.0
    drift_decay = exp(-0.5 * scaled_drift * scaled_drift * scaled_time)
    tail_upper = 0.0
    tail_lower = 0.0
    for n in range(1, 1000):
        decay = exp(-n * n * decay_scale)
        term = (
            2.0
            * pi
            * n
            * sin(n * pi * r)
            / (pi * pi * n * n + scaled_drift * scaled_drift)
            * decay
            * drift_decay
        )
        tail_lower += term
        tail_upper += term if n % 2 else -term
        if decay < 1e-16:
            break
    tail_upper *= exp(scaled_drift * (1.0 - r))
    tail_lower *= exp(-scaled_drift * r)
    return eventual_upper - tail_upper, 1.0 - eventual_upper - tail_lower


def first_passage_probabilities(
    *,
    upper_barrier_pct: float,
    lower_barrier_pct: float,
    horizon_minutes: int,
    volatility_per_sqrt_minute: float,
    drift_mode: Literal["log_zero", "price_zero"] = "log_zero",
) -> FirstPassageProbabilities:
    """Probability of upper-first, lower-first, or neither before the deadline.

    Percentages are fractions: `0.02` means +2%, `0.01` means -1%.
    Volatility is the standard deviation of ONE minute's log return, e.g.
    `0.001` means 0.1% per sqrt-minute. A zero volatility returns neither=1.

    `drift_mode="price_zero"` applies log drift -sigma**2 / 2, so the
    *price* is a martingale. The default preserves M1 zero-log-drift.

    Events are continuous-time first touches. In historical OHLC bars a
    same-minute upper/lower double touch may have unknown ordering.
    """
    for name, value in (
        ("upper_barrier_pct", upper_barrier_pct),
        ("lower_barrier_pct", lower_barrier_pct),
    ):
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    if (
        isinstance(horizon_minutes, bool)
        or not isinstance(horizon_minutes, int)
        or horizon_minutes < 1
    ):
        raise ValueError("horizon_minutes must be a positive integer")
    if drift_mode not in ("log_zero", "price_zero"):
        raise ValueError("drift_mode must be log_zero or price_zero")
    sigma = volatility_per_sqrt_minute
    if not isfinite(sigma) or sigma < 0.0:
        raise ValueError("volatility_per_sqrt_minute must be finite and nonnegative")
    if sigma == 0.0:
        return FirstPassageProbabilities(0.0, 0.0, 1.0)

    upper_distance = log1p(upper_barrier_pct)
    lower_distance = -log1p(-lower_barrier_pct)
    width = upper_distance + lower_distance
    # Work in units of barrier width. In particular, this avoids squaring
    # extremely small dollar-free percentage scales before dividing them.
    scaled_stddev = sigma * sqrt(horizon_minutes) / width
    scaled_time = scaled_stddev * scaled_stddev
    if scaled_time == 0.0:  # Numerical underflow: no resolvable movement.
        return FirstPassageProbabilities(0.0, 0.0, 1.0)
    # Images converge near t=0; eigenmodes converge quickly afterward.
    if drift_mode == "price_zero":
        upper, lower = _price_neutral_hits(
            upper_distance=upper_distance,
            lower_distance=lower_distance,
            scaled_time=scaled_time,
            eventual_upper=lower_barrier_pct / (upper_barrier_pct + lower_barrier_pct),
        )
    elif scaled_time < 0.25:
        upper = _short_horizon_hit(upper_distance / width, 1.0, scaled_time)
        lower = _short_horizon_hit(lower_distance / width, 1.0, scaled_time)
    else:
        upper, lower = _long_horizon_hits(lower_distance / width, 1.0, scaled_time)

    # Guard only against roundoff, not genuine numerical inconsistencies.
    if not (
        -1e-12 <= upper <= 1.0 + 1e-12
        and -1e-12 <= lower <= 1.0 + 1e-12
        and upper + lower <= 1.0 + 1e-12
    ):
        raise ArithmeticError("First-passage series did not yield valid probabilities")
    upper = max(0.0, min(1.0, upper))
    lower = max(0.0, min(1.0 - upper, lower))
    return FirstPassageProbabilities(upper, lower, 1.0 - upper - lower)

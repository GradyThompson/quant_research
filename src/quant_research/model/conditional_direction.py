"""Penalized log-odds correction to a mathematically specified first-touch model.

Conditional on a touch, logit P(upper first) = logit P_M2.5(upper first)
+ intercept + coefficient * standardized_signal.  In particular the
zero-correction model recovers the original asymmetric Brownian probabilities.
The likelihood is conditional on a touch; timeout cases do not train direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, log, sqrt
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit


def _utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _logit(probability: float) -> float:
    if not isfinite(probability) or not 0 < probability < 1:
        raise ValueError("baseline conditional probability must be strictly within (0, 1)")
    return log(probability / (1.0 - probability))


@dataclass(frozen=True, slots=True)
class ConditionalTouchExample:
    decision_at: datetime
    label_available_at: datetime
    feature: float
    baseline_upper_given_touch: float
    outcome: Literal["upper", "lower"]

    def __post_init__(self) -> None:
        if not _utc(self.decision_at) or not _utc(self.label_available_at):
            raise ValueError("timestamps must be UTC-aware")
        if self.label_available_at <= self.decision_at:
            raise ValueError("label must mature after decision")
        if not isfinite(self.feature):
            raise ValueError("feature must be finite")
        _logit(self.baseline_upper_given_touch)
        if self.outcome not in ("upper", "lower"):
            raise ValueError("only upper/lower first touches train direction")


@dataclass(frozen=True, slots=True)
class ConditionalDirectionModel:
    intercept: float
    coefficient: float
    feature_mean: float
    feature_scale: float
    train_until: datetime
    training_examples: int
    version: str
    uses_feature: bool

    def __post_init__(self) -> None:
        if not _utc(self.train_until):
            raise ValueError("training cutoff must be UTC-aware")
        if not self.version or self.training_examples < 2 or self.feature_scale <= 0:
            raise ValueError("invalid model metadata")
        if any(
            not isfinite(value)
            for value in (self.intercept, self.coefficient, self.feature_mean, self.feature_scale)
        ):
            raise ValueError("model parameters must be finite")
        if not self.uses_feature and self.coefficient != 0:
            raise ValueError("intercept-only model cannot have feature coefficient")

    def predict_upper_given_touch(
        self, baseline_upper_given_touch: float, feature: float, *, decision_at: datetime
    ) -> float:
        if not _utc(decision_at) or decision_at <= self.train_until:
            raise ValueError("decision must be UTC-aware and after training cutoff")
        if not isfinite(feature):
            raise ValueError("feature must be finite")
        offset = _logit(baseline_upper_given_touch)
        signal = (feature - self.feature_mean) / self.feature_scale
        return float(expit(offset + self.intercept + self.coefficient * signal))


def fit_conditional_direction(
    examples: Sequence[ConditionalTouchExample],
    *,
    train_until: datetime,
    version: str,
    include_feature: bool = True,
    min_examples: int = 30,
    regularization: float = 10.0,
) -> ConditionalDirectionModel:
    """Fit a Bernoulli conditional likelihood with fixed Brownian logit offset.

    The coefficient is ridge-penalized; intercept is an unpenalized correction
    to the unconditional odds. All examples must be from a permitted training
    partition, with outcome publication no later than train_until.
    """
    if not _utc(train_until) or not version:
        raise ValueError("training cutoff must be UTC-aware; version is required")
    if type(min_examples) is not int or min_examples < 2:
        raise ValueError("min_examples must be >= 2")
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if len(examples) < min_examples:
        raise ValueError("insufficient mature first-touch examples")
    if any(ex.decision_at >= train_until or ex.label_available_at > train_until for ex in examples):
        raise ValueError("training example or label is not known before cutoff")
    outcomes = {ex.outcome for ex in examples}
    if outcomes != {"upper", "lower"}:
        raise ValueError("training requires both first-touch classes")

    n = len(examples)
    mean = sum(ex.feature for ex in examples) / n
    scale = max(sqrt(sum((ex.feature - mean) ** 2 for ex in examples) / n), 1e-12)
    x = np.asarray([(ex.feature - mean) / scale for ex in examples], dtype=np.float64)
    y = np.asarray([float(ex.outcome == "upper") for ex in examples], dtype=np.float64)
    offset = np.asarray([_logit(ex.baseline_upper_given_touch) for ex in examples])

    if include_feature:

        def feature_objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            logits = offset + params[0] + params[1] * x
            loss = float(
                np.logaddexp(0.0, logits).sum()
                - np.dot(y, logits)
                + 0.5 * regularization * params[1] ** 2
            )
            error = expit(logits) - y
            gradient = np.asarray(
                [error.sum(), np.dot(error, x) + regularization * params[1]],
                dtype=np.float64,
            )
            return loss, gradient

        objective_fn: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]] = (
            feature_objective
        )
        initial = np.zeros(2, dtype=np.float64)
    else:

        def intercept_objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            logits = offset + params[0]
            loss = float(np.logaddexp(0.0, logits).sum() - np.dot(y, logits))
            error = expit(logits) - y
            return loss, np.asarray([error.sum()], dtype=np.float64)

        objective_fn = intercept_objective
        initial = np.zeros(1, dtype=np.float64)

    result = minimize(
        objective_fn,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"directional correction fit failed: {result.message}")
    return ConditionalDirectionModel(
        intercept=float(result.x[0]),
        coefficient=float(result.x[1]) if include_feature else 0.0,
        feature_mean=mean,
        feature_scale=scale,
        train_until=train_until,
        training_examples=n,
        version=version,
        uses_feature=include_feature,
    )

"""Reusable, regularized binary logistic model for conditional first-touch direction.

M3 predicts P(upper FIRST | either barrier is touched, observed feature).
It does not predict the total probability of a touch or timeout. Training
samples must be constructed from mature historical upper/lower-first labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite, log, sqrt
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True, slots=True)
class DirectionalExample:
    decision_at: datetime
    label_available_at: datetime
    feature: float
    outcome: Literal["upper", "lower"]

    def __post_init__(self) -> None:
        for stamp in (self.decision_at, self.label_available_at):
            if stamp.tzinfo is None or stamp.utcoffset() != UTC.utcoffset(stamp):
                raise ValueError("training times must be UTC-aware")
        if self.label_available_at <= self.decision_at:
            raise ValueError("outcome must mature after decision")
        if not isfinite(self.feature):
            raise ValueError("feature must be finite")
        if self.outcome not in ("upper", "lower"):
            raise ValueError("only observed first touches are used")


@dataclass(frozen=True, slots=True)
class DirectionalAlphaModel:
    """An immutable fitted model. Positive features favor upper-first."""

    intercept: float
    coefficient: float
    feature_mean: float
    feature_scale: float
    train_until: datetime
    training_examples: int
    version: str

    def __post_init__(self) -> None:
        if (
            not isfinite(self.intercept)
            or not isfinite(self.coefficient)
            or not isfinite(self.feature_mean)
            or not isfinite(self.feature_scale)
            or self.feature_scale <= 0
        ):
            raise ValueError("fitted directional parameters must be finite")
        if self.train_until.tzinfo is None or self.train_until.utcoffset() != UTC.utcoffset(
            self.train_until
        ):
            raise ValueError("training cutoff must be UTC-aware")
        if type(self.training_examples) is not int or self.training_examples < 2:
            raise ValueError("training_examples must be >= 2")
        if not self.version:
            raise ValueError("version must be nonempty")

    def predict_upper_given_touch(self, feature: float, *, decision_at: datetime) -> float:
        if not isfinite(feature):
            raise ValueError("feature must be finite")
        if decision_at.tzinfo is None or decision_at.utcoffset() != UTC.utcoffset(decision_at):
            raise ValueError("decision_at must be UTC-aware")
        if decision_at <= self.train_until:
            raise ValueError("predictions must occur after training cutoff")
        standardized = (feature - self.feature_mean) / self.feature_scale
        return float(expit(self.intercept + self.coefficient * standardized))


def fit_directional_alpha(
    examples: Sequence[DirectionalExample],
    *,
    train_until: datetime,
    min_examples: int = 30,
    regularization: float = 10.0,
    version: str,
) -> DirectionalAlphaModel:
    """Fit on labels known by train_until, rejecting any unsealed observations.

    The caller must supply only training-split examples. A held-out/embargo
    example causes a hard failure rather than being silently filtered.
    """
    if train_until.tzinfo is None or train_until.utcoffset() != UTC.utcoffset(train_until):
        raise ValueError("train_until must be UTC-aware")
    if type(min_examples) is not int or min_examples < 2:
        raise ValueError("min_examples must be >= 2")
    if not isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be nonnegative and finite")
    if not version:
        raise ValueError("version must be nonempty")
    if len(examples) < min_examples:
        raise ValueError("insufficient labeled first touches")
    if any(ex.label_available_at > train_until for ex in examples):
        raise ValueError("outcome not yet available at training cutoff")
    positives = sum(ex.outcome == "upper" for ex in examples)
    if positives in (0, len(examples)):
        raise ValueError("training needs both upper and lower first touches")

    n = len(examples)
    mean = sum(ex.feature for ex in examples) / n
    variance = sum((ex.feature - mean) ** 2 for ex in examples) / n
    scale = max(sqrt(variance), 1e-12)
    x = np.asarray([(ex.feature - mean) / scale for ex in examples], dtype=np.float64)
    y = np.asarray([float(ex.outcome == "upper") for ex in examples], dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        logits = params[0] + params[1] * x
        loss = (
            np.logaddexp(0.0, logits).sum()
            - np.dot(y, logits)
            + 0.5 * regularization * params[1] ** 2
        )
        errors = expit(logits) - y
        gradient = np.array(
            [errors.sum(), np.dot(errors, x) + regularization * params[1]],
            dtype=np.float64,
        )
        return float(loss), gradient

    result = minimize(
        objective,
        (log(positives / (n - positives)), 0.0),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"directional model fit failed: {result.message}")
    intercept, coefficient = (float(value) for value in result.x)
    return DirectionalAlphaModel(
        intercept=intercept,
        coefficient=coefficient,
        feature_mean=mean,
        feature_scale=scale,
        train_until=train_until,
        training_examples=n,
        version=version,
    )

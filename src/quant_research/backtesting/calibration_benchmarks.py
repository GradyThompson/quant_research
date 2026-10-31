"""Paired multiclass probability scores and clearly identified diagnostic baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from statistics import mean


@dataclass(frozen=True, slots=True)
class BenchmarkScores:
    count: int
    probabilities: tuple[float, float, float]
    brier_score: float | None
    log_loss: float | None


def score_constant_baseline(
    actual_classes: list[int],
    *,
    probabilities: tuple[float, float, float],
) -> BenchmarkScores:
    """Score a constant forecast on a *fixed* set of multiclass observations.

    Brier uses mean squared error averaged over all three classes, matching
    barrier_calibration. Never use evaluation-period frequencies as a
    train-time or out-of-sample baseline.
    """
    if any(not 0 <= p <= 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-10:
        raise ValueError("Baseline probabilities must be in [0, 1] and sum to 1")
    if any(class_index not in (0, 1, 2) for class_index in actual_classes):
        raise ValueError("Actual class must be 0, 1 or 2")
    if not actual_classes:
        return BenchmarkScores(0, probabilities, None, None)
    brier = mean(
        sum(
            (probability - float(index == actual)) ** 2
            for index, probability in enumerate(probabilities)
        )
        / 3
        for actual in actual_classes
    )
    loss = mean(-log(max(probabilities[actual], 1e-15)) for actual in actual_classes)
    return BenchmarkScores(len(actual_classes), probabilities, brier, loss)


def score_uniform_baseline(actual_classes: list[int]) -> BenchmarkScores:
    return score_constant_baseline(actual_classes, probabilities=(1 / 3, 1 / 3, 1 / 3))


def score_retrospective_frequency_baseline(actual_classes: list[int]) -> BenchmarkScores:
    """In-sample oracle reference ONLY; uses realized evaluation labels."""
    if not actual_classes:
        return score_constant_baseline([], probabilities=(1 / 3, 1 / 3, 1 / 3))
    n = len(actual_classes)
    probabilities = (
        actual_classes.count(0) / n,
        actual_classes.count(1) / n,
        actual_classes.count(2) / n,
    )
    return score_constant_baseline(actual_classes, probabilities=probabilities)

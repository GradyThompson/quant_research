"""Paired OOF scoring of continuous directional-return research forecasts.

MSE and improvements are expressed in squared basis points. Intervals resample
market sessions (not individual observations) and are exploratory only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, sqrt
from typing import Sequence

from quant_research.model.first_passage_diagnostics import session_block_interval

BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class ReturnPrediction:
    day: date
    decision_at: datetime
    realized: float
    predictions: dict[str, float]

    def __post_init__(self) -> None:
        if not isfinite(self.realized) or not all(
            isfinite(value) for value in self.predictions.values()
        ):
            raise ValueError("realized returns and predictions must be finite")
        if self.decision_at.date() != self.day:
            raise ValueError("observation day must match decision timestamp")


def score_return_predictions(
    rows: Sequence[ReturnPrediction], *, bootstrap_replicates: int = 2000
) -> dict[str, object]:
    if not rows:
        raise ValueError("no validation predictions")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("invalid bootstrap count")
    if len({(row.day, row.decision_at) for row in rows}) != len(rows):
        raise ValueError("duplicate validation decisions")
    required = {"intercept", "factor", "residual", "combined"}
    if any(set(row.predictions) != required for row in rows):
        raise ValueError("all alternatives must predict the same decisions")
    errors: dict[str, list[float]] = {kind: [] for kind in required}
    paired: dict[str, dict[date, list[float]]] = {
        name: defaultdict(list) for name in ("factor", "residual", "combined")
    }
    for row in rows:
        for model in required:
            errors[model].append(((row.realized - row.predictions[model]) * BPS) ** 2)
        for model in paired:
            paired[model][row.day].append(errors["intercept"][-1] - errors[model][-1])

    def correlation(model: str) -> float | None:
        x = [row.predictions[model] for row in rows]
        y = [row.realized for row in rows]
        x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
        numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
        xx = sum((a - x_mean) ** 2 for a in x)
        yy = sum((b - y_mean) ** 2 for b in y)
        return numerator / sqrt(xx * yy) if xx > 1e-30 and yy > 1e-30 else None

    return {
        "scored": len(rows),
        "scored_sessions": len({row.day for row in rows}),
        "models": {
            model: {
                "mse_bps2": sum(errors[model]) / len(rows),
                "rmse_bps": sqrt(sum(errors[model]) / len(rows)),
                "prediction_realization_correlation": correlation(model),
            }
            for model in sorted(required)
        },
        "intercept_minus_candidate_mse_bps2": {
            model: sum(sum(values) for values in by_day.values()) / len(rows)
            for model, by_day in paired.items()
        },
        "intercept_minus_candidate_95pct_session_bootstrap": {
            model: session_block_interval(
                dict(by_day), replications=bootstrap_replicates, seed=20250113
            )
            for model, by_day in paired.items()
        },
    }

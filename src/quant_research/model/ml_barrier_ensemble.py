"""Four complementary research forecasts for three-outcome barrier trading.

No fitted model sees validation labels. Ensemble weights are predeclared.
Timeout P&L uses a separate regression conditional on neither barrier touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from quant_research.model.ml_barrier import BPS, BarrierRow

CLASSES = (0, 1, 2)
ALTERNATIVES = ("math", "logistic", "direct", "decomposed", "correction", "ensemble")


def _clip_normalize(prob: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(prob, dtype=float), 1e-6, 1.0)
    return result / result.sum(axis=1, keepdims=True)


def _prior(y: np.ndarray, *, classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=classes).astype(float) + 1.0
    return counts / counts.sum()


class _ClassificationHead:
    def __init__(self, *, classes: int, direct: bool, seed: int) -> None:
        self.classes = classes
        self.direct = direct
        self.seed = seed
        self.model: Any = None
        self.constant: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) != len(y) or len(y) == 0:
            raise ValueError("invalid training data")
        if len(np.unique(y)) < 2 or min(np.bincount(y, minlength=self.classes)) < 5:
            self.constant = _prior(y, classes=self.classes)
            return
        if self.direct:
            self.model = LGBMClassifier(
                n_estimators=100,
                num_leaves=15,
                max_depth=5,
                learning_rate=0.04,
                min_child_samples=30,
                reg_lambda=10.0,
                random_state=self.seed,
                verbosity=-1,
                n_jobs=2,
            )
        else:
            self.model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.05,
                    max_iter=300,
                    random_state=self.seed,
                ),
            )
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.tile(self.constant, (len(x), 1))
        if self.model is None:
            raise RuntimeError("head must be fit before predict")
        result = np.zeros((len(x), self.classes))
        probabilities = self.model.predict_proba(x)
        for i, label in enumerate(self.model.classes_):
            result[:, int(label)] = probabilities[:, i]
        return _clip_normalize(result)


@dataclass(frozen=True, slots=True)
class Predictions:
    probabilities: dict[str, np.ndarray]
    timeout_bps: np.ndarray
    empirical_lower_bps: float


def forecast_all(
    train: Sequence[BarrierRow], test: Sequence[BarrierRow], *, seed: int = 2026
) -> Predictions:
    """Fit independent touch, direction, timeout and direct-class heads.

    An untrained/unobserved outcome head falls back to a Laplace-smoothed
    training prior. No validation outcomes are used for calibration or weights.
    """
    if not train or not test:
        raise ValueError("train and test must not be empty")
    x_train = np.asarray([r.features for r in train], dtype=float)
    x_test = np.asarray([r.features for r in test], dtype=float)
    y = np.asarray([r.outcome for r in train], dtype=int)
    math = _clip_normalize(np.asarray([r.baseline for r in test], dtype=float))
    logistic = _ClassificationHead(classes=3, direct=False, seed=seed)
    logistic.fit(x_train, y)
    direct = _ClassificationHead(classes=3, direct=True, seed=seed)
    direct.fit(x_train, y)
    touch = _ClassificationHead(classes=2, direct=True, seed=seed)
    touch.fit(x_train, (y != 2).astype(int))
    touch_idx = np.flatnonzero(y != 2)
    direction = _ClassificationHead(classes=2, direct=True, seed=seed)
    if len(touch_idx):
        direction.fit(x_train[touch_idx], (y[touch_idx] == 0).astype(int))
        conditional_upper = direction.predict(x_test)[:, 1]
    else:
        conditional_upper = math[:, 0] / np.maximum(math[:, 0] + math[:, 1], 1e-6)
    touch_prob = touch.predict(x_test)[:, 1]
    decomposed = _clip_normalize(
        np.column_stack(
            (
                touch_prob * conditional_upper,
                touch_prob * (1.0 - conditional_upper),
                1.0 - touch_prob,
            )
        )
    )
    predicted = {
        "math": math,
        "logistic": logistic.predict(x_test),
        "direct": direct.predict(x_test),
        "decomposed": decomposed,
    }
    # Fixed (not validation-tuned) likelihood-ratio correction against M2
    # baseline. Unlike M3, all three outcomes, including timeout, can move.
    unconditional = _prior(y, classes=3)
    ratio = np.clip(predicted["direct"] / unconditional[None, :], 1e-6, 1e6)
    predicted["correction"] = _clip_normalize(math * np.sqrt(ratio))
    predicted["ensemble"] = _clip_normalize(
        0.15 * math
        + 0.25 * decomposed
        + 0.25 * predicted["correction"]
        + 0.35 * predicted["direct"]
    )
    timeout_mask = y == 2
    if int(timeout_mask.sum()) >= 25:
        timeout_model = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
        timeout_model.fit(
            x_train[timeout_mask],
            np.asarray([r.timeout_return_bps for r in train])[timeout_mask],
        )
        timeout_prediction = np.asarray(timeout_model.predict(x_test), dtype=float)
    else:
        timeout_mean = (
            float(np.mean([r.timeout_return_bps for r in train if r.outcome == 2]))
            if any(timeout_mask)
            else 0.0
        )
        timeout_prediction = np.full(len(test), timeout_mean)
    # Wins are capped at +U; losses can gap below the stop. Empirical
    # conditional lower loss is used for expected payoff, not exact -D.
    losses = [r.gross_return_bps for r in train if r.outcome == 1]
    lower_bps = min(-200.0, float(np.mean(losses))) if losses else -200.0
    return Predictions(
        probabilities=predicted,
        timeout_bps=np.clip(timeout_prediction, -200.0, 200.0),
        empirical_lower_bps=lower_bps,
    )


def score_predictions(
    *,
    rows: Sequence[BarrierRow],
    pred: Predictions,
    cost_bps: tuple[float, ...],
    positions: int,
    edge_margin_bps: float,
    upper_bps: float = 50.0,
    notional_dollars: float = 10_000.0,
) -> dict[str, Any]:
    """Chronological per-session portfolio slots; select using base cost only.

    Fixed same-session selected positions across cost sensitivity scenarios.
    Cash remains idle for no-trade slots, and each day's positions close that
    session. Returns use observed arithmetic price-proxy payoffs, not log P&L.
    """
    if (
        positions < 1
        or not cost_bps
        or not rows
        or notional_dollars <= 0
        or any(c < 0 or not isfinite(c) for c in cost_bps)
    ):
        raise ValueError("invalid scoring configuration")
    days = sorted({r.day for r in rows})
    days_arr = np.asarray([r.day for r in rows])
    result: dict[str, Any] = {}
    y = np.asarray([r.outcome for r in rows])
    actual = np.asarray([r.gross_return_bps for r in rows])
    # IBKR Pro Tiered approximation: 0.0035 USD/share/side; minimum 0.35
    # USD/order, excluding regulatory/venue fees and any account discounts.
    # Execution scenarios below represent ADDITIONAL round-trip friction.
    shares = np.asarray([notional_dollars / r.entry_price for r in rows])
    commission_bps = (2 * np.maximum(0.35, 0.0035 * shares) / notional_dollars) * BPS

    for model, probabilities in pred.probabilities.items():
        if len(probabilities) != len(rows) or probabilities.shape[1] != 3:
            raise ValueError("misaligned probabilities")
        indicator = np.eye(3)[y]
        brier = float(np.mean(np.sum((probabilities - indicator) ** 2, axis=1) / 3))
        log_loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-10, 1.0))))
        expected = (
            probabilities[:, 0] * upper_bps
            + probabilities[:, 1] * pred.empirical_lower_bps
            + probabilities[:, 2] * pred.timeout_bps
        )
        # Explicitly fixed selection decision at base cost plus required margin.
        selected: list[int] = []
        for day in days:
            picks = [
                int(index)
                for index in np.flatnonzero(days_arr == day)
                if expected[int(index)] - cost_bps[0] - commission_bps[int(index)] > edge_margin_bps
            ]
            picks.sort(key=lambda i: (-expected[i], rows[i].ticker))
            selected.extend(picks[:positions])
        detail: dict[str, Any] = {}
        for cost in cost_bps:
            day_scores = {day.isoformat(): 0.0 for day in days}
            trades: list[dict[str, Any]] = []
            for i in selected:
                net = float(actual[i] - cost - commission_bps[i])
                day_scores[rows[i].day.isoformat()] += net / positions
                trades.append(
                    {
                        "day": rows[i].day.isoformat(),
                        "ticker": rows[i].ticker,
                        "expected_gross_bps": float(expected[i]),
                        "gross_bps": float(actual[i]),
                        "net_bps": net,
                        "commission_bps": float(commission_bps[i]),
                        "outcome": int(y[i]),
                    }
                )
            detail[str(cost)] = {
                "trades": len(selected),
                "sessions": len(days),
                "net_bps_per_session": float(sum(day_scores.values()) / len(days)),
                "daily_bps": day_scores,
                "trades_detail": trades,
            }
        result[model] = {
            "brier": brier,
            "log_loss": log_loss,
            "predicted_upper_mean": float(probabilities[:, 0].mean()),
            "observed_upper_frequency": float((y == 0).mean()),
            "predicted_lower_mean": float(probabilities[:, 1].mean()),
            "observed_lower_frequency": float((y == 1).mean()),
            "predicted_timeout_mean": float(probabilities[:, 2].mean()),
            "observed_timeout_frequency": float((y == 2).mean()),
            "cost_scenarios": detail,
        }
    return result

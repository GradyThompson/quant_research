"""Pure chronological splits for separately fitted baseline and alpha models.

Unlike row-label walk-forward, these splits allocate entire market sessions
and isolate baseline calibration, later alpha training, embargo and evaluation.
No data access or labels are required to construct the partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True, slots=True)
class StagedSessionFold:
    number: int
    baseline_fit_days: tuple[date, ...]
    alpha_fit_days: tuple[date, ...]
    embargo_days: tuple[date, ...]
    validation_days: tuple[date, ...]

    @property
    def permitted_days(self) -> tuple[date, ...]:
        return (
            *self.baseline_fit_days,
            *self.alpha_fit_days,
            *self.embargo_days,
            *self.validation_days,
        )


def staged_session_folds(
    sessions: Sequence[date],
    *,
    baseline_fit_sessions: int = 19,
    alpha_fit_sessions: int = 10,
    embargo_sessions: int = 1,
    validation_sessions: int = 5,
    step_sessions: int = 5,
) -> tuple[StagedSessionFold, ...]:
    """Generate forward-only folds with a frozen earliest baseline fit prefix.

    The alpha fit window rolls toward validation; previous validation dates
    may enter later alpha training only after their outcomes have matured.
    Both model fits precede the embargo. No dates beyond the supplied set are
    examined. The caller must supply ONLY pre-holdout session dates.
    """
    sizes = (
        baseline_fit_sessions,
        alpha_fit_sessions,
        embargo_sessions,
        validation_sessions,
        step_sessions,
    )
    if any(type(size) is not int or size < 1 for size in sizes):
        raise ValueError("all session counts must be positive integers")
    if step_sessions < validation_sessions:
        raise ValueError("step_sessions must not overlap validation sessions")
    days = tuple(sessions)
    if any(days[i] >= days[i + 1] for i in range(len(days) - 1)):
        raise ValueError("session dates must be strictly increasing")
    first_eval = baseline_fit_sessions + alpha_fit_sessions + embargo_sessions
    folds: list[StagedSessionFold] = []
    for eval_start in range(first_eval, len(days) - validation_sessions + 1, step_sessions):
        alpha_end = eval_start - embargo_sessions
        alpha_start = alpha_end - alpha_fit_sessions
        if alpha_start < baseline_fit_sessions:
            continue
        fold = StagedSessionFold(
            number=len(folds) + 1,
            baseline_fit_days=days[:baseline_fit_sessions],
            alpha_fit_days=days[alpha_start:alpha_end],
            embargo_days=days[alpha_end:eval_start],
            validation_days=days[eval_start : eval_start + validation_sessions],
        )
        folds.append(fold)
    return tuple(folds)

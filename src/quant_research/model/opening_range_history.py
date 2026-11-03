"""Fixed-rule multi-session opening-range research: no fitting or execution.

Keeps trading decisions distinct from ex-post scoring. Reports yearly stability,
position exposure and per-session outcomes without modifying the original model.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from quant_research.model.first_passage_validation import Session, select_receipt_window
from quant_research.model.opening_range import (
    VARIANTS,
    build_opening_range_trades,
    score_breakout_trades,
)


def evaluate_opening_range_history(
    *,
    asset_sessions: Sequence[Session],
    factor_sessions: Sequence[Session],
    validation_start: date,
    horizons: tuple[int, ...] = (30, 60),
    costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    minimum_warmup_sessions: int = 10,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Score fixed ORB hypotheses only after validation_start.

    Prior sessions can supply past opening-volume statistics, but their outcomes
    never enter the validation score. Requiring whole sessions avoids intraday
    overlap problems. The caller is responsible for protected-date exclusion.
    """
    if (
        not horizons
        or any(h not in (30, 60) for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be unique 30/60 minute values")
    if not costs_bps or any(c < 0 for c in costs_bps) or len(set(costs_bps)) != len(costs_bps):
        raise ValueError("cost scenarios must be nonnegative and unique")
    days = tuple(session.day for session in asset_sessions)
    if len(days) != len(set(days)) or list(days) != sorted(days):
        raise ValueError("asset session days must be sorted and unique")
    if tuple(session.day for session in factor_sessions) != days:
        raise ValueError("factor sessions must align with asset sessions")
    eligible = tuple(day for day in days if day >= validation_start)
    if len(days) - len(eligible) < minimum_warmup_sessions:
        raise ValueError("insufficient strictly earlier warmup sessions")
    if len(eligible) < 2:
        raise ValueError("need at least two validation sessions")
    experiments: list[dict[str, Any]] = []
    for horizon in horizons:
        diagnostics: dict[str, int] = {}
        variants = build_opening_range_trades(
            asset_sessions=asset_sessions,
            factor_sessions=factor_sessions,
            horizon_minutes=horizon,
            diagnostics=diagnostics,
        )
        variant_results: dict[str, dict[str, Any]] = {}
        for name in VARIANTS:
            trades = tuple(t for t in variants[name] if t.day >= validation_start)
            by_year: list[dict[str, Any]] = []
            for year in sorted({d.year for d in eligible}):
                selected_days = tuple(d for d in eligible if d.year == year)
                selected_trades = [t for t in trades if t.day.year == year]
                by_year.append(
                    {
                        "year": year,
                        "sessions": len(selected_days),
                        "cost_sensitivity": [
                            score_breakout_trades(
                                selected_trades,
                                eligible_days=selected_days,
                                roundtrip_cost_bps=cost,
                                bootstrap_replicates=bootstrap_replicates,
                            )
                            for cost in costs_bps
                        ],
                    }
                )
            variant_results[name] = {
                "cost_sensitivity": [
                    score_breakout_trades(
                        trades,
                        eligible_days=eligible,
                        roundtrip_cost_bps=cost,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                    for cost in costs_bps
                ],
                "by_year": by_year,
                "trade_examples": [
                    {
                        "day": t.day.isoformat(),
                        "side": "long" if t.side > 0 else "short",
                        "entry_at": t.entry_at.isoformat(),
                        "exit_available_at": t.exit_available_at.isoformat(),
                        "gross_bps": t.gross_bps,
                    }
                    for t in trades
                ],
            }
        experiments.append(
            {
                "horizon_minutes": horizon,
                "diagnostics": diagnostics,
                "validation_sessions": len(eligible),
                "variants": variant_results,
            }
        )
    return {
        "model_family": "fixed-opening-range-continuation-long-history",
        "research_only": True,
        "execution_model": "minute open/close price proxy; not fillable quotations",
        "validation_start": validation_start.isoformat(),
        "warmup_sessions": len(days) - len(eligible),
        "validation_sessions": len(eligible),
        "experiments": experiments,
        "not_portfolio_pnl": True,
    }


def protected_research_days(original_receipt: dict[str, object]) -> set[date]:
    """Frozen original embargo/holdout plus original evaluation session."""
    _, _, protected = select_receipt_window(
        original_receipt,
        ticker="AAPL",
        evaluation_start=date(2025, 1, 13),
        training_sessions=38,
        validation_sessions=10,
        holdout_sessions=10,
        embargo_sessions=1,
    )
    return {date.fromisoformat(d) for d in protected} | {date(2025, 1, 13)}

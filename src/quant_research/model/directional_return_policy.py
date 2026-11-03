"""Cost-aware, out-of-fold directional-return *proxy* policy diagnostics.

One unit of notional may be long, short, or in cash for each independent
horizon window. Return labels are open-to-close historical price proxies,
not executable fills. No portfolio leverage, short availability, fills, or
concurrent cross-security position accounting is simulated here.

All thresholds and costs must be declared before examining validation results.
This module never fits a model, searches a threshold, or reads market data.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from math import isfinite
from typing import Sequence

from quant_research.model.directional_return_scoring import BPS, ReturnPrediction
from quant_research.model.first_passage_diagnostics import session_block_interval


def score_return_policy(
    rows: Sequence[ReturnPrediction],
    *,
    roundtrip_cost_bps: float,
    excess_edge_bps: float = 2.0,
    bootstrap_replicates: int = 2000,
) -> dict[str, object]:
    """Evaluate fixed sign-of-forecast trades after an assumed round-trip cost.

    Rule: LONG if predicted_bps > cost + margin; SHORT if below the
    corresponding negative threshold; otherwise remain in cash. The
    prediction must be available at decision time, and the realized label
    must correspond to an entry strictly after that time (caller-enforced).
    All rows must be chronologically distinct; there is no model selection.

    Gross P&L is signed log price return times 10,000. A flat round-trip
    cost is deducted per trade (irrespective of side). Nontrade observations
    contribute zero to the per-opportunity return and its bootstrap CI.
    """
    if not rows:
        raise ValueError("no out-of-fold predictions")
    if any(not isfinite(v) or v < 0 for v in (roundtrip_cost_bps, excess_edge_bps)):
        raise ValueError("cost and margin must be nonnegative finite numbers")
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if len({(r.day, r.decision_at) for r in rows}) != len(rows):
        raise ValueError("duplicate out-of-fold decisions")
    expected = set(rows[0].predictions)
    if not expected or any(set(r.predictions) != expected for r in rows):
        raise ValueError("every model must predict the same decisions")

    threshold = roundtrip_cost_bps + excess_edge_bps
    models: dict[str, dict[str, object]] = {}
    for model in sorted(expected):
        by_day_opportunity: dict[date, list[float]] = defaultdict(list)
        by_day_trade: dict[date, list[float]] = defaultdict(list)
        trade_gross: list[float] = []
        trade_net: list[float] = []
        long_count = 0
        short_count = 0
        for row in rows:
            forecast_bps = row.predictions[model] * BPS
            side = 1 if forecast_bps > threshold else -1 if forecast_bps < -threshold else 0
            pnl = 0.0
            if side:
                gross = side * row.realized * BPS
                pnl = gross - roundtrip_cost_bps
                trade_gross.append(gross)
                trade_net.append(pnl)
                by_day_trade[row.day].append(pnl)
                if side > 0:
                    long_count += 1
                else:
                    short_count += 1
            by_day_opportunity[row.day].append(pnl)

        n_trades = len(trade_net)
        net_total = sum(trade_net)
        traded_sessions = len(by_day_trade)
        models[model] = {
            "opportunities": len(rows),
            "sessions": len(by_day_opportunity),
            "trades": n_trades,
            "traded_sessions": traded_sessions,
            "long_trades": long_count,
            "short_trades": short_count,
            "participation": n_trades / len(rows),
            "gross_bps_per_trade": sum(trade_gross) / n_trades if n_trades else None,
            "net_bps_per_trade": net_total / n_trades if n_trades else None,
            "net_bps_per_opportunity": net_total / len(rows),
            "total_net_bps_unit_notional": net_total,
            "win_fraction_net": sum(n > 0 for n in trade_net) / n_trades if n_trades else None,
            "net_bps_per_opportunity_95pct_session_bootstrap": session_block_interval(
                dict(by_day_opportunity), replications=bootstrap_replicates, seed=20250113
            ),
            "net_bps_per_trade_95pct_session_bootstrap": (
                session_block_interval(
                    dict(by_day_trade), replications=bootstrap_replicates, seed=20250113
                )
                if traded_sessions >= 2
                else None
            ),
        }
    return {
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "excess_edge_bps": excess_edge_bps,
        "entry_rule": "abs(forecast_bps) > roundtrip_cost_bps + excess_edge_bps",
        "exit_rule": "research label horizon close; no executable exit simulation",
        "baseline_cash_net_bps_per_opportunity": 0.0,
        "proxy_not_executable": True,
        "models": models,
    }

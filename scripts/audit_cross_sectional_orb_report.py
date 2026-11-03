"""Audit an existing P4.5 report without touching historical source or holdout.

This is a diagnostic research script. It cannot establish executable P&L.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class AuditTrade:
    day: str
    ticker: str
    side: int
    decision_at: str
    entry_at: str
    gross_bps: float
    rvol: float

    @classmethod
    def read(cls, raw: dict[str, Any]) -> AuditTrade:
        result = cls(
            day=str(raw["day"]),
            ticker=str(raw["ticker"]),
            side=int(raw["side"]),
            decision_at=str(raw["decision_at"]),
            entry_at=str(raw["entry_at"]),
            gross_bps=float(raw["gross_bps"]),
            rvol=float(raw["rvol"]),
        )
        if result.side not in (-1, 1) or not isfinite(result.gross_bps):
            raise ValueError("invalid selected trade")
        return result

    @property
    def key(self) -> tuple[str, str, int, str, str]:
        return (self.day, self.ticker, self.side, self.decision_at, self.entry_at)


def _interval(values: list[float], *, repeats: int, seed: int) -> list[float]:
    if not values or repeats < 1:
        raise ValueError("invalid bootstrap inputs")
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(repeats)
    )
    lo = draws[int((repeats - 1) * 0.025)]
    hi = draws[int((repeats - 1) * 0.975)]
    return [lo, hi]


def _trades(cohort: dict[str, Any]) -> list[AuditTrade]:
    return [AuditTrade.read(item) for item in cohort["trades"]]


def _daily_returns(
    trades: list[AuditTrade],
    days: list[str],
    slots: int,
    cost: float,
) -> dict[str, float]:
    if slots < 1 or cost < 0:
        raise ValueError("invalid positions or cost")
    daily = dict.fromkeys(days, 0.0)
    for t in trades:
        if t.day not in daily:
            raise ValueError(f"trade on unscored day: {t.day}")
        daily[t.day] += (t.gross_bps - cost) / slots
    return daily


def _summary(
    trades: list[AuditTrade],
    days: list[str],
    slots: int,
    cost: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    daily = _daily_returns(trades, days, slots, cost)
    totals = [daily[d] for d in days]
    per_trade_net = [t.gross_bps - cost for t in trades]
    top_trades = sorted(trades, key=lambda t: t.gross_bps, reverse=True)
    by_side: dict[str, dict[str, float | int | None]] = {}
    for side, label in ((1, "long"), (-1, "short")):
        side_trades = [t for t in trades if t.side == side]
        by_side[label] = {
            "trades": len(side_trades),
            "net_bps_per_trade": (
                mean(t.gross_bps - cost for t in side_trades) if side_trades else None
            ),
            "net_contribution_bps_per_session": (
                sum(t.gross_bps - cost for t in side_trades) / (len(days) * slots)
            ),
        }
    removal: dict[str, float] = {}
    for remove in (1, 3, 5):
        removed = {t.key for t in top_trades[:remove]}
        surviving = [t for t in trades if t.key not in removed]
        removal[f"exclude_best_{remove}"] = mean(
            _daily_returns(surviving, days, slots, cost).values()
        )
    by_month: dict[str, dict[str, int | float]] = {}
    for month in sorted({d[:7] for d in days}):
        month_days = [d for d in days if d.startswith(month)]
        month_trades = [t for t in trades if t.day.startswith(month)]
        by_month[month] = {
            "sessions": len(month_days),
            "trades": len(month_trades),
            "net_bps_per_session": mean(daily[d] for d in month_days),
        }
    return {
        "sessions": len(days),
        "trades": len(trades),
        "active_sessions": len({t.day for t in trades}),
        "unique_symbols": len({t.ticker for t in trades}),
        "gross_bps_per_trade": mean(t.gross_bps for t in trades) if trades else None,
        "net_bps_per_trade": mean(per_trade_net) if trades else None,
        "net_bps_per_session": mean(totals),
        "session_bootstrap_95pct": _interval(totals, repeats=repeats, seed=seed),
        "by_side": by_side,
        "by_month": by_month,
        "after_removing_best_trades_bps_per_session": removal,
        "best_trades": [
            {
                "day": t.day,
                "ticker": t.ticker,
                "side": t.side,
                "gross_bps": t.gross_bps,
                "net_bps": t.gross_bps - cost,
                "rvol": t.rvol,
            }
            for t in top_trades[:5]
        ],
        "daily": daily,
    }


def _comparison(
    a: list[AuditTrade],
    b: list[AuditTrade],
    days: list[str],
    slots: int,
    cost: float,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    a_keys = {t.key for t in a}
    b_keys = {t.key for t in b}
    daily_a = _daily_returns(a, days, slots, cost)
    daily_b = _daily_returns(b, days, slots, cost)
    diff = [daily_a[d] - daily_b[d] for d in days]
    return {
        "shared_trades": len(a_keys & b_keys),
        "first_only_trades": len(a_keys - b_keys),
        "second_only_trades": len(b_keys - a_keys),
        "overlap_fraction_of_first": len(a_keys & b_keys) / len(a_keys) if a_keys else None,
        "paired_difference_net_bps_per_session": mean(diff),
        "paired_session_bootstrap_95pct": _interval(diff, repeats=repeats, seed=seed),
    }


def audit_report(
    report: dict[str, Any],
    *,
    cost: float = 12.0,
    repeats: int = 4000,
    seed: int = 20250113,
) -> dict[str, Any]:
    if report.get("model_family") != "cross-sectional-abnormal-volume-opening-range":
        raise ValueError("unexpected report family")
    cfg = report["configuration"]
    if report.get("holdout_accessed") is not False:
        raise ValueError("holdout state must be explicitly false")
    slots = int(cfg["max_positions"])
    days = sorted(report["candidate_counts"])
    if len(days) != int(cfg["scored_sessions"]):
        raise ValueError("scored days do not match candidate_counts")
    if not days or len(set(days)) != len(days):
        raise ValueError("invalid scored days")
    if any(date.fromisoformat(day) < date.fromisoformat(cfg["validation_start"]) for day in days):
        raise ValueError("a scored day predates validation start")
    result: dict[str, Any] = {
        "source_report_model_family": report["model_family"],
        "scored_sessions": len(days),
        "cost_bps": cost,
        "bootstrap_replicates": repeats,
        "research_only": True,
        "note": "Trade-proxy diagnostics; reuses development results; no new validation.",
        "horizons": {},
    }
    for horizon, obj in sorted(report["results"].items(), key=lambda x: int(x[0])):
        cohorts = obj["cohorts"]
        cohort_trades = {name: _trades(value) for name, value in cohorts.items()}
        summaries = {}
        for name, trades in cohort_trades.items():
            diag = _summary(trades, days, slots, cost, repeats, seed)
            published = next(
                (
                    scenario
                    for scenario in cohorts[name]["cost_sensitivity"]
                    if float(scenario["roundtrip_cost_bps"]) == cost
                ),
                None,
            )
            if published is not None:
                if (
                    int(published["trades"]) != len(trades)
                    or abs(float(published["net_bps_per_session"]) - diag["net_bps_per_session"])
                    > 1e-7
                ):
                    raise ValueError(f"report score does not reproduce: {horizon}m {name}")
            summaries[name] = diag
        comparisons = {
            "high_vs_normal": _comparison(
                cohort_trades["high_rvol_all"],
                cohort_trades["normal_rvol_control_all"],
                days,
                slots,
                cost,
                repeats,
                seed,
            ),
            "market_confirmation_vs_unfiltered": _comparison(
                cohort_trades["high_rvol_market"],
                cohort_trades["high_rvol_all"],
                days,
                slots,
                cost,
                repeats,
                seed,
            ),
        }
        result["horizons"][horizon] = {
            "cohorts": summaries,
            "comparisons": comparisons,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing report overwrite: {args.output}")
    report_path = args.report / "validation.json" if args.report.is_dir() else args.report
    if not report_path.is_file():
        parser.error(
            f"report not found: {report_path}\n"
            "Locate completed reports with: "
            "find data/dev/models -type f -name validation.json | sort\n"
            "Pass --report with the actual validation.json path or its directory."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = audit_report(
        report,
        cost=args.cost_bps,
        repeats=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for horizon, value in summary["horizons"].items():
        high = value["cohorts"]["high_rvol_all"]
        ci = high["session_bootstrap_95pct"]
        hc = value["comparisons"]["market_confirmation_vs_unfiltered"]
        vs = value["comparisons"]["high_vs_normal"]
        print(
            f"{horizon}m high RVOL: trades={high['trades']} "
            f"net/session={high['net_bps_per_session']:+.3f}bps "
            f"95% CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        )
        print(
            f"  high-normal delta={vs['paired_difference_net_bps_per_session']:+.3f}bps "
            f"shared market-confirmed trades={hc['shared_trades']} "
            f"market-only={hc['first_only_trades']} "
            f"unfiltered-only={hc['second_only_trades']}"
        )
        print(
            f"  excluding best 3: "
            f"{high['after_removing_best_trades_bps_per_session']['exclude_best_3']:+.3f}bps/session"
        )
    print(f"Audit report: {args.output}")
    print("NOTICE: development price proxies only; no new historical source or holdout accessed")


if __name__ == "__main__":
    main()

"""Read-only P5-A diagnostic audit: capital overlap, concentration, and hedges.

This script does NOT read market data, fit models, or estimate tradable P&L.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, median
from typing import Any


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric (got {type(value).__name__})")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _cost_key(by_cost: dict[str, Any], cost_bps: float) -> str:
    candidates = [key for key in by_cost if isclose(float(key), cost_bps, abs_tol=1e-9)]
    if len(candidates) != 1:
        raise ValueError(f"cost {cost_bps} not in report; available: {sorted(by_cost)}")
    return candidates[0]


def audit_report(
    report: dict[str, Any],
    *,
    horizon: int = 10,
    side: str = "positive_short",
    regime: str = "all",
    cost_bps: float = 4.0,
    remove_best: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Audit exactly the saved trade set without refitting or choosing a model."""
    if (
        horizon < 1
        or side not in {"negative_long", "positive_short"}
        or regime not in {"all", "range", "trend"}
        or not isfinite(cost_bps)
        or cost_bps < 0
        or any(n < 1 for n in remove_best)
    ):
        raise ValueError("invalid audit parameters")
    if report.get("holdout_accessed") is not False:
        raise ValueError("report does not explicitly certify holdout exclusion")
    name = f"{horizon}d_{side}_{regime}"
    results = report.get("results", {})
    if name not in results:
        raise ValueError(f"result {name!r} missing from report")
    by_cost = results[name]
    record = by_cost[_cost_key(by_cost, cost_bps)]
    entries = record.get("trades_detail")
    if not isinstance(entries, list):
        raise ValueError(f"{name} is missing trade-level detail")
    slots = int(report["configuration"]["max_per_side"])
    n_days = int(report["configuration"]["eligible_days"])
    if slots < 1 or n_days < 1:
        raise ValueError("invalid session-count or position-slot metadata")
    if int(record["trades"]) != len(entries):
        raise ValueError("trade count does not match trade detail")

    nets: list[float] = []
    gross: list[float] = []
    unhedged: list[float] = []
    per_ticker: Counter[str] = Counter()
    per_factor: Counter[str] = Counter()
    per_month: dict[str, list[float]] = defaultdict(list)
    seen_entry_dates: set[date] = set()
    windows: list[tuple[date, date]] = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("trade detail must be an object")
        net = _numeric(item.get("net_factor_hedged_bps"), "net_factor_hedged_bps")
        gr = _numeric(item.get("gross_factor_hedged_bps"), "gross_factor_hedged_bps")
        naked = _numeric(item.get("unhedged_stock_bps"), "unhedged_stock_bps")
        beta = _numeric(item.get("beta"), "beta")
        if not isclose(gr - net, cost_bps * (1 + abs(beta)), abs_tol=1e-6):
            raise ValueError("stored trade costs are inconsistent with hedge ratio")
        formation = date.fromisoformat(item["formation_day"])
        entry = date.fromisoformat(item["entry_day"])
        exit_ = date.fromisoformat(item["exit_day"])
        if not formation < entry < exit_:
            raise ValueError("trade event timestamps are not chronological")
        ticker = str(item["ticker"])
        factor = str(item["factor"])
        if not ticker or not factor:
            raise ValueError("missing ticker or factor")
        nets.append(net)
        gross.append(gr)
        unhedged.append(naked)
        per_ticker[ticker] += 1
        per_factor[factor] += 1
        per_month[formation.strftime("%Y-%m")].append(net)
        seen_entry_dates.add(entry)
        windows.append((entry, exit_))

    calculated_session_bps = sum(nets) / (slots * n_days)
    published_session_bps = _numeric(
        record["net_bps_per_entry_session"], "net_bps_per_entry_session"
    )
    if not isclose(calculated_session_bps, published_session_bps, abs_tol=1e-5):
        raise ValueError(
            "saved report's net per entry session does not match its trade detail: "
            f"{calculated_session_bps} != {published_session_bps}"
        )

    ordered = sorted(nets, reverse=True)
    stress = {
        f"remove_best_{n}": (sum(nets) - sum(ordered[:n])) / (slots * n_days)
        for n in remove_best
        if n <= len(nets)
    }
    sampled_concurrency = [
        sum(start <= day < end for start, end in windows) for day in sorted(seen_entry_dates)
    ]
    # Sampled at actual entry dates, not every exchange session; an upper bound
    # on capital demand cannot be inferred from the report alone.
    implied_hedge_gain = [hedged - naked for hedged, naked in zip(nets, unhedged, strict=True)]
    top_tickers = per_ticker.most_common(10)
    top_factors = per_factor.most_common(10)
    return {
        "result": name,
        "cost_bps": cost_bps,
        "trades": len(nets),
        "eligible_entry_sessions": n_days,
        "slots_per_entry_session": slots,
        "reconstructed_net_bps_per_entry_session": calculated_session_bps,
        "net_bps_per_trade_mean": mean(nets) if nets else 0.0,
        "net_bps_per_trade_median": median(nets) if nets else 0.0,
        "gross_bps_per_trade_mean": mean(gross) if gross else 0.0,
        "unhedged_stock_net_bps_per_trade_mean": mean(unhedged) if unhedged else 0.0,
        "hedge_incremental_net_bps_per_trade_mean": (
            mean(implied_hedge_gain) if implied_hedge_gain else 0.0
        ),
        "positive_net_trade_fraction": sum(x > 0 for x in nets) / len(nets) if nets else 0.0,
        "worst_net_trade_bps": min(nets) if nets else None,
        "best_net_trade_bps": max(nets) if nets else None,
        "best_trade_removal_net_bps_per_entry_session": stress,
        "top_three_trade_sum_bps": sum(ordered[:3]),
        "total_trade_net_bps": sum(nets),
        "distinct_tickers": len(per_ticker),
        "top_tickers_by_trade_count": top_tickers,
        "top_factors_by_trade_count": top_factors,
        "by_formation_month_net_bps_per_trade": {
            month: {"trades": len(vals), "mean_net_bps_per_trade": mean(vals)}
            for month, vals in sorted(per_month.items())
        },
        "open_positions_on_entry_dates": {
            "sampled_dates": len(sampled_concurrency),
            "maximum": max(sampled_concurrency, default=0),
            "mean": mean(sampled_concurrency) if sampled_concurrency else 0.0,
        },
        "reported_block_bootstrap_95_interval": record.get("block_bootstrap_95_interval"),
        "selection_diagnostics": report.get("candidate_diagnostics", {}),
        "critical_caveats": [
            "Per-entry-session return is NOT portfolio P&L: multi-day trades overlap.",
            "Overlapping position counts are sampled on saved entry dates only.",
            "Short-stock borrow fees, locate availability, forced buy-ins and financing are missing.",
            "Unadjusted ticker-based prices can contain corporate actions or identity changes.",
            "Missing delisted symbols/exits may censor the worst outcomes.",
            "ETF proxy factors are not verified historical industry assignments.",
            "Only exploratory development dates have been evaluated; multiple tests were inspected.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--side", choices=("positive_short", "negative_long"), default="positive_short"
    )
    parser.add_argument("--regime", choices=("all", "range", "trend"), default="all")
    parser.add_argument("--cost-bps", type=float, default=4.0)
    args = parser.parse_args()
    source = args.report / "validation.json" if args.report.is_dir() else args.report
    if not source.is_file():
        raise FileNotFoundError(
            f"P5 report not found: {source}. Locate it using: "
            "find data/dev/models -name validation.json"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = json.loads(source.read_text(encoding="utf-8"))
    audited = audit_report(
        report,
        horizon=args.horizon,
        side=args.side,
        regime=args.regime,
        cost_bps=args.cost_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Result: {audited['result']} cost={args.cost_bps:g}bps")
    print(f"Trades: {audited['trades']}; symbols: {audited['distinct_tickers']}")
    print(f"Net/entry_session: {audited['reconstructed_net_bps_per_entry_session']:+.3f}bps ")
    print(f"Net/trade: {audited['net_bps_per_trade_mean']:+.3f}bps")
    print(f"Median net/trade: {audited['net_bps_per_trade_median']:+.3f}bps")
    print("After removing best trades (net/entry_session):")
    for key, value in audited["best_trade_removal_net_bps_per_entry_session"].items():
        print(f"  {key}: {value:+.3f}bps")
    conc = audited["open_positions_on_entry_dates"]
    print(f"Maximum overlapping positions sampled at entry dates: {conc['maximum']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

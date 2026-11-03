"""Read-only price-discontinuity diagnostic for P5-A positive-residual shorts.

This analysis uses saved trades and cannot establish whether a corporate action
occurred. Real verification requires split-adjusted prices and corporate actions.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from math import exp, isclose, isfinite
from pathlib import Path
from typing import Any

BPS = 10_000.0
# Unadjusted post-split / pre-split price ratios. These are diagnostic examples,
# not a complete or authoritative list of corporate actions.
SPLIT_PRICE_RATIOS = {
    "6-for-5": 5 / 6,
    "5-for-4": 4 / 5,
    "4-for-3": 3 / 4,
    "7-for-5": 5 / 7,
    "3-for-2": 2 / 3,
}


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _match_split_ratio(price_ratio: float, tolerance: float) -> str | None:
    if not 0.0 < price_ratio < 1.0:
        return None
    name, distance = min(
        ((name, abs(price_ratio / ratio - 1.0)) for name, ratio in SPLIT_PRICE_RATIOS.items()),
        key=lambda item: item[1],
    )
    return name if distance <= tolerance else None


def audit_p5_split_risk(
    report: dict[str, Any],
    *,
    horizon: int = 10,
    cost_bps: float = 4.0,
    top: int = 15,
    large_drop_pct: float = 15.0,
    split_ratio_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Reconstruct raw-price diagnostics using only saved P5-A trade detail.

    Treat excluded trades as abstentions using the ORIGINAL session denominator.
    A split-like price ratio is a warning, not a verified split classification.
    """
    if report.get("holdout_accessed") is not False:
        raise ValueError("holdout protection must be explicitly certified")
    if (
        horizon < 1
        or top < 1
        or cost_bps < 0
        or not isfinite(cost_bps)
        or not 0 < large_drop_pct < 100
        or not 0 < split_ratio_tolerance < 0.5
    ):
        raise ValueError("invalid diagnostic settings")
    config = report.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("missing configuration")
    slots = int(config["max_per_side"])
    eligible_days = int(config["eligible_days"])
    if slots < 1 or eligible_days < 1:
        raise ValueError("invalid eligible session / slot count")
    label = f"{horizon}d_positive_short_all"
    results = report.get("results")
    if not isinstance(results, dict) or label not in results:
        raise ValueError(f"missing result {label}")
    by_cost = results[label]
    if not isinstance(by_cost, dict):
        raise ValueError("invalid cost results")
    keys = [
        key
        for key in by_cost
        if isclose(
            float(key),
            cost_bps,
            abs_tol=1e-10,
        )
    ]
    if len(keys) != 1:
        raise ValueError(f"cost scenario {cost_bps} not in report")
    record = by_cost[keys[0]]
    trades = record.get("trades_detail")
    if not isinstance(trades, list) or len(trades) != int(record.get("trades", -1)):
        raise ValueError("invalid trade detail count")
    detailed: list[dict[str, Any]] = []
    for trade in trades:
        if not isinstance(trade, dict):
            raise ValueError("trade must be a mapping")
        net = _number(trade.get("net_factor_hedged_bps"), "net")
        gross = _number(trade.get("gross_factor_hedged_bps"), "gross")
        naked = _number(trade.get("unhedged_stock_bps"), "unhedged stock")
        beta = _number(trade.get("beta"), "beta")
        if not isclose(gross - net, cost_bps * (1.0 + abs(beta)), abs_tol=1e-5):
            raise ValueError("saved transaction costs are inconsistent")
        entry = date.fromisoformat(str(trade["entry_day"]))
        exit_ = date.fromisoformat(str(trade["exit_day"]))
        formation = date.fromisoformat(str(trade["formation_day"]))
        if not formation < entry < exit_:
            raise ValueError("nonchronological trade")
        # P5-A: unhedged_stock_bps = -stock_log_return_bps - cost_bps
        # for positive_short. Reverse that equation, rather than inventing
        # observed prices or corporate-action information.
        stock_log_return_bps = -naked - cost_bps
        price_ratio = exp(stock_log_return_bps / BPS)
        price_change_pct = 100.0 * (price_ratio - 1.0)
        split_like = _match_split_ratio(price_ratio, split_ratio_tolerance)
        detailed.append(
            {
                "ticker": str(trade["ticker"]),
                "formation_day": formation.isoformat(),
                "entry_day": entry.isoformat(),
                "exit_day": exit_.isoformat(),
                "factor": str(trade["factor"]),
                "beta": beta,
                "net_bps": net,
                "stock_short_gross_bps": naked + cost_bps,
                "hedge_incremental_net_bps": net - naked,
                "raw_open_to_open_price_change_pct": price_change_pct,
                "implied_raw_price_ratio": price_ratio,
                "split_like_ratio": split_like,
                "large_raw_price_decline": price_change_pct <= -large_drop_pct,
            }
        )
    total = sum(float(item["net_bps"]) for item in detailed)
    denominator = slots * eligible_days
    original = _number(record.get("net_bps_per_entry_session"), "reported net per session")
    if not isclose(total / denominator, original, abs_tol=1e-4):
        raise ValueError("reported net does not reconcile with trade detail")
    ranked = sorted(detailed, key=lambda row: float(row["net_bps"]), reverse=True)
    drop_large = [row for row in detailed if row["large_raw_price_decline"]]
    split_like = [row for row in detailed if row["split_like_ratio"] is not None]
    suspected = {id(row) for row in split_like}
    large = {id(row) for row in drop_large}
    worst_case_exclusions = [row for row in detailed if id(row) not in large]
    return {
        "status": "EXPLORATORY_DIAGNOSTIC_NOT_CORPORATE_ACTION_VERIFIED",
        "result": label,
        "source_cost_per_leg_bps": cost_bps,
        "total_trades": len(detailed),
        "original_entry_sessions": eligible_days,
        "original_slots_per_entry_session": slots,
        "original_net_bps_per_entry_session": total / denominator,
        "large_raw_price_drop_threshold_pct": large_drop_pct,
        "large_raw_decline_trades": len(drop_large),
        "large_raw_decline_net_bps_contribution": sum(float(x["net_bps"]) for x in drop_large)
        / denominator,
        "after_excluding_large_raw_declines_net_bps_per_entry_session": (
            sum(float(x["net_bps"]) for x in worst_case_exclusions) / denominator
        ),
        "split_like_ratio_trades": len(split_like),
        "split_like_ratio_net_bps_contribution": sum(float(x["net_bps"]) for x in split_like)
        / denominator,
        "after_excluding_split_like_trades_net_bps_per_entry_session": (
            sum(float(x["net_bps"]) for x in detailed if id(x) not in suspected) / denominator
        ),
        "top_trade_count": min(top, len(ranked)),
        "top_trades": ranked[:top],
        "selection_diagnostics": report.get("candidate_diagnostics", {}),
        "warnings": [
            "Split-like ratios are NOT confirmed stock splits; market price changes can resemble them.",
            "Exclusion stress tests preserve the original entry-session denominator.",
            "Only splits and adjusted historical security identities can resolve these flags.",
            "Short borrowing, hedge execution, stock borrow availability, and overlapping capital are not validated.",
            "This analysis does not load the protected final holdout or reconstruct daily equity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--large-drop-pct", type=float, default=15.0)
    args = parser.parse_args()
    path = args.report / "validation.json" if args.report.is_dir() else args.report
    if not path.is_file():
        raise FileNotFoundError(f"P5 report not found: {path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    report = json.loads(path.read_text(encoding="utf-8"))
    audited = audit_p5_split_risk(
        report,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        top=args.top,
        large_drop_pct=args.large_drop_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Audited {audited['total_trades']} trades; "
        f"original={audited['original_net_bps_per_entry_session']:+.3f} bps/entry-session"
    )
    print(
        f"Large raw-price-decline flags: {audited['large_raw_decline_trades']}; "
        f"without flags={audited['after_excluding_large_raw_declines_net_bps_per_entry_session']:+.3f}bps"
    )
    print(
        f"Split-like ratio flags (unverified): {audited['split_like_ratio_trades']}; "
        f"without flags={audited['after_excluding_split_like_trades_net_bps_per_entry_session']:+.3f}bps"
    )
    print("Top saved trades (net bps, raw stock percent change, suspected ratio):")
    for row in audited["top_trades"]:
        print(
            f"  {row['ticker']} {row['entry_day']}..{row['exit_day']}: "
            f"net={row['net_bps']:+.1f}bps "
            f"stock_price={row['raw_open_to_open_price_change_pct']:+.1f}% "
            f"ratio={row['split_like_ratio'] or '-'}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

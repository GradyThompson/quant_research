"""Read-only P5-A corporate-action and instrument-type reconciliation.

The original P5-A ranking was made without a true common-stock universe;
post-hoc exclusions here are diagnostics, NEVER a valid corrected backtest.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from math import exp, isclose, isfinite
from typing import Any

BPS = 10_000.0


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _share_multiplier(
    splits: list[dict[str, Any]], *, entry: date, exit_: date
) -> tuple[float, list[dict[str, Any]]]:
    """Multiplicative share-count change between two regular-session opens."""
    multiplier = 1.0
    events: list[dict[str, Any]] = []
    for row in splits:
        execution = date.fromisoformat(str(row["execution_date"]))
        if not (entry < execution <= exit_):
            continue
        before = _number(row.get("split_from"), "split_from")
        after = _number(row.get("split_to"), "split_to")
        if before <= 0 or after <= 0:
            raise ValueError("invalid split share-count ratio")
        multiplier *= after / before
        events.append(
            {
                "execution_date": execution.isoformat(),
                "split_from": before,
                "split_to": after,
                "adjustment_type": str(row.get("adjustment_type", "unknown")),
            }
        )
    return multiplier, events


def reconcile_p5_report(
    report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    horizon: int = 10,
    cost_bps: float = 4.0,
) -> dict[str, Any]:
    """Check P5 positive-short returns against dated reference and splits.

    Evidence is immutable vendor REST results, not guessed ticker categories.
    No missing evidence is treated as confirming a valid common stock or zero
    splits. Arithmetic returns measure P&L per $1 of SHORT stock notional,
    BEFORE borrow fees, financing, dividends or portfolio gross leverage.
    """
    if report.get("holdout_accessed") is not False:
        raise ValueError("holdout exclusion is not confirmed")
    if horizon < 1 or cost_bps < 0 or not isfinite(cost_bps):
        raise ValueError("invalid audit parameters")
    config = report.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("missing configuration")
    slots, sessions = int(config["max_per_side"]), int(config["eligible_days"])
    if slots < 1 or sessions < 1:
        raise ValueError("invalid slots/session count")
    key = f"{horizon}d_positive_short_all"
    bucket = report["results"][key]
    matching = [str(c) for c in bucket if isclose(float(c), cost_bps, abs_tol=1e-9)]
    if len(matching) != 1:
        raise ValueError("cost scenario is missing or ambiguous")
    score = bucket[matching[0]]
    trades = score["trades_detail"]
    if len(trades) != int(score["trades"]):
        raise ValueError("trade count mismatch")
    references = evidence.get("references")
    split_data = evidence.get("splits")
    if not isinstance(references, dict) or not isinstance(split_data, dict):
        raise ValueError("missing dated reference or split evidence")

    reasons: Counter[str] = Counter()
    outputs: list[dict[str, Any]] = []
    original_sum = 0.0
    for trade in trades:
        ticker = str(trade["ticker"])
        factor = str(trade["factor"])
        formation = date.fromisoformat(str(trade["formation_day"]))
        entry = date.fromisoformat(str(trade["entry_day"]))
        exit_ = date.fromisoformat(str(trade["exit_day"]))
        if not (formation < entry < exit_):
            raise ValueError("nonchronological trade")
        beta = _number(trade["beta"], "beta")
        if not 0.0 <= beta <= 3.0:
            raise ValueError("unexpected factor beta")
        gross_log = _number(trade["gross_factor_hedged_bps"], "gross")
        net_log = _number(trade["net_factor_hedged_bps"], "net")
        naked = _number(trade["unhedged_stock_bps"], "unhedged")
        if not isclose(gross_log - net_log, cost_bps * (1.0 + beta), abs_tol=1e-5):
            raise ValueError("inconsistent original cost")
        original_sum += net_log
        # P5 positive_short defines naked as -stock_log*BPS - cost.
        stock_log = (-naked - cost_bps) / BPS
        short_stock_log_gross = -stock_log * BPS
        hedge_log_gross = gross_log - short_stock_log_gross
        factor_log = (hedge_log_gross / beta / BPS) if beta > 0 else 0.0
        if beta == 0 and not isclose(hedge_log_gross, 0.0, abs_tol=1e-5):
            raise ValueError("nonzero hedge return with zero beta")
        ref_key = f"{formation.isoformat()}|{ticker}"
        ref_payload = references.get(ref_key)
        if not isinstance(ref_payload, dict):
            reasons["missing_dated_reference"] += 1
            continue
        rows = ref_payload.get("results")
        if not isinstance(rows, list):
            reasons["invalid_dated_reference"] += 1
            continue
        matches = [row for row in rows if isinstance(row, dict) and row.get("ticker") == ticker]
        if len(matches) != 1:
            reasons["ambiguous_or_missing_identity"] += 1
            continue
        reference = matches[0]
        if reference.get("type") != "CS":
            reasons[f"non_common_stock:{reference.get('type', 'missing')}"] += 1
            continue
        if reference.get("locale") != "us" or not reference.get("share_class_figi"):
            reasons["incomplete_common_stock_identity"] += 1
            continue
        stock_payload = split_data.get(ticker)
        factor_payload = split_data.get(factor)
        if not isinstance(stock_payload, dict) or not isinstance(factor_payload, dict):
            reasons["missing_split_evidence"] += 1
            continue
        stock_splits = stock_payload.get("results")
        factor_splits = factor_payload.get("results")
        if not isinstance(stock_splits, list) or not isinstance(factor_splits, list):
            reasons["invalid_split_evidence"] += 1
            continue
        # Verify evidence covers entire holding interval even when results=[];
        # an empty *bounded* vendor response is distinct from missing data.
        start = date.fromisoformat(str(evidence.get("split_start")))
        end = date.fromisoformat(str(evidence.get("split_end")))
        if entry < start or exit_ > end:
            reasons["split_window_not_covered"] += 1
            continue
        stock_multiplier, stock_actions = _share_multiplier(stock_splits, entry=entry, exit_=exit_)
        factor_multiplier, factor_actions = _share_multiplier(
            factor_splits, entry=entry, exit_=exit_
        )
        # Correct for *share count* changes, not guessed price ratios.
        stock_simple = exp(stock_log) * stock_multiplier - 1.0
        factor_simple = exp(factor_log) * factor_multiplier - 1.0
        gross_arithmetic = BPS * (-stock_simple + beta * factor_simple)
        net_arithmetic = gross_arithmetic - cost_bps * (1.0 + beta)
        outputs.append(
            {
                "ticker": ticker,
                "factor": factor,
                "formation_day": formation.isoformat(),
                "entry_day": entry.isoformat(),
                "exit_day": exit_.isoformat(),
                "figi": reference["share_class_figi"],
                "beta": beta,
                "original_log_proxy_net_bps": net_log,
                "stock_short_arithmetic_gross_bps": -stock_simple * BPS,
                "hedge_arithmetic_gross_bps": beta * factor_simple * BPS,
                "reconciled_arithmetic_net_bps_per_stock_notional": net_arithmetic,
                "stock_actions": stock_actions,
                "factor_actions": factor_actions,
            }
        )
        reasons["verified_common_stock_trade"] += 1
    original = _number(score["net_bps_per_entry_session"], "original score")
    if not isclose(original_sum / (slots * sessions), original, abs_tol=1e-4):
        raise ValueError("original reported score does not reconcile")
    reconciled = sum(row["reconciled_arithmetic_net_bps_per_stock_notional"] for row in outputs)
    ranked = sorted(
        outputs,
        key=lambda row: row["reconciled_arithmetic_net_bps_per_stock_notional"],
        reverse=True,
    )
    return {
        "status": "POST_HOC_RECONCILIATION_ONLY_NOT_A_REBACKTEST",
        "original_trades": len(trades),
        "verified_common_stock_trades_after_original_selection": len(outputs),
        "excluded_or_unverified_counts": dict(sorted(reasons.items())),
        "original_log_proxy_bps_per_entry_session": original,
        "verified_subset_arithmetic_bps_per_original_entry_session": reconciled
        / (slots * sessions),
        "verified_subset_arithmetic_bps_per_trade": reconciled / len(outputs) if outputs else None,
        "verified_subset_trades_crossing_splits": sum(
            bool(x["stock_actions"] or x["factor_actions"]) for x in outputs
        ),
        "verified_subset_gross_stock_bps": sum(
            x["stock_short_arithmetic_gross_bps"] for x in outputs
        ),
        "verified_subset_gross_hedge_bps": sum(x["hedge_arithmetic_gross_bps"] for x in outputs),
        "top_reconciled_trades": ranked[:20],
        "warnings": [
            "Classifications applied AFTER original candidate selection; cannot establish clean strategy alpha.",
            "Stock and hedge use fixed-notional arithmetic approximation; not a daily portfolio ledger.",
            "Dividend obligations, stock borrow fees, hard-to-borrow restrictions, delistings and margin omitted.",
            "Only split records dated strictly after entry through exit are applied; formation shocks remain unadjusted.",
            "A clean rerun must screen common-stock identity and adjust prices BEFORE signal selection.",
        ],
    }

"""P4.3: fixed opening-range continuation hypotheses, pinned OHLCV, no broker.

Research-only price proxies; validates only chronological development sessions.
Does not tune candidate rules on outcomes or access the original final holdout.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.views.pinned_minute_sessions import (
    load_sessions,
    read_pinned_receipt,
    select_pinned_factor_entries,
)
from quant_research.model.first_passage_validation import select_receipt_window
from quant_research.model.opening_range import (
    VARIANTS,
    BreakoutTrade,
    build_opening_range_trades,
    score_breakout_trades,
)
from quant_research.paths import resolve_project_path
from quant_research.validation.staged_sessions import staged_session_folds


def _describe_trade(trade: BreakoutTrade) -> dict[str, object]:
    return {
        "day": trade.day.isoformat(),
        "variant": trade.variant,
        "side": "long" if trade.side > 0 else "short",
        "decision_at": trade.decision_at.isoformat(),
        "entry_at": trade.entry_at.isoformat(),
        "exit_available_at": trade.exit_available_at.isoformat(),
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "gross_bps": trade.gross_bps,
    }


def run_opening_range_research(
    *,
    ticker: str,
    factor_ticker: str,
    asset_receipt_path: Path,
    factor_receipt_path: Path,
    store: ParquetDatasetStore,
    output_dir: Path,
    evaluation_start: date = date(2025, 1, 13),
    horizons: tuple[int, ...] = (30, 60),
    training_sessions: int = 38,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
    warmup_sessions: int = 19,
    alpha_fit_sessions: int = 10,
    fold_embargo_sessions: int = 1,
    fold_validation_sessions: int = 5,
    fold_step_sessions: int = 5,
    roundtrip_costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Run preregistered untuned candidates on staged, forward validation days."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_dir}")
    if ticker == factor_ticker:
        raise ValueError("factor and asset ticker must differ")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(h not in (30, 60) for h in horizons)
    ):
        raise ValueError("pilot prespecifies 30m and/or 60m horizons")
    if not roundtrip_costs_bps or len(set(roundtrip_costs_bps)) != len(roundtrip_costs_bps):
        raise ValueError("invalid unique cost scenarios")
    asset_receipt, asset_hash = read_pinned_receipt(asset_receipt_path)
    factor_receipt, factor_hash = read_pinned_receipt(factor_receipt_path)
    # Immutable original M2.5/P3 reserved embargo + final holdout. Do not
    # change this anchor even if a caller requests different research dates.
    _, _, original_protected = select_receipt_window(
        asset_receipt,
        ticker=ticker,
        evaluation_start=date(2025, 1, 13),
        training_sessions=38,
        validation_sessions=10,
        holdout_sessions=10,
        embargo_sessions=1,
    )
    asset_entries, _, withheld = select_receipt_window(
        asset_receipt,
        ticker=ticker,
        evaluation_start=evaluation_start,
        training_sessions=training_sessions,
        validation_sessions=validation_sessions,
        holdout_sessions=holdout_sessions,
        embargo_sessions=embargo_sessions,
    )
    days = tuple(date.fromisoformat(str(entry["session"])) for entry in asset_entries)
    if set(day.isoformat() for day in days) & set(original_protected):
        raise ValueError("requested window would read the original M2.5/P3 embargo or holdout")
    folds = staged_session_folds(
        days,
        baseline_fit_sessions=warmup_sessions,
        alpha_fit_sessions=alpha_fit_sessions,
        embargo_sessions=fold_embargo_sessions,
        validation_sessions=fold_validation_sessions,
        step_sessions=fold_step_sessions,
    )
    if not folds:
        raise ValueError("insufficient permitted sessions for a forward fold")
    factor_entries = select_pinned_factor_entries(
        factor_receipt, ticker=factor_ticker, allowed_days=set(days)
    )
    asset = load_sessions(store, asset_entries)
    factor = load_sessions(store, factor_entries)
    if tuple(s.day for s in asset) != days or tuple(s.day for s in factor) != days:
        raise ValueError("pinned sessions differ from authorized dates")

    experiments = []
    for horizon in horizons:
        diagnostics: dict[str, int] = {}
        candidates = build_opening_range_trades(
            asset_sessions=asset,
            factor_sessions=factor,
            horizon_minutes=horizon,
            diagnostics=diagnostics,
        )
        fold_reports: list[dict[str, object]] = []
        pooled_days: list[date] = []
        pooled_trades: dict[str, list[BreakoutTrade]] = {variant: [] for variant in VARIANTS}
        for fold in folds:
            validation = tuple(fold.validation_days)
            pooled_days.extend(validation)
            description: dict[str, object] = {
                "fold": fold.number,
                "baseline_fit_days": [day.isoformat() for day in fold.baseline_fit_days],
                "alpha_fit_days": [day.isoformat() for day in fold.alpha_fit_days],
                "embargo_days": [day.isoformat() for day in fold.embargo_days],
                "validation_days": [day.isoformat() for day in validation],
            }
            cost_sensitivity: dict[str, list[dict[str, object]]] = {}
            for variant in VARIANTS:
                these = [trade for trade in candidates[variant] if trade.day in validation]
                pooled_trades[variant].extend(these)
                cost_sensitivity[variant] = [
                    score_breakout_trades(
                        these,
                        eligible_days=validation,
                        roundtrip_cost_bps=cost,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                    for cost in roundtrip_costs_bps
                ]
            description["cost_sensitivity"] = cost_sensitivity
            fold_reports.append(description)
        experiments.append(
            {
                "horizon_minutes": horizon,
                "diagnostics": diagnostics,
                "folds": fold_reports,
                "pooled_scored_days": len(pooled_days),
                "pooled_cost_sensitivity": {
                    variant: [
                        score_breakout_trades(
                            pooled_trades[variant],
                            eligible_days=pooled_days,
                            roundtrip_cost_bps=cost,
                            bootstrap_replicates=bootstrap_replicates,
                        )
                        for cost in roundtrip_costs_bps
                    ]
                    for variant in VARIANTS
                },
                "pooled_trades": {
                    variant: [_describe_trade(t) for t in pooled_trades[variant]]
                    for variant in VARIANTS
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "model_family": "fixed-opening-range-continuation-p4.3",
        "research_only": True,
        "holdout_accessed": False,
        "execution_model": "historical-minute-price-proxy; NOT executable quoted fills",
        "asset_ticker": ticker,
        "factor_ticker": factor_ticker,
        "asset_receipt_sha256": asset_hash,
        "factor_receipt_sha256": factor_hash,
        "source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in asset_entries
        ],
        "factor_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in factor_entries
        ],
        "withheld_sessions_not_loaded": withheld,
        "original_protected_sessions_not_loaded": original_protected,
        "configuration": {
            "opening_minutes": 15,
            "first_scan_minutes": 120,
            "breakout_buffer_bps": 5.0,
            "prior_open_volume_sessions": 10,
            "minimum_open_volume_ratio": 1.5,
            "horizons": list(horizons),
            "roundtrip_costs_bps": list(roundtrip_costs_bps),
            "fold_validation_sessions": fold_validation_sessions,
            "fold_step_sessions": fold_step_sessions,
            "thresholds_tuned_on_validation": False,
        },
        "variants": list(VARIANTS),
        "experiments": experiments,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--factor-ticker", default="QQQ")
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_training_receipt_20250113.json")
    )
    parser.add_argument("--factor-receipt", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--roundtrip-cost-bps", type=float, action="append")
    args = parser.parse_args()
    report = run_opening_range_research(
        ticker=args.ticker.upper(),
        factor_ticker=args.factor_ticker.upper(),
        asset_receipt_path=resolve_project_path(args.receipt),
        factor_receipt_path=resolve_project_path(args.factor_receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        output_dir=resolve_project_path(args.output_dir),
        horizons=tuple(args.horizon or [30, 60]),
        roundtrip_costs_bps=tuple(args.roundtrip_cost_bps or [4.0, 8.0, 12.0]),
    )
    for experiment in report["experiments"]:
        print(
            f"{args.ticker} {experiment['horizon_minutes']}m: "
            f"{experiment['pooled_scored_days']} OOF sessions"
        )
        for variant, scenarios in experiment["pooled_cost_sensitivity"].items():
            for scenario in scenarios:
                print(
                    f"  {variant} {scenario['roundtrip_cost_bps']:g}bps: "
                    f"trades={scenario['trades']} "
                    f"net_per_session={scenario['net_bps_per_session']:+.3f}bps "
                    "[PROXY ONLY]"
                )
    print(f"Report: {args.output_dir}/validation.json")
    print("NOTICE: Development only; reserved final holdout untouched; no broker access")


if __name__ == "__main__":
    main()

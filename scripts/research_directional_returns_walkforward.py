"""P4: development-only, pinned-session directional-return walk-forward study.

Uses next-eligible-open strictly after decision through horizon close (price proxy),
and compares a training intercept to factor, residual, and combined ridge
signals. The original final holdout is never loaded. No trading is performed.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any

from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.views.pinned_minute_sessions import (
    load_sessions,
    read_pinned_receipt,
    select_pinned_factor_entries,
)
from quant_research.model.directional_return_policy import score_return_policy
from quant_research.model.directional_return_scoring import (
    ReturnPrediction,
    score_return_predictions,
)
from quant_research.model.directional_returns import (
    MODEL_KINDS,
    build_return_rows,
    fit_directional_return_model,
)
from quant_research.model.first_passage_validation import select_receipt_window
from quant_research.paths import resolve_project_path
from quant_research.validation.staged_sessions import staged_session_folds


def run_directional_return_walkforward(
    *,
    ticker: str,
    factor_ticker: str,
    asset_receipt_path: Path,
    factor_receipt_path: Path,
    store: ParquetDatasetStore,
    output_dir: Path,
    evaluation_start: date,
    horizons: list[int],
    training_sessions: int = 38,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
    warmup_sessions: int = 19,
    alpha_fit_sessions: int = 10,
    fold_embargo_sessions: int = 1,
    fold_validation_sessions: int = 5,
    fold_step_sessions: int = 5,
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
    min_training_rows: int = 30,
    regularization: float = 10.0,
    bootstrap_replicates: int = 2000,
    roundtrip_costs_bps: tuple[float, ...] = (4.0, 8.0, 12.0),
    excess_edge_bps: float = 2.0,
) -> dict[str, Any]:
    """Fit each candidate only on mature labels in each fold's alpha days.

    The earliest `warmup_sessions` are for feature history, not alpha fitting.
    The original reservation of holdout days is inherited from M2.5/P3.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite research report: {output_dir}")
    if ticker == factor_ticker:
        raise ValueError("asset and factor must be distinct")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(type(h) is not int or h < 1 for h in horizons)
    ):
        raise ValueError("horizons must be unique positive integers")
    if type(min_training_rows) is not int or min_training_rows < 2:
        raise ValueError("min_training_rows must be >= 2")
    if not roundtrip_costs_bps or len(set(roundtrip_costs_bps)) != len(roundtrip_costs_bps):
        raise ValueError("roundtrip cost scenarios must be nonempty and unique")
    if any(not isfinite(cost) or cost < 0 for cost in roundtrip_costs_bps):
        raise ValueError("roundtrip costs must be nonnegative and finite")
    if not isfinite(excess_edge_bps) or excess_edge_bps < 0:
        raise ValueError("excess edge must be nonnegative and finite")

    asset_receipt, asset_hash = read_pinned_receipt(asset_receipt_path)
    factor_receipt, factor_hash = read_pinned_receipt(factor_receipt_path)
    # Preserve the ORIGINAL M2.5/P3 embargo+holdout boundary even if a
    # caller changes the evaluation anchor or window lengths. An anchor
    # shift must never silently convert legacy protected days to training.
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
        raise ValueError("pinned sessions do not match receipt days")
    asset_by_day = {s.day: s for s in asset}

    experiments: list[dict[str, object]] = []
    for horizon in horizons:
        # Pure point-in-time features; labels are used only after explicit
        # chronological partition and maturity checks below.
        row_diagnostics: dict[str, int] = {}
        rows = build_return_rows(
            asset_sessions=asset,
            factor_sessions=factor,
            horizon_minutes=horizon,
            return_minutes=return_minutes,
            beta_lookback=beta_lookback,
            min_beta_observations=min_beta_observations,
            diagnostics=row_diagnostics,
        )
        pooled: list[ReturnPrediction] = []
        descriptions: list[dict[str, object]] = []
        for fold in folds:
            alpha_days = set(fold.alpha_fit_days)
            validation_days = set(fold.validation_days)
            train_until = asset_by_day[fold.alpha_fit_days[-1]].bars[-1].available_at
            train = [
                row
                for row in rows
                if row.day in alpha_days and row.label_available_at <= train_until
            ]
            validation = [row for row in rows if row.day in validation_days]
            description: dict[str, object] = {
                "fold": fold.number,
                "warmup_days": [d.isoformat() for d in fold.baseline_fit_days],
                "alpha_fit_days": [d.isoformat() for d in fold.alpha_fit_days],
                "embargo_days": [d.isoformat() for d in fold.embargo_days],
                "validation_days": [d.isoformat() for d in fold.validation_days],
                "train_until": train_until.isoformat(),
                "training_rows": len(train),
                "validation_rows": len(validation),
            }
            if len(train) < min_training_rows or not validation:
                description["status"] = "insufficient_rows"
                descriptions.append(description)
                continue
            fitted = {
                kind: fit_directional_return_model(
                    train,
                    kind=kind,
                    train_until=train_until,
                    min_rows=min_training_rows,
                    regularization=regularization,
                )
                for kind in MODEL_KINDS
            }
            predictions = [
                ReturnPrediction(
                    day=row.day,
                    decision_at=row.decision_at,
                    realized=row.realized_log_return,
                    predictions={kind: model.predict(row) for kind, model in fitted.items()},
                )
                for row in validation
            ]
            pooled.extend(predictions)
            description.update(
                {
                    "status": "scored",
                    "coefficients": {
                        kind: list(model.coefficients) for kind, model in fitted.items()
                    },
                    "fold_scores": score_return_predictions(
                        predictions, bootstrap_replicates=bootstrap_replicates
                    ),
                    "fold_cost_aware_proxy": [
                        score_return_policy(
                            predictions,
                            roundtrip_cost_bps=cost,
                            excess_edge_bps=excess_edge_bps,
                            bootstrap_replicates=bootstrap_replicates,
                        )
                        for cost in roundtrip_costs_bps
                    ],
                }
            )
            descriptions.append(description)
        experiments.append(
            {
                "horizon_minutes": horizon,
                "eligible_rows": len(rows),
                "row_diagnostics": row_diagnostics,
                "folds": descriptions,
                "pooled_cost_aware_proxy": (
                    [
                        score_return_policy(
                            pooled,
                            roundtrip_cost_bps=cost,
                            excess_edge_bps=excess_edge_bps,
                            bootstrap_replicates=bootstrap_replicates,
                        )
                        for cost in roundtrip_costs_bps
                    ]
                    if pooled
                    else None
                ),
                "pooled_scores": (
                    score_return_predictions(pooled, bootstrap_replicates=bootstrap_replicates)
                    if pooled
                    else None
                ),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "model_family": "directional-log-return-ridge-v1",
        "research_only": True,
        "holdout_accessed": False,
        "ticker": ticker,
        "factor_ticker": factor_ticker,
        "asset_receipt_sha256": asset_hash,
        "factor_receipt_sha256": factor_hash,
        "asset_source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in asset_entries
        ],
        "factor_source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in factor_entries
        ],
        "withheld_sessions_not_loaded": withheld,
        "original_protected_sessions_not_loaded": original_protected,
        "label_definition": ("log(horizon_end_close / first_bar_open_strictly_after_decision)"),
        "label_is_executable_return": False,
        "evaluation_start": evaluation_start.isoformat(),
        "fold_count": len(folds),
        "configuration": {
            "warmup_sessions": warmup_sessions,
            "alpha_fit_sessions": alpha_fit_sessions,
            "fold_embargo_sessions": fold_embargo_sessions,
            "fold_validation_sessions": fold_validation_sessions,
            "fold_step_sessions": fold_step_sessions,
            "return_minutes": return_minutes,
            "beta_lookback": beta_lookback,
            "min_beta_observations": min_beta_observations,
            "min_training_rows": min_training_rows,
            "regularization": regularization,
            "roundtrip_costs_bps": list(roundtrip_costs_bps),
            "excess_edge_bps": excess_edge_bps,
            "policy_thresholds_selected_on_validation": False,
        },
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
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append")
    parser.add_argument("--training-sessions", type=int, default=38)
    parser.add_argument("--validation-sessions", type=int, default=10)
    parser.add_argument("--holdout-sessions", type=int, default=10)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--warmup-sessions", type=int, default=19)
    parser.add_argument("--alpha-fit-sessions", type=int, default=10)
    parser.add_argument("--fold-validation-sessions", type=int, default=5)
    parser.add_argument("--fold-step-sessions", type=int, default=5)
    parser.add_argument("--return-minutes", type=int, default=5)
    parser.add_argument("--beta-lookback", type=int, default=120)
    parser.add_argument("--min-beta-observations", type=int, default=60)
    parser.add_argument("--regularization", type=float, default=10.0)
    parser.add_argument("--roundtrip-cost-bps", type=float, action="append")
    parser.add_argument("--excess-edge-bps", type=float, default=2.0)
    args = parser.parse_args()
    report = run_directional_return_walkforward(
        ticker=args.ticker.upper(),
        factor_ticker=args.factor_ticker.upper(),
        asset_receipt_path=resolve_project_path(args.receipt),
        factor_receipt_path=resolve_project_path(args.factor_receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        output_dir=resolve_project_path(args.output_dir),
        evaluation_start=args.evaluation_start,
        horizons=args.horizon or [5, 10, 15],
        training_sessions=args.training_sessions,
        validation_sessions=args.validation_sessions,
        holdout_sessions=args.holdout_sessions,
        embargo_sessions=args.embargo_sessions,
        warmup_sessions=args.warmup_sessions,
        alpha_fit_sessions=args.alpha_fit_sessions,
        fold_validation_sessions=args.fold_validation_sessions,
        fold_step_sessions=args.fold_step_sessions,
        return_minutes=args.return_minutes,
        beta_lookback=args.beta_lookback,
        min_beta_observations=args.min_beta_observations,
        regularization=args.regularization,
        roundtrip_costs_bps=tuple(args.roundtrip_cost_bps or [4.0, 8.0, 12.0]),
        excess_edge_bps=args.excess_edge_bps,
    )
    for item in report["experiments"]:
        score = item["pooled_scores"]
        if score is None:
            summary = item["row_diagnostics"]
            fold_counts = ", ".join(
                f"{f['fold']}:{f['status']}(train={f['training_rows']},val={f['validation_rows']})"
                for f in item["folds"]
            )
            print(
                f"{report['ticker']} {item['horizon_minutes']}m: no scored folds; "
                f"eligible={item['eligible_rows']}; "
                f"row_diagnostics={summary}; folds=[{fold_counts}]"
            )
            continue
        deltas = score["intercept_minus_candidate_mse_bps2"]
        print(
            f"{report['ticker']} {item['horizon_minutes']}m: "
            f"scored={score['scored']} sessions={score['scored_sessions']} "
            + " ".join(f"{k}={deltas[k]:+.3f}bps2" for k in ("factor", "residual", "combined"))
        )
        for scenario in item["pooled_cost_aware_proxy"]:
            costs = scenario["roundtrip_cost_bps"]
            for kind in ("intercept", "factor", "residual", "combined"):
                entry = scenario["models"][kind]
                print(
                    f"  assumed_cost={costs:g}bps {kind}: "
                    f"trades={entry['trades']} "
                    f"net_per_opportunity={entry['net_bps_per_opportunity']:+.3f}bps "
                    f"[PROXY ONLY]"
                )
    print(f"Report: {args.output_dir / 'validation.json'}")
    print("NOTICE: Development walk-forward only; reserved final holdout untouched")


if __name__ == "__main__":
    main()

"""Chronological, development-only M2.5/M3 staged-session walk-forward study.

Only pinned asset/factor sessions approved by the original reserved-holdout
split are read. No broker access, execution or model selection is performed.
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
from quant_research.model.conditional_direction import fit_conditional_direction
from quant_research.model.directional_research import (
    build_directional_rows,
    predict_paired_directional_rows,
)
from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import (
    Session,
    evaluate_sessions,
    select_receipt_window,
)
from quant_research.model.paired_probability_scoring import (
    PairedProbabilityObservation,
    score_paired_probabilities,
)
from quant_research.paths import resolve_project_path
from quant_research.validation.staged_sessions import staged_session_folds


def _forecasts(
    sessions: list[Session],
    *,
    scenario: BarrierScenario,
    evaluation_days: set[date],
    seasonal_fit_days: set[date],
    lookback_returns: int,
    min_returns: int,
):
    predictions, _ = evaluate_sessions(
        sessions=sessions,
        evaluation_days=evaluation_days,
        seasonality_training_days=seasonal_fit_days,
        scenarios=[scenario],
        drift_modes=("price_zero",),
        volatility_methods=("seasonal_close",),
        per_scenario_stride=True,
        lookback_returns=lookback_returns,
        min_returns=min_returns,
    )
    return predictions


def run_directional_walkforward(
    *,
    ticker: str,
    factor_ticker: str,
    asset_receipt_path: Path,
    factor_receipt_path: Path,
    store: ParquetDatasetStore,
    output_dir: Path,
    evaluation_start: date,
    scenarios: list[BarrierScenario],
    training_sessions: int = 38,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
    baseline_fit_sessions: int = 19,
    alpha_fit_sessions: int = 10,
    fold_embargo_sessions: int = 1,
    fold_validation_sessions: int = 5,
    fold_step_sessions: int = 5,
    lookback_returns: int = 390,
    min_returns: int = 60,
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
    min_directional_touches: int = 30,
    regularization: float = 10.0,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Fit separate models per forward fold; pool only their validation predictions.

    A fold can train on dates evaluated by an earlier fold, but not on data
    from its own validation or embargo. The original withheld dates are never
    loaded. Duplicate validation decisions are forbidden, even across folds.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace research report: {output_dir}")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be unique and nonempty")
    if ticker == factor_ticker:
        raise ValueError("asset and factor must be distinct")
    if fold_step_sessions < fold_validation_sessions:
        raise ValueError("fold_step_sessions cannot overlap evaluation sessions")
    if type(min_directional_touches) is not int or min_directional_touches < 2:
        raise ValueError("min_directional_touches must be >= 2")

    asset_receipt, asset_hash = read_pinned_receipt(asset_receipt_path)
    factor_receipt, factor_hash = read_pinned_receipt(factor_receipt_path)
    permitted_entries, _, withheld = select_receipt_window(
        asset_receipt,
        ticker=ticker,
        evaluation_start=evaluation_start,
        training_sessions=training_sessions,
        validation_sessions=validation_sessions,
        holdout_sessions=holdout_sessions,
        embargo_sessions=embargo_sessions,
    )
    dates = tuple(date.fromisoformat(str(entry["session"])) for entry in permitted_entries)
    folds = staged_session_folds(
        dates,
        baseline_fit_sessions=baseline_fit_sessions,
        alpha_fit_sessions=alpha_fit_sessions,
        embargo_sessions=fold_embargo_sessions,
        validation_sessions=fold_validation_sessions,
        step_sessions=fold_step_sessions,
    )
    if not folds:
        raise ValueError("not enough permitted sessions for a staged walk-forward fold")
    factor_entries = select_pinned_factor_entries(
        factor_receipt, ticker=factor_ticker, allowed_days=set(dates)
    )
    asset = load_sessions(store, permitted_entries)
    factor = load_sessions(store, factor_entries)
    if tuple(session.day for session in asset) != dates or tuple(s.day for s in factor) != dates:
        raise ValueError("asset/factor sessions do not match pinned dates")
    date_to_index = {day: index for index, day in enumerate(dates)}

    summaries: list[dict[str, object]] = []
    for scenario in scenarios:
        pooled: list[PairedProbabilityObservation] = []
        fold_reports: list[dict[str, object]] = []
        for fold in folds:
            cutoff_index = date_to_index[fold.alpha_fit_days[-1]] + 1
            validation_end_index = date_to_index[fold.validation_days[-1]] + 1
            train_asset = asset[:cutoff_index]
            train_factor = factor[:cutoff_index]
            complete_asset = asset[:validation_end_index]
            complete_factor = factor[:validation_end_index]
            fit_days = set(fold.baseline_fit_days)
            alpha_days = set(fold.alpha_fit_days)
            train_until = train_asset[-1].bars[-1].available_at
            training_predictions = _forecasts(
                train_asset,
                scenario=scenario,
                evaluation_days=alpha_days,
                seasonal_fit_days=fit_days,
                lookback_returns=lookback_returns,
                min_returns=min_returns,
            )
            training_rows = build_directional_rows(
                asset_sessions=train_asset,
                factor_sessions=train_factor,
                forecasts=training_predictions,
                return_minutes=return_minutes,
                beta_lookback=beta_lookback,
                min_beta_observations=min_beta_observations,
            )
            touches = [
                example
                for row in training_rows
                if row.label_available_at <= train_until
                for example in [row.training_touch()]
                if example is not None
            ]
            description: dict[str, object] = {
                "fold": fold.number,
                "baseline_fit_days": [day.isoformat() for day in fold.baseline_fit_days],
                "alpha_fit_days": [day.isoformat() for day in fold.alpha_fit_days],
                "embargo_days": [day.isoformat() for day in fold.embargo_days],
                "validation_days": [day.isoformat() for day in fold.validation_days],
                "train_until": train_until.isoformat(),
                "training_touches": len(touches),
            }
            if len(touches) < min_directional_touches or len({x.outcome for x in touches}) < 2:
                description["status"] = "insufficient_training_touches"
                fold_reports.append(description)
                continue
            intercept_model = fit_conditional_direction(
                touches,
                train_until=train_until,
                version="m3-p3-offset-intercept-v1",
                include_feature=False,
                min_examples=min_directional_touches,
                regularization=regularization,
            )
            signal_model = fit_conditional_direction(
                touches,
                train_until=train_until,
                version="m3-p3-offset-reversion-v1",
                include_feature=True,
                min_examples=min_directional_touches,
                regularization=regularization,
            )
            validation_predictions = _forecasts(
                complete_asset,
                scenario=scenario,
                evaluation_days=set(fold.validation_days),
                seasonal_fit_days=fit_days,
                lookback_returns=lookback_returns,
                min_returns=min_returns,
            )
            validation_rows = build_directional_rows(
                asset_sessions=complete_asset,
                factor_sessions=complete_factor,
                forecasts=validation_predictions,
                return_minutes=return_minutes,
                beta_lookback=beta_lookback,
                min_beta_observations=min_beta_observations,
            )
            if not validation_rows:
                description["status"] = "no_aligned_validation_features"
                fold_reports.append(description)
                continue
            predictions = predict_paired_directional_rows(
                validation_rows,
                intercept_only=intercept_model,
                with_signal=signal_model,
            )
            pooled.extend(predictions)
            description.update(
                {
                    "status": "scored",
                    "validation_decisions": len(validation_rows),
                    "signal_coefficient": signal_model.coefficient,
                    "scores": score_paired_probabilities(
                        predictions, bootstrap_replicates=bootstrap_replicates
                    ),
                }
            )
            fold_reports.append(description)
        summaries.append(
            {
                "upper_barrier_pct": scenario.upper_barrier_pct,
                "lower_barrier_pct": scenario.lower_barrier_pct,
                "horizon_minutes": scenario.horizon_minutes,
                "folds": fold_reports,
                "pooled_scores": (
                    score_paired_probabilities(pooled, bootstrap_replicates=bootstrap_replicates)
                    if pooled
                    else None
                ),
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "research_only": True,
        "holdout_accessed": False,
        "ticker": ticker,
        "factor_ticker": factor_ticker,
        "asset_receipt_sha256": asset_hash,
        "factor_receipt_sha256": factor_hash,
        "asset_source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in permitted_entries
        ],
        "factor_source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in factor_entries
        ],
        "withheld_sessions_not_loaded": withheld,
        "fold_count": len(folds),
        "configuration": {
            "baseline_fit_sessions": baseline_fit_sessions,
            "alpha_fit_sessions": alpha_fit_sessions,
            "fold_embargo_sessions": fold_embargo_sessions,
            "fold_validation_sessions": fold_validation_sessions,
            "fold_step_sessions": fold_step_sessions,
            "regularization": regularization,
            "return_minutes": return_minutes,
            "beta_lookback": beta_lookback,
            "min_beta_observations": min_beta_observations,
            "min_directional_touches": min_directional_touches,
            "lookback_returns": lookback_returns,
            "min_returns": min_returns,
        },
        "scenarios": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _scenario(value: str) -> BarrierScenario:
    try:
        up, down, minutes = value.split(",")
        return BarrierScenario(float(up), float(down), int(minutes))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Use UP,DOWN,MINUTES") from exc


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
    parser.add_argument("--scenario", action="append", type=_scenario)
    parser.add_argument("--baseline-fit-sessions", type=int, default=19)
    parser.add_argument("--alpha-fit-sessions", type=int, default=10)
    parser.add_argument("--fold-validation-sessions", type=int, default=5)
    parser.add_argument("--fold-step-sessions", type=int, default=5)
    parser.add_argument("--return-minutes", type=int, default=5)
    parser.add_argument("--beta-lookback", type=int, default=120)
    parser.add_argument("--min-beta-observations", type=int, default=60)
    parser.add_argument("--regularization", type=float, default=10.0)
    args = parser.parse_args()
    report = run_directional_walkforward(
        ticker=args.ticker.upper(),
        factor_ticker=args.factor_ticker.upper(),
        asset_receipt_path=resolve_project_path(args.receipt),
        factor_receipt_path=resolve_project_path(args.factor_receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        output_dir=resolve_project_path(args.output_dir),
        evaluation_start=args.evaluation_start,
        scenarios=args.scenario
        or [
            BarrierScenario(0.002, 0.002, 5),
            BarrierScenario(0.003, 0.002, 10),
            BarrierScenario(0.005, 0.003, 15),
        ],
        baseline_fit_sessions=args.baseline_fit_sessions,
        alpha_fit_sessions=args.alpha_fit_sessions,
        fold_validation_sessions=args.fold_validation_sessions,
        fold_step_sessions=args.fold_step_sessions,
        return_minutes=args.return_minutes,
        beta_lookback=args.beta_lookback,
        min_beta_observations=args.min_beta_observations,
        regularization=args.regularization,
    )
    for summary in report["scenarios"]:
        scores = summary["pooled_scores"]
        if scores is None:
            print(f"{report['ticker']} {summary['horizon_minutes']}m: no scored folds")
            continue
        print(
            f"{report['ticker']} {summary['horizon_minutes']}m: "
            f"folds={report['fold_count']} scored={scores['scored']} "
            f"base={scores['models']['base']['brier']:.6f} "
            f"intercept={scores['models']['intercept']['brier']:.6f} "
            f"signal={scores['models']['signal']['brier']:.6f} "
            f"delta={scores['intercept_minus_signal_brier']:+.6f} "
            f"CI={scores['intercept_minus_signal_95pct_session_bootstrap']}"
        )
    print(f"Report: {args.output_dir}/validation.json")
    print("NOTICE: Development walk-forward only; final holdout untouched")


if __name__ == "__main__":
    main()

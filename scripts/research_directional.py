"""M3 development-only market-relative directional research on pinned minute sessions.

No broker access, no execution simulation, and no holdout reads. A preceding
training prefix fits the M2.5 seasonal profile; the later training prefix
fits M3. Neither uses the evaluation/embargo outcomes to fit parameters.
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
    score_directional_models,
)
from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_validation import (
    ForecastResult,
    Session,
    evaluate_sessions,
    select_receipt_window,
)
from quant_research.paths import resolve_project_path


def _forecast_rows(
    *,
    sessions: list[Session],
    evaluation_days: set[date],
    seasonality_training_days: set[date],
    scenario: BarrierScenario,
    lookback_returns: int,
    min_returns: int,
) -> list[ForecastResult]:
    forecasts, _ = evaluate_sessions(
        sessions=sessions,
        evaluation_days=evaluation_days,
        seasonality_training_days=seasonality_training_days,
        scenarios=[scenario],
        drift_modes=("price_zero",),
        volatility_methods=("seasonal_close",),
        per_scenario_stride=True,
        lookback_returns=lookback_returns,
        min_returns=min_returns,
    )
    return forecasts


def run_directional_research(
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
    seasonality_fit_sessions: int = 19,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
    lookback_returns: int = 390,
    min_returns: int = 60,
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
    min_directional_touches: int = 30,
    regularization: float = 10.0,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Fit and evaluate M3 without accessing the reserved final holdout."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace research report: {output_dir}")
    if not 1 <= seasonality_fit_sessions < training_sessions:
        raise ValueError("seasonality_fit_sessions must be inside training prefix")
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be nonempty and unique")
    if ticker == factor_ticker:
        raise ValueError("asset and market factor must be different securities")

    asset_receipt, asset_hash = read_pinned_receipt(asset_receipt_path)
    factor_receipt, factor_hash = read_pinned_receipt(factor_receipt_path)
    asset_entries, evaluation_days, withheld = select_receipt_window(
        asset_receipt,
        ticker=ticker,
        evaluation_start=evaluation_start,
        training_sessions=training_sessions,
        validation_sessions=validation_sessions,
        holdout_sessions=holdout_sessions,
        embargo_sessions=embargo_sessions,
    )
    allowed_days = {date.fromisoformat(str(item["session"])) for item in asset_entries}
    factor_entries = select_pinned_factor_entries(
        factor_receipt, ticker=factor_ticker, allowed_days=allowed_days
    )
    asset = load_sessions(store, asset_entries)
    factor = load_sessions(store, factor_entries)
    # Freeze the identical Brownian seasonal offset for M3 training and validation.
    first_training = asset[:seasonality_fit_sessions]
    last_training = asset[seasonality_fit_sessions:training_sessions]
    entire_training = asset[:training_sessions]
    if not last_training or not entire_training[-1].bars:
        raise ValueError("insufficient training sessions")
    train_until = entire_training[-1].bars[-1].available_at
    fit_days = {session.day for session in first_training}
    model_train_days = {session.day for session in last_training}

    summaries: list[dict[str, object]] = []
    for scenario in scenarios:
        train_forecasts = _forecast_rows(
            sessions=entire_training,
            evaluation_days=model_train_days,
            seasonality_training_days=fit_days,
            scenario=scenario,
            lookback_returns=lookback_returns,
            min_returns=min_returns,
        )
        train_rows = build_directional_rows(
            asset_sessions=entire_training,
            factor_sessions=factor[:training_sessions],
            forecasts=train_forecasts,
            return_minutes=return_minutes,
            beta_lookback=beta_lookback,
            min_beta_observations=min_beta_observations,
        )
        train_touches = [
            example
            for row in train_rows
            if row.label_available_at <= train_until
            for example in [row.training_touch()]
            if example is not None
        ]
        description: dict[str, object] = {
            "upper_barrier_pct": scenario.upper_barrier_pct,
            "lower_barrier_pct": scenario.lower_barrier_pct,
            "horizon_minutes": scenario.horizon_minutes,
            "m2_training_decisions": len(train_forecasts),
            "m3_training_decisions": len(train_rows),
            "m3_training_touches": len(train_touches),
        }
        if (
            len(train_touches) < min_directional_touches
            or len({example.outcome for example in train_touches}) < 2
        ):
            description["status"] = "insufficient_training_touches"
            summaries.append(description)
            continue
        intercept = fit_conditional_direction(
            train_touches,
            train_until=train_until,
            version="m3-offset-intercept-v1",
            include_feature=False,
            min_examples=min_directional_touches,
            regularization=regularization,
        )
        signal = fit_conditional_direction(
            train_touches,
            train_until=train_until,
            version="m3-offset-reversion-v1",
            include_feature=True,
            min_examples=min_directional_touches,
            regularization=regularization,
        )
        validation_forecasts = _forecast_rows(
            sessions=asset,
            evaluation_days=evaluation_days,
            seasonality_training_days=fit_days,
            scenario=scenario,
            lookback_returns=lookback_returns,
            min_returns=min_returns,
        )
        validation_rows = build_directional_rows(
            asset_sessions=asset,
            factor_sessions=factor,
            forecasts=validation_forecasts,
            return_minutes=return_minutes,
            beta_lookback=beta_lookback,
            min_beta_observations=min_beta_observations,
        )
        description.update(
            {
                "status": "scored" if validation_rows else "no_aligned_validation_features",
                "m2_validation_decisions": len(validation_forecasts),
                "m3_validation_decisions": len(validation_rows),
                "intercept_only": {"intercept": intercept.intercept},
                "with_signal": {
                    "intercept": signal.intercept,
                    "coefficient": signal.coefficient,
                    "feature_mean": signal.feature_mean,
                    "feature_scale": signal.feature_scale,
                    "training_examples": signal.training_examples,
                },
                "scores": (
                    score_directional_models(
                        validation_rows,
                        intercept_only=intercept,
                        with_signal=signal,
                        bootstrap_replicates=bootstrap_replicates,
                    )
                    if validation_rows
                    else None
                ),
            }
        )
        summaries.append(description)
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
            for entry in asset_entries
        ],
        "factor_source_versions": [
            {key: entry[key] for key in ("session", "dataset_id", "dataset_version")}
            for entry in factor_entries
        ],
        "seasonality_fit_days": sorted(day.isoformat() for day in fit_days),
        "directional_fit_days": sorted(day.isoformat() for day in model_train_days),
        "seasonality_profile_frozen_across_m3_fit_and_validation": True,
        "directional_fit_cutoff": train_until.isoformat(),
        "validation_days": sorted(day.isoformat() for day in evaluation_days),
        "withheld_sessions_not_loaded": withheld,
        "config": {
            "return_minutes": return_minutes,
            "beta_lookback": beta_lookback,
            "min_beta_observations": min_beta_observations,
            "min_directional_touches": min_directional_touches,
            "regularization": regularization,
            "lookback_returns": lookback_returns,
            "min_returns": min_returns,
            "bootstrap_replicates": bootstrap_replicates,
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
    parser.add_argument("--factor-receipt", type=Path)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", action="append", type=_scenario)
    parser.add_argument("--return-minutes", type=int, default=5)
    parser.add_argument("--beta-lookback", type=int, default=120)
    parser.add_argument("--min-beta-observations", type=int, default=60)
    parser.add_argument("--regularization", type=float, default=10.0)
    args = parser.parse_args()
    asset_path = resolve_project_path(args.receipt)
    report = run_directional_research(
        ticker=args.ticker.upper(),
        factor_ticker=args.factor_ticker.upper(),
        asset_receipt_path=asset_path,
        factor_receipt_path=resolve_project_path(args.factor_receipt)
        if args.factor_receipt
        else asset_path,
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        output_dir=resolve_project_path(args.output_dir),
        evaluation_start=args.evaluation_start,
        scenarios=args.scenario
        or [
            BarrierScenario(0.002, 0.002, 5),
            BarrierScenario(0.003, 0.002, 10),
            BarrierScenario(0.005, 0.003, 15),
        ],
        return_minutes=args.return_minutes,
        beta_lookback=args.beta_lookback,
        min_beta_observations=args.min_beta_observations,
        regularization=args.regularization,
    )
    for summary in report["scenarios"]:
        scores = summary.get("scores")
        if scores is None:
            print(f"{report['ticker']} {summary['horizon_minutes']}m: {summary['status']}")
            continue
        print(
            f"{report['ticker']} {summary['horizon_minutes']}m: "
            f"base={scores['models']['base']['brier']:.6f} "
            f"intercept={scores['models']['intercept']['brier']:.6f} "
            f"signal={scores['models']['signal']['brier']:.6f} "
            f"incremental_delta={scores['intercept_minus_signal_brier']:+.6f} "
            f"CI={scores['intercept_minus_signal_95pct_session_bootstrap']}"
        )
    print(f"Report: {args.output_dir}/validation.json")
    print("NOTICE: Development only; no final holdout or broker access")


if __name__ == "__main__":
    main()

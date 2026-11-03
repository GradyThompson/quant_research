"""Evaluate M1 mathematical first-passage calibration on pinned minute sessions.

No model fitting, trade execution or holdout reads. Historical OHLC
first-touches cannot resolve within-minute double-barrier touches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.views.pinned_minute_sessions import load_sessions
from quant_research.model.first_passage import BarrierScenario
from quant_research.model.first_passage_benchmark import (
    benchmark_scores,
    fit_historical_frequency_benchmark,
)
from quant_research.model.first_passage_diagnostics import analyze_first_passage
from quant_research.model.first_passage_seasonality import fit_intraday_profile
from quant_research.model.first_passage_validation import (
    VolatilityMethod,
    evaluate_sessions,
    select_receipt_window,
    summarize_forecasts,
)
from quant_research.paths import resolve_project_path


def _scenario(value: str) -> BarrierScenario:
    """Parse UP,DOWN,MINUTES, where barrier distances are fractions."""
    try:
        up, down, horizon = value.split(",")
        parsed = BarrierScenario(float(up), float(down), int(horizon))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Use UP,DOWN,MINUTES e.g. 0.02,0.01,60") from exc
    return parsed


def run_validation(
    *,
    ticker: str,
    receipt_path: Path,
    store: ParquetDatasetStore,
    output_dir: Path,
    evaluation_start: date,
    scenarios: list[BarrierScenario],
    decision_stride: int,
    lookback_returns: int,
    min_returns: int,
    training_sessions: int = 38,
    validation_sessions: int = 10,
    holdout_sessions: int = 10,
    embargo_sessions: int = 1,
    volatility_methods: tuple[VolatilityMethod, ...] = ("close",),
    per_scenario_stride: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace validation report: {output_dir}")
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    entries, validation_days, withheld = select_receipt_window(
        receipt,
        ticker=ticker,
        evaluation_start=evaluation_start,
        training_sessions=training_sessions,
        validation_sessions=validation_sessions,
        holdout_sessions=holdout_sessions,
        embargo_sessions=embargo_sessions,
    )
    # Holdout and post-validation embargo are not read by this function.
    sessions = load_sessions(store, entries)
    training_days = {
        date.fromisoformat(str(entry["session"])) for entry in entries[:training_sessions]
    }
    training_only = [session for session in sessions if session.day in training_days]
    warmup_embargo_days = sorted(
        session.day
        for session in sessions
        if session.day < min(validation_days) and session.day not in training_days
    )
    results, exclusions = evaluate_sessions(
        sessions=sessions,
        evaluation_days=validation_days,
        scenarios=scenarios,
        decision_stride=decision_stride,
        lookback_returns=lookback_returns,
        min_returns=min_returns,
        volatility_methods=volatility_methods,
        per_scenario_stride=per_scenario_stride,
        seasonality_training_days=training_days,
    )
    summaries = summarize_forecasts(results)
    diagnostics = analyze_first_passage(results)
    # A metadata-only/holdout-access test can have no scorable forecasts.
    # Without validation decisions there is no frequency comparison to make.
    # Do not fabricate a baseline or reject an otherwise valid empty report.
    frequency_benchmark = (
        fit_historical_frequency_benchmark(
            sessions=training_only,
            scenarios=scenarios,
            decision_stride=decision_stride,
            per_scenario_stride=per_scenario_stride,
        )
        if results
        else None
    )
    frequency_scores = (
        benchmark_scores(results, frequency_benchmark) if frequency_benchmark is not None else []
    )
    seasonality_profiles: dict[str, Any] = {}
    for method in volatility_methods:
        if not method.startswith("seasonal_"):
            continue
        profile = fit_intraday_profile(
            training_only,
            method="close" if method == "seasonal_close" else "parkinson",
        )
        seasonality_profiles[method] = {
            "training_sessions": profile.session_count,
            "factors": {
                "first_60_bars": profile.factors[0],
                "middle_240_bars": profile.factors[1],
                "last_90_bars": profile.factors[2],
            },
            "observations_by_period": profile.observation_counts,
            "training_days": sorted(day.isoformat() for day in training_days),
            "embargo_days_excluded_from_fit": [day.isoformat() for day in warmup_embargo_days],
            "shrink_observations_per_period": 390,
        }
    counts = Counter(
        (
            e.scenario.upper_barrier_pct,
            e.scenario.lower_barrier_pct,
            e.scenario.horizon_minutes,
            e.volatility_method,
            e.reason,
        )
        for e in exclusions
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "ticker": ticker,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "evaluation_start_exclusive": evaluation_start.isoformat(),
        "validation_sessions": sorted(d.isoformat() for d in validation_days),
        "parameter_training_sessions": sorted(d.isoformat() for d in training_days),
        "embargo_sessions_used_only_for_volatility_warmup": [
            day.isoformat() for day in warmup_embargo_days
        ],
        "withheld_sessions_not_loaded": withheld,
        "source_versions": [
            {
                "session": entry["session"],
                "dataset_id": entry["dataset_id"],
                "dataset_version": entry["dataset_version"],
            }
            for entry in entries
        ],
        "decision_stride": decision_stride,
        "per_scenario_stride": per_scenario_stride,
        "volatility_methods": list(volatility_methods),
        "intraday_seasonality": "three fixed RTH buckets, prior sessions only, 390-observation shrinkage"
        if any(method.startswith("seasonal_") for method in volatility_methods)
        else None,
        "lookback_returns": lookback_returns,
        "min_returns": min_returns,
        "scenarios": summaries,
        "seasonality_profiles": seasonality_profiles,
        "diagnostics": diagnostics,
        "historical_frequency_benchmark": {
            "description": "Three session periods, training-only class frequencies with pooled Dirichlet shrinkage",
            "period_prior_strength": (
                frequency_benchmark.period_prior_strength
                if frequency_benchmark is not None
                else None
            ),
            "status": ("scored" if frequency_benchmark is not None else "no_validation_forecasts"),
            "scores": frequency_scores,
        },
        "exclusions": [
            {
                "upper_barrier_pct": up,
                "lower_barrier_pct": down,
                "horizon_minutes": horizon,
                "volatility_method": method,
                "reason": reason,
                "count": count,
            }
            for (up, down, horizon, method, reason), count in sorted(counts.items())
        ],
        "total_scored_forecasts": len(results),
        "holdout_accessed": False,
        "reference_data_point_in_time_verified": bool(
            receipt.get("reference_data_point_in_time_verified", False)
        ),
        "bar_publication_times_verified": bool(
            receipt.get("minute_data_publication_time_verified", False)
        ),
        "research_only": True,
        "notes": [
            "Previous validation dates are development data, not an untouched holdout.",
            "Barrier reference is last published close, not unknown future entry open.",
            "Intra-minute double touches are excluded; such exclusions may bias calibration.",
            "No trading profits or executable fills are measured.",
            "Range volatility is an uncalibrated Parkinson estimator, not a fitted correction.",
            "Seasonal factors and historical class frequencies use designated training sessions only; embargo bars may warm up volatility but cannot fit parameters.",
            "Frequency benchmark uses training-only outcomes and is not a trading strategy.",
            "Per-scenario decision grids are nonoverlapping within each scenario but not across scenarios.",
            "Forward variance is EX-POST diagnostic data and never influences forecasts.",
            "Session-period buckets use bar positions, not verified exchange-clock timestamps.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/minute_training_receipt_20250113.json"),
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        type=_scenario,
        help="UP,DOWN,HORIZON as fractions; repeat for a grid",
    )
    parser.add_argument("--decision-stride", type=int, default=62)
    parser.add_argument("--lookback-returns", type=int, default=390)
    parser.add_argument("--min-returns", type=int, default=60)
    parser.add_argument(
        "--volatility-method",
        action="append",
        choices=("close", "parkinson", "seasonal_close", "seasonal_parkinson"),
        help="Compare close and Parkinson range volatility; repeat to run both",
    )
    parser.add_argument(
        "--per-scenario-stride",
        action="store_true",
        help="Evaluate each scenario every (horizon + 1) minutes instead of the global stride",
    )
    args = parser.parse_args()
    configs = args.scenario or [
        BarrierScenario(0.005, 0.005, 15),
        BarrierScenario(0.01, 0.005, 30),
        BarrierScenario(0.02, 0.01, 60),
        BarrierScenario(0.02, 0.01, 120),
    ]
    store = ParquetDatasetStore(resolve_project_path(args.store_root))
    report = run_validation(
        ticker=args.ticker.upper(),
        receipt_path=resolve_project_path(args.receipt),
        store=store,
        output_dir=resolve_project_path(args.output_dir),
        evaluation_start=args.evaluation_start,
        scenarios=configs,
        decision_stride=args.decision_stride,
        lookback_returns=args.lookback_returns,
        min_returns=args.min_returns,
        volatility_methods=tuple(args.volatility_method or ["close"]),
        per_scenario_stride=args.per_scenario_stride,
    )
    print(f"{report['ticker']}: mathematical first-passage calibration (development only)")
    for summary in report["scenarios"]:
        print(
            f"  up={summary['upper_barrier_pct']:.1%} down={summary['lower_barrier_pct']:.1%} "
            f"horizon={summary['horizon_minutes']}m mode={summary['drift_mode']} "
            f"vol={summary['volatility_method']}: "
            f"n={summary['scored']} Brier={summary['brier']:.6f} "
            f"log_loss={summary['log_loss']:.6f} "
            f"observed={summary['observed_counts']} "
            f"predicted={summary['predicted_mean']} "
            f"any_touch={summary['observed_any_touch_fraction']:.1%}/"
            f"{summary['predicted_any_touch_fraction']:.1%} (observed/predicted) "
            f"mean_sigma={summary['volatility_mean']:.4%}"
        )
    if report["seasonality_profiles"]:
        print("Intraday variance profiles (fit on earlier sessions only):")
        for method, profile in report["seasonality_profiles"].items():
            factors = profile["factors"]
            print(
                f"  {method}: n_sessions={profile['training_sessions']} "
                f"first={factors['first_60_bars']:.2f}x "
                f"middle={factors['middle_240_bars']:.2f}x "
                f"last={factors['last_90_bars']:.2f}x"
            )
    print("Calibration diagnostics (price_zero; Brier skill >0 beats always-neither):")
    for item in report["diagnostics"]["calibration"]:
        if item["drift_mode"] != "price_zero":
            continue
        skill = item["brier_skill_vs_neither"]
        skill_text = f"{skill:+.3f}" if skill is not None else "undefined"
        gap = item["observed_minus_expected_touch_rate"]
        interval = item["touch_rate_gap_95pct_session_bootstrap"]
        interval_text = (
            f"[{interval[0]:+.1%}, {interval[1]:+.1%}]" if interval is not None else "N/A"
        )
        print(
            f"  up={item['upper_barrier_pct']:.1%} down={item['lower_barrier_pct']:.1%} "
            f"{item['horizon_minutes']}m vol={item['volatility_method']}: "
            f"touch_gap={gap:+.1%} CI={interval_text} Brier_skill={skill_text}"
        )
    print("Training-only historical-frequency benchmark (Brier skill >0 beats frequency):")
    for row in report["historical_frequency_benchmark"]["scores"]:
        if row["drift_mode"] != "price_zero":
            continue
        skill = row["brier_skill_vs_frequency"]
        text = f"{skill:+.3f}" if skill is not None else "N/A"
        interval = row["frequency_minus_model_brier_95pct_session_bootstrap"]
        ci = f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval else "N/A"
        print(
            f"  up={row['upper_barrier_pct']:.1%} down={row['lower_barrier_pct']:.1%} "
            f"{row['horizon_minutes']}m vol={row['volatility_method']}: "
            f"model_brier={row['model_brier']:.5f} "
            f"frequency_brier={row['frequency_brier']:.5f} "
            f"skill={text} paired_delta_CI={ci}"
        )
    print("Class-level calibration (price_zero; observed/predicted):")
    for row in report["diagnostics"]["class_calibration"]:
        if row["drift_mode"] != "price_zero":
            continue
        gap_ci = row["gap_95pct_session_bootstrap"]
        ci = f"[{gap_ci[0]:+.1%}, {gap_ci[1]:+.1%}]" if gap_ci else "N/A"
        print(
            f"  up={row['upper_barrier_pct']:.1%} down={row['lower_barrier_pct']:.1%} "
            f"{row['horizon_minutes']}m vol={row['volatility_method']} "
            f"outcome={row['outcome']}: "
            f"{row['observed_fraction']:.1%}/{row['predicted_fraction']:.1%} "
            f"gap_CI={ci}"
        )
    print("Paired volatility Brier differences (positive = Parkinson better):")
    for item in report["diagnostics"]["paired_volatility"]:
        if item["drift_mode"] != "price_zero":
            continue
        interval = item["close_minus_parkinson_brier_95pct_session_bootstrap"]
        interval_text = (
            f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval is not None else "N/A"
        )
        print(
            f"  up={item['upper_barrier_pct']:.1%} down={item['lower_barrier_pct']:.1%} "
            f"{item['horizon_minutes']}m: "
            f"delta={item['close_minus_parkinson_brier']:+.5f} CI={interval_text}"
        )
    print("Paired intraday seasonality Brier differences (positive = seasonal better):")
    for item in report["diagnostics"]["paired_seasonality"]:
        if item["drift_mode"] != "price_zero":
            continue
        interval = item["baseline_minus_seasonal_brier_95pct_session_bootstrap"]
        text = f"[{interval[0]:+.5f}, {interval[1]:+.5f}]" if interval else "N/A"
        print(
            f"  up={item['upper_barrier_pct']:.1%} down={item['lower_barrier_pct']:.1%} "
            f"{item['horizon_minutes']}m vol={item['volatility_method']}: "
            f"delta={item['baseline_minus_seasonal_brier']:+.5f} CI={text}"
        )
    print("Forward volatility diagnostic (EX-POST, price_zero; forecast/future variance):")
    for item in report["diagnostics"]["forward_variance"]:
        stats = item["overall"]
        if stats["scored"] == 0:
            print(
                f"  up={item['upper_barrier_pct']:.1%} "
                f"down={item['lower_barrier_pct']:.1%} "
                f"{item['horizon_minutes']}m vol={item['volatility_method']}: "
                f"NO ALIGNED FORWARD WINDOWS "
                f"({item['without_forward_variance']} unscored)"
            )
            continue
        ratio = stats["predicted_to_realized_variance"]
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "undefined"
        print(
            f"  up={item['upper_barrier_pct']:.1%} down={item['lower_barrier_pct']:.1%} "
            f"{item['horizon_minutes']}m vol={item['volatility_method']}: "
            f"n={stats['scored']} forecast_sigma={stats['predicted_sigma']:.4%} "
            f"future_sigma={stats['realized_sigma']:.4%} variance_ratio={ratio_text}"
        )
        for period in item["by_session_period"]:
            ratio = period["predicted_to_realized_variance"]
            ratio_text = f"{ratio:.2f}x" if ratio is not None else "undefined"
            print(
                f"    {period['period']}: n={period['scored']} "
                f"variance_ratio={ratio_text} "
                f"touch={period['observed_touch_rate']:.1%}/"
                f"{period['predicted_touch_rate']:.1%} (observed/predicted)"
            )
    print(f"Exclusions: {report['exclusions']}")
    print(f"Withheld sessions not loaded: {len(report['withheld_sessions_not_loaded'])}")
    print(f"Report: {args.output_dir}/validation.json")
    print("NOTICE: Holdout untouched; OHLC ordering is ambiguous for double touches")


if __name__ == "__main__":
    main()

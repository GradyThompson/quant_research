"""Run first pinned historical AAPL rolling-model intraday backtest."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

import exchange_calendars as xcals
import pandas as pd
from run_historical_rolling_barrier import load_private_rolling_model

from quant_research.backtesting.barrier_alignment_audit import audit_barrier_alignment
from quant_research.backtesting.barrier_calibration import (
    ReferenceConvention,
    evaluate_barrier_calibration,
)
from quant_research.backtesting.forecast_diagnostics import (
    RecordingRollingBarrierModel,
    analyze_forecasts,
)
from quant_research.backtesting.minute_run_setup import (
    resolve_initial_decision_at,
    resolve_update_dataset,
)
from quant_research.backtesting.rolling_minute_backtest import run_rolling_minute_backtest
from quant_research.core import DatasetVersion
from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.storage import ParquetDatasetStore
from quant_research.strategy.barrier_decision import BarrierDecisionConfig
from quant_research.strategy.barrier_trade_plan import BarrierTradePlanConfig
from quant_research.strategy.pinned_rolling_barrier import load_following_minute_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/massive_minute_window_receipt.json")
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("private_strategies/barrier_model/rolling_model.py")
    )
    parser.add_argument("--initial-decision-at", type=datetime.fromisoformat, default=None)
    parser.add_argument("--update-session", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--update-dataset-id", default=None)
    parser.add_argument("--update-version", type=UUID, default=None)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--max-holding-minutes", type=int, default=60)
    parser.add_argument("--min-upper-probability", type=float, default=0.6)
    parser.add_argument("--max-lower-probability", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--shares", type=int, default=10)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print forecast distribution and threshold sensitivity",
    )
    parser.add_argument("--calibration", action="store_true", help="Score barrier forecasts")
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Compare with uniform and retrospective frequency baselines",
    )
    parser.add_argument(
        "--alignment-audit", action="store_true", help="Audit forecast/reference timing"
    )
    parser.add_argument(
        "--calibration-reference",
        choices=[option.value for option in ReferenceConvention],
        default=ReferenceConvention.EXECUTABLE.value,
        help="Reference for primary calibration (default: executable)",
    )
    parser.add_argument(
        "--compare-calibration-references",
        action="store_true",
        help="Report both executable and historical legacy calibration labels",
    )
    args = parser.parse_args()

    store = ParquetDatasetStore(args.store_root)
    decision_at = args.initial_decision_at or resolve_initial_decision_at(args.receipt)
    dataset_id, version = resolve_update_dataset(
        store=store,
        receipt=args.receipt,
        session=args.update_session,
        dataset_id=args.update_dataset_id,
        version=DatasetVersion(args.update_version) if args.update_version else None,
    )
    print(f"Using initialization decision: {decision_at.isoformat()}")
    print(f"Using update dataset: {dataset_id}/{version}")
    initial = load_minute_model_input(
        store=store,
        receipt_path=args.receipt,
        decision_at=decision_at,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
    )
    bars = load_following_minute_session(
        store=store,
        dataset_id=dataset_id,
        version=version,
        initial=initial,
        session=args.update_session,
    )
    session_close = (
        xcals.get_calendar("XNYS").session_close(pd.Timestamp(args.update_session)).to_pydatetime()
    )
    model = load_private_rolling_model(args.model_path)
    recorder = RecordingRollingBarrierModel(model)
    decision_config = BarrierDecisionConfig(
        min_upper_probability=args.min_upper_probability,
        max_lower_probability=args.max_lower_probability,
        min_confidence=args.min_confidence,
    )
    result = run_rolling_minute_backtest(
        model=recorder,
        initial=initial,
        bars=bars,
        session_close=session_close,
        decision_config=decision_config,
        plan_config=BarrierTradePlanConfig(
            shares=args.shares,
            upper_barrier_pct=args.upper_pct,
            lower_barrier_pct=args.lower_pct,
            max_holding_minutes=args.max_holding_minutes,
        ),
        initial_cash=args.initial_cash,
    )
    print(f"Security: {initial.security_id}")
    print(f"Evaluation session: {args.update_session}")
    print(f"Rolling forecasts: {result.forecasts}")
    print(f"Approved entry signals: {result.approved_signals}")
    print(f"Skipped by thresholds: {result.rejected_signals}")
    print(f"Skipped while position active: {result.busy_signals}")
    print(f"Skipped for insufficient cash: {result.rejected_for_cash}")
    print(f"No eligible entry: {result.unavailable_entries}")
    print(f"Completed trades: {len(result.trades)}")
    print(f"Wins: {result.wins}")
    print(f"Win rate: {result.wins / len(result.trades):.2%}" if result.trades else "Win rate: n/a")
    print(f"Commissions: ${result.total_commissions:.2f}")
    print(f"Net realized P&L: ${result.net_pnl:.2f}")
    print(f"Ending cash: ${result.ending_cash:.2f}")
    if args.diagnostics:
        diagnostics = analyze_forecasts(recorder.forecasts, config=decision_config)
        print("--- Forecast diagnostics (all predictions, including busy periods) ---")
        for label, stats in (
            ("Upper", diagnostics.upper),
            ("Lower", diagnostics.lower),
            ("Neither", diagnostics.neither),
            ("Confidence", diagnostics.confidence),
        ):
            if stats.count:
                print(
                    f"{label}: n={stats.count} min={stats.minimum:.4f} "
                    f"mean={stats.average:.4f} median={stats.median:.4f} "
                    f"max={stats.maximum:.4f}"
                )
            else:
                print(f"{label}: no values")
        print(f"Missing confidence: {diagnostics.missing_confidence}")
        print("Rejections by first failing policy condition:")
        for reason, count in sorted(diagnostics.rejection_reasons.items()):
            print(f"  {reason}: {count}")
        print("Eligible signals by minimum upper probability (other thresholds fixed):")
        for threshold, count in diagnostics.threshold_counts.items():
            print(f"  {threshold}: {count}")
        print("NOTE: threshold counts are signal diagnostics, not simulated trades")
    if args.calibration or args.compare_calibration_references or args.benchmarks:
        reference = ReferenceConvention(args.calibration_reference)
        scores = evaluate_barrier_calibration(
            forecasts=recorder.forecasts,
            bars=bars,
            upper_barrier_pct=args.upper_pct,
            lower_barrier_pct=args.lower_pct,
            max_holding_minutes=args.max_holding_minutes,
            reference_convention=reference,
        )
        print(
            f"--- Historical barrier calibration ({reference.value} open reference, no fills) ---"
        )
        print(f"Total forecasts: {scores.total_forecasts}")
        print(f"Scored: {scores.scored}")
        print(f"Ambiguous excluded: {scores.ambiguous}")
        print(f"Incomplete excluded: {scores.incomplete}")
        print(f"Multiclass Brier (lower is better): {scores.brier_score}")
        print(f"Multiclass log loss (lower is better): {scores.log_loss}")
        print(f"Observed outcome counts: {scores.observed}")
        print("Upper probability buckets (count, mean predicted, observed frequency):")
        for bucket, metrics in scores.upper_probability_buckets.items():
            print(f"  {bucket}: {metrics}")
        print("Confidence buckets (count, top-class accuracy):")
        for bucket, metrics in scores.confidence_buckets.items():
            print(f"  {bucket}: {metrics}")
        print("NOTE: overlapping forecast windows are correlated; no uncertainty estimates")
        if args.benchmarks:
            print("--- Calibration benchmarks (identical scored forecasts) ---")
            print(f"Model: Brier={scores.brier_score} log_loss={scores.log_loss}")
            for name, baseline in (
                ("Uniform", scores.uniform_baseline),
                (
                    "Retrospective frequencies (IN-SAMPLE ORACLE)",
                    scores.retrospective_frequency_baseline,
                ),
            ):
                print(
                    f"{name}: n={baseline.count} probabilities={baseline.probabilities} "
                    f"Brier={baseline.brier_score} log_loss={baseline.log_loss}"
                )
                if scores.brier_score is not None and baseline.brier_score is not None:
                    print(
                        f"  Brier improvement vs {name}: {baseline.brier_score - scores.brier_score:+.6f}"
                    )
                if scores.log_loss is not None and baseline.log_loss is not None:
                    print(
                        f"  Log-loss improvement vs {name}: {baseline.log_loss - scores.log_loss:+.6f}"
                    )
            print(
                "NOTE: retrospective frequencies use this evaluation session's outcomes; not an out-of-sample benchmark"
            )
        if args.compare_calibration_references:
            other = (
                ReferenceConvention.LEGACY
                if reference is ReferenceConvention.EXECUTABLE
                else ReferenceConvention.EXECUTABLE
            )
            comparison = evaluate_barrier_calibration(
                forecasts=recorder.forecasts,
                bars=bars,
                upper_barrier_pct=args.upper_pct,
                lower_barrier_pct=args.lower_pct,
                max_holding_minutes=args.max_holding_minutes,
                reference_convention=other,
            )
            print(f"--- Comparison calibration ({other.value}) ---")
            print(f"Scored: {comparison.scored}")
            print(f"Ambiguous: {comparison.ambiguous}")
            print(f"Incomplete: {comparison.incomplete}")
            print(f"Multiclass Brier: {comparison.brier_score}")
            print(f"Multiclass log loss: {comparison.log_loss}")
            print(f"Observed outcome counts: {comparison.observed}")
            print("NOTE: differences reflect reference conventions, not model retraining")
    if args.alignment_audit:
        audit = audit_barrier_alignment(forecasts=recorder.forecasts, bars=bars)
        print("--- Forecast-to-label alignment audit ---")
        print(f"Forecasts checked: {audit.checked}")
        print(f"Calibration/execution entry times differ: {audit.differing_entry_times}")
        print(f"Calibration/execution reference opens differ: {audit.differing_reference_prices}")
        print(f"Missing calibration entry: {audit.missing_calibration_entry}")
        print(f"Missing execution entry: {audit.missing_execution_entry}")
        for example in audit.examples:
            print(
                f"  forecast={example.forecast_at.isoformat()} "
                f"last_known_close={example.last_known_close} "
                f"calibration={example.calibration_entry_at} @ {example.calibration_open} "
                f"execution={example.execution_entry_at} @ {example.execution_open} "
                f"last-close-to-calibration-open-gap={example.reference_gap_pct}%"
            )
        print("NOTE: this audits observable timestamps, not private-model internals")
    print("PASS: first historical intraday backtest completed")


if __name__ == "__main__":
    main()

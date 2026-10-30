"""Evaluate a local private barrier model and print an entry decision."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from run_private_barrier_model import load_private_model

from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.storage import ParquetDatasetStore
from quant_research.strategy.barrier_decision import BarrierDecisionConfig
from quant_research.strategy.barrier_signal import evaluate_barrier_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/dev/massive_minute_window_receipt.json"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("private_strategies/barrier_model/model.py"),
    )
    parser.add_argument("--decision-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--max-holding-minutes", type=int, default=60)
    parser.add_argument("--min-upper-probability", type=float, default=0.6)
    parser.add_argument("--max-lower-probability", type=float, default=0.35)
    parser.add_argument("--has-open-position", action="store_true")
    parser.add_argument("--has-pending-entry", action="store_true")
    args = parser.parse_args()

    window = load_minute_model_input(
        store=ParquetDatasetStore(args.store_root),
        receipt_path=args.receipt,
        decision_at=args.decision_at,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
    )
    model = load_private_model(args.model_path)
    result = evaluate_barrier_signal(
        model,
        window,
        BarrierDecisionConfig(
            min_upper_probability=args.min_upper_probability,
            max_lower_probability=args.max_lower_probability,
        ),
        has_open_position=args.has_open_position,
        has_pending_entry=args.has_pending_entry,
    )
    print(f"Security: {result.forecast.security_id}")
    print(f"Decision time: {result.forecast.generated_at.isoformat()}")
    print(f"Input bars: {window.minute_bars.num_rows}")
    print(f"P(upper first): {result.forecast.probability_upper_first:.6f}")
    print(f"P(lower first): {result.forecast.probability_lower_first:.6f}")
    print(f"P(neither): {result.forecast.probability_neither:.6f}")
    print(f"Action: {result.decision.action.value}")
    print(f"Reason: {result.decision.reason}")
    print("PASS: private model forecast and public entry policy")


if __name__ == "__main__":
    main()

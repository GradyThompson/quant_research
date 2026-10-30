"""Evaluate one local private model against a pinned five-session input."""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import cast

from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.storage import ParquetDatasetStore
from quant_research.model.barrier import BarrierModel
from quant_research.model.barrier_runner import run_barrier_model


def load_private_model(path: Path) -> BarrierModel:
    """Load the user's local model without importing it into the public package."""
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("private_barrier_model", path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load private model module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, "PrivateBarrierModel", None)
    if not isinstance(model_class, type):
        raise ValueError("Private model must define a PrivateBarrierModel class")
    instance = model_class()
    if not callable(getattr(instance, "predict", None)):
        raise ValueError("PrivateBarrierModel must implement predict")
    return cast(BarrierModel, instance)


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
    args = parser.parse_args()

    data = load_minute_model_input(
        store=ParquetDatasetStore(args.store_root),
        receipt_path=args.receipt,
        decision_at=args.decision_at,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
    )
    model = load_private_model(args.model_path)
    result = run_barrier_model(model, data)
    print(f"Security: {result.security_id}")
    print(f"Decision: {result.generated_at.isoformat()}")
    print(f"Input minute bars: {data.minute_bars.num_rows}")
    print(f"P(upper first): {result.probability_upper_first:.6f}")
    print(f"P(lower first): {result.probability_lower_first:.6f}")
    print(f"P(neither): {result.probability_neither:.6f}")
    print("PASS: private model interface and output validation")


if __name__ == "__main__":
    main()

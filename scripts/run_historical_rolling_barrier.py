"""Run a local rolling barrier model on one pinned subsequent minute session."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.ingestion.minute_model_window import load_minute_model_input
from quant_research.data.storage import ParquetDatasetStore
from quant_research.model.rolling_barrier import RollingBarrierModel
from quant_research.strategy.barrier_decision import BarrierAction, BarrierDecisionConfig
from quant_research.strategy.pinned_rolling_barrier import load_following_minute_session
from quant_research.strategy.rolling_barrier_signal import stream_rolling_barrier_signals


@contextmanager
def _sibling_import_path(directory: Path) -> Iterator[None]:
    """Temporarily permit absolute sibling imports such as `from features import ...`."""
    location = str(directory.resolve())
    sys.path.insert(0, location)
    try:
        yield
    finally:
        sys.path.remove(location)


def load_private_rolling_model(path: Path) -> RollingBarrierModel:
    """Load the explicitly named local model, supporting sibling imports.

    Register the module before execution so dataclasses and other annotations
    can find their defining module in sys.modules during class creation.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"_private_rolling_barrier_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load private rolling model")
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        with _sibling_import_path(path.parent):
            spec.loader.exec_module(module)
        cls = getattr(module, "PrivateRollingBarrierModel", None)
        if not isinstance(cls, type):
            raise ValueError("Model must define PrivateRollingBarrierModel")
        model = cls()
        for name in ("initialize", "update", "forecast"):
            if not callable(getattr(model, name, None)):
                raise ValueError(f"Private model is missing {name}()")
        return cast(RollingBarrierModel, model)
    except BaseException:
        if prior is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = prior
        raise


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
        default=Path("private_strategies/barrier_model/rolling_model.py"),
    )
    parser.add_argument("--initial-decision-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--update-session", type=date.fromisoformat, required=True)
    parser.add_argument("--update-dataset-id", required=True)
    parser.add_argument("--update-version", type=UUID, required=True)
    parser.add_argument("--upper-pct", type=float, default=0.005)
    parser.add_argument("--lower-pct", type=float, default=0.004)
    parser.add_argument("--max-holding-minutes", type=int, default=60)
    parser.add_argument("--min-upper-probability", type=float, default=0.6)
    parser.add_argument("--max-lower-probability", type=float, default=0.35)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Optional minimum forecast confidence in [0, 1]; omitted ignores confidence.",
    )
    args = parser.parse_args()

    config = BarrierDecisionConfig(
        min_upper_probability=args.min_upper_probability,
        max_lower_probability=args.max_lower_probability,
        min_confidence=args.min_confidence,
    )
    store = ParquetDatasetStore(args.store_root)
    initial = load_minute_model_input(
        store=store,
        receipt_path=args.receipt,
        decision_at=args.initial_decision_at,
        upper_barrier_pct=args.upper_pct,
        lower_barrier_pct=args.lower_pct,
        max_holding_minutes=args.max_holding_minutes,
    )
    updates = load_following_minute_session(
        store=store,
        dataset_id=DatasetId(args.update_dataset_id),
        version=DatasetVersion(args.update_version),
        initial=initial,
        session=args.update_session,
    )
    model = load_private_rolling_model(args.model_path)
    actions: Counter[BarrierAction] = Counter()
    reasons: Counter[str] = Counter()
    last_decision: datetime | None = None
    with_confidence = 0
    for result in stream_rolling_barrier_signals(
        model=model,
        initial=initial,
        new_bars=updates,
        config=config,
    ):
        actions[result.decision.action] += 1
        reasons[result.decision.reason] += 1
        with_confidence += result.forecast.confidence is not None
        last_decision = result.forecast.generated_at

    print(f"Security: {initial.security_id}")
    print(f"Initialization bars: {initial.minute_bars.num_rows}")
    print(f"New bars processed: {updates.num_rows}")
    print(f"Minimum confidence: {config.min_confidence}")
    print(f"Forecasts with confidence: {with_confidence}")
    print(f"ENTER_LONG signals: {actions[BarrierAction.ENTER_LONG]}")
    print(f"SKIP signals: {actions[BarrierAction.SKIP]}")
    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")
    if last_decision is not None:
        print(f"Last decision: {last_decision.isoformat()}")
    print("PASS: historical rolling forecasts and research decisions")
    print("NOTE: exposure flags are constant; these are signals, not simulated trades")


if __name__ == "__main__":
    main()

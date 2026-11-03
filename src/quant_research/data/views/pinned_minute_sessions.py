"""Load immutable pinned OHLCV sessions with identity and row-count checks."""

from __future__ import annotations

import json
from datetime import date
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage import ParquetDatasetStore
from quant_research.model.first_passage_validation import MinuteBar, Session


def load_sessions(
    store: ParquetDatasetStore,
    entries: list[dict[str, object]],
) -> list[Session]:
    """Read exactly and only pinned entries approved by select_receipt_window."""
    sessions: list[Session] = []
    for entry in entries:
        dataset_id = DatasetId(str(entry["dataset_id"]))
        version = DatasetVersion(UUID(str(entry["dataset_version"])))
        table = store.read(dataset_id, version)
        if table.num_rows == 0 or table.num_rows != entry["rows"]:
            raise ValueError(f"Pinned row-count mismatch: {dataset_id}")
        columns = {"timestamp", "available_at", "security_id", "open", "high", "low", "close"}
        if not columns.issubset(table.column_names):
            raise ValueError(f"Missing minute columns: {dataset_id}")
        selected = columns | ({"volume"} if "volume" in table.column_names else set())
        rows = table.select(sorted(selected)).to_pylist()
        if {row["security_id"] for row in rows} != {entry["security_id"]}:
            raise ValueError(f"Pinned identity mismatch: {dataset_id}")
        day = date.fromisoformat(str(entry["session"]))
        bars = tuple(
            MinuteBar(
                timestamp=row["timestamp"],
                available_at=row["available_at"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if row.get("volume") is not None else None,
            )
            for row in rows
        )
        if any(bar.timestamp.date() != day for bar in bars):
            raise ValueError(f"Pinned session-date mismatch: {dataset_id}")
        sessions.append(Session(day, bars))
    return sessions


def read_pinned_receipt(path: Path) -> tuple[dict[str, object], str]:
    """Read a pinned receipt and preserve its exact content hash for provenance."""
    payload = path.read_bytes()
    receipt: dict[str, object] = json.loads(payload)
    return receipt, sha256(payload).hexdigest()


def select_pinned_factor_entries(
    receipt: dict[str, object], *, ticker: str, allowed_days: set[date]
) -> list[dict[str, object]]:
    """Validate exact factor-day coverage BEFORE loading any pinned datasets."""
    entries = receipt.get("entries")
    if receipt.get("schema_version") != 1 or not isinstance(entries, dict):
        raise ValueError("factor receipt must use pinned schema_version=1")
    selected: list[dict[str, object]] = []
    for entry in entries.values():
        if not isinstance(entry, dict):
            raise ValueError("invalid factor receipt entry")
        if (
            entry.get("ticker") == ticker
            and date.fromisoformat(str(entry["session"])) in allowed_days
        ):
            selected.append(entry)
    selected.sort(key=lambda item: str(item["session"]))
    actual = [date.fromisoformat(str(item["session"])) for item in selected]
    if len(selected) != len(allowed_days) or set(actual) != allowed_days:
        missing = sorted(allowed_days - set(actual))
        raise ValueError(
            f"missing pinned {ticker} factor sessions: {len(missing)} permitted dates "
            f"(first missing: {missing[0] if missing else 'duplicate or unexpected dates'}). "
            "Run scripts/ingest_research_factor.py with --ticker and --factor-receipt; "
            "then pass --factor-receipt to research_directional.py. "
            "Do not include holdout dates."
        )
    if len({str(item["security_id"]) for item in selected}) != 1:
        raise ValueError("factor security identity changes in pinned period")
    return selected

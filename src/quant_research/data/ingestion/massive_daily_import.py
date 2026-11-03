"""Import verified Massive day aggregates using a dated security-master snapshot."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.storage import DatasetStore
from quant_research.security_master.resolver import SecurityResolver

_FIELDS: list[pa.Field] = [
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("mic", pa.string(), nullable=False),
    pa.field("session", pa.date32(), nullable=False),
    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("open", pa.float64(), nullable=False),
    pa.field("high", pa.float64(), nullable=False),
    pa.field("low", pa.float64(), nullable=False),
    pa.field("close", pa.float64(), nullable=False),
    pa.field("volume", pa.int64(), nullable=False),
]
_SCHEMA = pa.schema(_FIELDS)
_MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class ImportAudit:
    source_rows: int
    mapped_rows: int
    unmapped_tickers: int
    ambiguous_tickers: int
    invalid_rows: int
    duplicate_tickers: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ticker_index(
    resolver: SecurityResolver, *, session: date
) -> dict[str, set[tuple[SecurityId, str]]]:
    """Return all visible listing claims without guessing exchange from a flat-file ticker."""
    valid_at = datetime.combine(session, time(12), tzinfo=UTC)
    index: dict[str, set[tuple[SecurityId, str]]] = defaultdict(set)
    for listing in resolver.listings:
        if listing.validity.contains(valid_at):
            index[listing.ticker].add((listing.security_id, listing.mic))
    return index


def parse_daily_flat_file(
    *, source: Path, session: date, resolver: SecurityResolver
) -> tuple[pa.Table, ImportAudit]:
    """Validate and map an entire compressed daily file; never infer missing identities."""
    index = _ticker_index(resolver, session=session)
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    duplicate_keys: set[str] = set()
    counts: Counter[str] = Counter()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ticker", "open", "high", "low", "close", "volume", "window_start"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing Massive flat-file fields: {sorted(required - set(reader.fieldnames or []))}"
            )
        for raw in reader:
            counts["source_rows"] += 1
            ticker = raw["ticker"]
            if not ticker or ticker in seen:
                counts["duplicate_tickers"] += 1
                duplicate_keys.add(ticker)
                continue
            seen.add(ticker)
            matches = index.get(ticker, set())
            if not matches:
                counts["unmapped_tickers"] += 1
                continue
            if len(matches) != 1:
                counts["ambiguous_tickers"] += 1
                continue
            try:
                opening, high, low, close = (
                    float(raw[key]) for key in ("open", "high", "low", "close")
                )
                volume = int(raw["volume"])
                from math import isfinite

                if (
                    not all(isfinite(p) and p > 0 for p in (opening, high, low, close))
                    or not low <= min(opening, close) <= max(opening, close) <= high
                    or volume < 0
                ):
                    raise ValueError("Invalid OHLCV")
                epoch_ns = int(raw["window_start"])
                timestamp = datetime.fromtimestamp(epoch_ns / 1_000_000_000, tz=UTC)
                market_timestamp = timestamp.astimezone(_MARKET_TZ)
                if market_timestamp.date() != session or market_timestamp.time() != time(0):
                    raise ValueError("Unexpected session timestamp")
            except (TypeError, ValueError, OverflowError):
                counts["invalid_rows"] += 1
                continue
            security_id, mic = next(iter(matches))
            rows.append(
                {
                    "security_id": str(security_id),
                    "ticker": ticker,
                    "mic": mic,
                    "session": session,
                    "timestamp": timestamp,
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
    # A repeated vendor ticker invalidates *every* matching record, not just later ones.
    rows = [row for row in rows if row["ticker"] not in duplicate_keys]
    rows.sort(key=lambda row: (str(row["security_id"]), str(row["ticker"])))
    if duplicate_keys:
        raise ValueError("Duplicate ticker rows in source; refusing ambiguous import")
    audit = ImportAudit(
        source_rows=counts["source_rows"],
        mapped_rows=len(rows),
        unmapped_tickers=counts["unmapped_tickers"],
        ambiguous_tickers=counts["ambiguous_tickers"],
        invalid_rows=counts["invalid_rows"],
        duplicate_tickers=counts["duplicate_tickers"],
    )
    if audit.source_rows != sum(
        (
            audit.mapped_rows,
            audit.unmapped_tickers,
            audit.ambiguous_tickers,
            audit.invalid_rows,
            audit.duplicate_tickers,
        )
    ):
        raise AssertionError("Unaccounted flat-file rows")
    if not rows:
        raise ValueError(f"No mapped bars; audit={audit}")
    return pa.Table.from_pylist(rows, schema=_SCHEMA), audit


def import_verified_daily_file(
    *,
    store: DatasetStore,
    resolver: SecurityResolver,
    session: date,
    source_manifest: Path,
    source_sha256: str,
    snapshot_version: DatasetVersion,
    output_manifest: Path,
) -> dict[str, object]:
    """Publish one immutable dataset and its provenance, with fail-closed reimports."""
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    if raw.get("key") != f"us_stocks_sip/day_aggs_v1/{session:%Y/%m/%Y-%m-%d}.csv.gz":
        raise ValueError("Flat-file source manifest belongs to another session")
    candidates = [item for item in raw.get("revisions", []) if item.get("sha256") == source_sha256]
    if len(candidates) != 1:
        raise ValueError("Source SHA-256 must identify exactly one verified revision")
    revision = candidates[0]
    source = Path(revision["path"])
    if _sha256(source) != source_sha256:
        raise ValueError("Raw source checksum mismatch")
    identity = {
        "session": session.isoformat(),
        "sha256": source_sha256,
        "snapshot_version": str(snapshot_version),
    }
    if output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Output manifest refers to different source or security master")
        dataset_id = DatasetId(existing["dataset_id"])
        version = DatasetVersion(UUID(existing["dataset_version"]))
        if not store.exists(dataset_id, version):
            raise ValueError("Pinned imported dataset missing")
        return existing
    table, audit = parse_daily_flat_file(source=source, session=session, resolver=resolver)
    if audit.source_rows != int(revision["rows"]):
        raise ValueError("Flat-file record count differs from source manifest")
    dataset_id = DatasetId(f"market.ohlcv.daily.us.s{session:%Y%m%d}")
    metadata = store.write(dataset_id, table)
    result: dict[str, object] = {
        **identity,
        "dataset_id": str(metadata.dataset_id),
        "dataset_version": str(metadata.version),
        "source_manifest": str(source_manifest.resolve()),
        "audit": asdict(audit),
        "reference_data_point_in_time_verified": False,
        "source_adjusted": False,
    }
    _write_json_atomic(output_manifest, result)
    return result

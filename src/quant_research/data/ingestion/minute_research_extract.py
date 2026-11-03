"""Fast, point-in-time selected-ticker extraction for research minute flat files.

A market-wide Massive minute CSV is much larger than a research portfolio.
When the first field is ``ticker``, reject other symbols before running the
CSV decoder. The fallback supports arbitrary column order. No change to the
separate, general-purpose historical ingestion parser is required.

The optional extraction cache is keyed by the VERIFIED source SHA-256, symbol
set, publication delay, and parser version. The caller MUST verify the source
manifest before using this cache; a cache hit does not validate the raw gzip.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from datetime import UTC, date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Callable, cast

_REQUIRED = {"ticker", "window_start", "open", "high", "low", "close", "volume"}
_CACHE_VERSION = 1
MinuteRows = dict[str, list[dict[str, object]]]
MinuteParser = Callable[..., MinuteRows]


def parse_selected_minutes_fast(
    source: Path,
    *,
    session: date,
    tickers: set[str],
    opening: datetime,
    closing: datetime,
    publication_lag_seconds: int,
    ingested_at: datetime,
) -> MinuteRows:
    """Read a market-wide gzip once, decoding only requested ticker rows.

    Massive's CSV uses single-line numeric records and the ticker field is
    typically first. A generic csv.reader fallback handles other header orders.
    Identical price, duplicate, time, and publication checks to the standard
    ingestion parser are applied to all *selected* rows.
    """
    del session  # Kept for a drop-in-compatible parsing interface.
    records: MinuteRows = {ticker: [] for ticker in tickers}
    seen: set[tuple[str, datetime]] = set()
    lag = timedelta(seconds=publication_lag_seconds)
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Minute aggregate fields missing") from exc
        missing = _REQUIRED - set(header)
        if missing:
            raise ValueError(f"Minute aggregate fields missing: {sorted(missing)}")
        indexes = {name: header.index(name) for name in _REQUIRED}

        def add_selected(columns: list[str]) -> None:
            ticker = columns[indexes["ticker"]]
            if ticker not in tickers:
                return
            start_ns = int(columns[indexes["window_start"]])
            stamp = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=UTC) + timedelta(minutes=1)
            if not opening < stamp <= closing:
                return
            key = (ticker, stamp)
            if key in seen:
                raise ValueError(f"Duplicate minute bar: {ticker} {stamp}")
            seen.add(key)
            prices = [float(columns[indexes[name]]) for name in ("open", "high", "low", "close")]
            volume = int(columns[indexes["volume"]])
            if (
                not all(isfinite(p) and p > 0 for p in prices)
                or not prices[2]
                <= min(prices[0], prices[3])
                <= max(prices[0], prices[3])
                <= prices[1]
                or volume < 0
            ):
                raise ValueError(f"Invalid OHLCV: {ticker} {stamp}")
            records[ticker].append(
                {
                    "timestamp": stamp,
                    "available_at": stamp + lag,
                    "ingested_at": ingested_at,
                    "open": prices[0],
                    "high": prices[1],
                    "low": prices[2],
                    "close": prices[3],
                    "volume": volume,
                    "currency": "USD",
                }
            )

        if indexes["ticker"] == 0:
            # Numeric market aggregates cannot have embedded newlines. Parsing
            # all rows into dicts here can dominate a multi-year research run.
            for line in handle:
                first = line.partition(",")[0]
                if first in tickers:
                    add_selected(next(csv.reader([line])))
                elif first.startswith('"'):
                    # Preserve valid quoted ticker fields without making the
                    # common path pay for a full CSV parse.
                    add_selected(next(csv.reader([line])))
        else:
            for columns in reader:
                add_selected(columns)

    for rows in records.values():
        rows.sort(key=lambda row: cast(datetime, row["timestamp"]))
    return records


def _cache_path(
    *,
    cache_root: Path,
    session: date,
    tickers: set[str],
    source_sha256: str,
    publication_lag_seconds: int,
) -> Path:
    identity = {
        "version": _CACHE_VERSION,
        "day": session.isoformat(),
        "source_sha256": source_sha256,
        "tickers": sorted(tickers),
        "publication_lag_seconds": publication_lag_seconds,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return cache_root / f"{session:%Y}" / f"{session:%m}" / f"{session}-{digest}.json.gz"


def load_or_parse_selected_minutes(
    source: Path,
    *,
    cache_root: Path,
    source_sha256: str,
    session: date,
    tickers: set[str],
    opening: datetime,
    closing: datetime,
    publication_lag_seconds: int,
    ingested_at: datetime,
    parser: MinuteParser = parse_selected_minutes_fast,
) -> tuple[MinuteRows, bool]:
    """Return rows and cache-hit flag. Caller verifies the source SHA first.

    A corrupt/incomplete cache is never treated as valid; it raises rather than
    silently falling back to a partial or unrelated dataset.
    """
    path = _cache_path(
        cache_root=cache_root,
        session=session,
        tickers=tickers,
        source_sha256=source_sha256,
        publication_lag_seconds=publication_lag_seconds,
    )
    if path.is_file():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            payload.get("version") != _CACHE_VERSION
            or payload.get("source_sha256") != source_sha256
            or payload.get("session") != session.isoformat()
            or payload.get("tickers") != sorted(tickers)
            or payload.get("publication_lag_seconds") != publication_lag_seconds
        ):
            raise ValueError(f"research extraction cache identity mismatch: {path}")
        rows: MinuteRows = {}
        for ticker in tickers:
            entries = payload["bars"][ticker]
            rows[ticker] = []
            for entry in entries:
                stamp = datetime.fromisoformat(entry[0])
                if stamp.tzinfo is None or not opening < stamp <= closing:
                    raise ValueError(f"research extraction cache invalid timestamp: {path}")
                rows[ticker].append(
                    {
                        "timestamp": stamp,
                        "available_at": stamp + timedelta(seconds=publication_lag_seconds),
                        "ingested_at": ingested_at,
                        "open": float(entry[1]),
                        "high": float(entry[2]),
                        "low": float(entry[3]),
                        "close": float(entry[4]),
                        "volume": int(entry[5]),
                        "currency": "USD",
                    }
                )
        return rows, True

    rows = parser(
        source,
        session=session,
        tickers=tickers,
        opening=opening,
        closing=closing,
        publication_lag_seconds=publication_lag_seconds,
        ingested_at=ingested_at,
    )
    payload = {
        "version": _CACHE_VERSION,
        "session": session.isoformat(),
        "source_sha256": source_sha256,
        "tickers": sorted(tickers),
        "publication_lag_seconds": publication_lag_seconds,
        "bars": {
            ticker: [
                [
                    cast(datetime, row["timestamp"]).isoformat(),
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                ]
                for row in entries
            ]
            for ticker, entries in rows.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = handle.name
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return rows, False

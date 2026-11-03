"""Source-pinned market-wide OHLCV summaries for event-driven research.

This is intentionally an *instrument-type-unverified* research screen. Raw
minute flat files alone do not identify historical common-stock listings.
Every daily cache is tied to the SHA256 of a previously verified source.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

# The independent daily summary can be calculated in a worker process.
# Keep this worker in an importable module (not a CLI script) so it works
# with multiprocessing start methods that require importing worker functions.
SummaryJob = tuple[date, Path, str, Path, datetime, datetime]

NS_MINUTE = 60_000_000_000
VERSION = 2


@dataclass(frozen=True, slots=True)
class DailySummary:
    opening_volume: int
    opening_high: float
    opening_low: float
    opening_open: float
    opening_close: float
    last_close: float
    dollar_volume: float
    close_eligible: bool = True


SummaryResult = tuple[date, dict[str, DailySummary], bool]


def summarize_market_minutes(
    path: Path, *, opening: datetime, closing: datetime
) -> dict[str, DailySummary]:
    """Scan a gzip once; retain ~one summary per symbol, not all minute bars.

    Source minute timestamps denote bar *start*. The first 15 bars occupy
    [market open, market open + 15 minutes). All daily dollar volume is a
    historical liquidity proxy; it is never used for same-day decisions.
    """
    start_ns = int(opening.timestamp()) * 1_000_000_000
    stop_ns = int(closing.timestamp()) * 1_000_000_000
    raw: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
        required = {"ticker", "window_start", "open", "high", "low", "close", "volume"}
        if header is None or not required.issubset(set(header)):
            raise ValueError("missing Massive minute columns")
        loc = {name: header.index(name) for name in required}
        for line in handle:
            columns = next(csv.reader([line])) if '"' in line else line.rstrip("\r\n").split(",")
            if len(columns) != len(header):
                raise ValueError("invalid market-wide minute CSV row")
            ticker = columns[loc["ticker"]]
            # Do not mistake ETFs/ADRs for common stock: this is only a
            # syntactic screen. Historical identity verification is later.
            if not ticker.isascii() or not ticker.isalpha() or not ticker.isupper():
                continue
            stamp_ns = int(columns[loc["window_start"]])
            if not start_ns <= stamp_ns < stop_ns:
                continue
            minute, remainder = divmod(stamp_ns - start_ns, NS_MINUTE)
            if remainder:
                raise ValueError(f"non-minute aligned bar: {ticker}")
            op = float(columns[loc["open"]])
            hi = float(columns[loc["high"]])
            lo = float(columns[loc["low"]])
            cl = float(columns[loc["close"]])
            vol = int(columns[loc["volume"]])
            if (
                not all(isfinite(v) and v > 0 for v in (op, hi, lo, cl))
                or lo > min(op, cl)
                or hi < max(op, cl)
                or vol < 0
            ):
                raise ValueError(f"invalid market minute OHLCV: {ticker}")
            row = raw.setdefault(
                ticker,
                {
                    "mask": 0,
                    "ov": 0,
                    "hi": 0.0,
                    "lo": float("inf"),
                    "open": None,
                    "close15": None,
                    "last_minute": -1,
                    "last_close": 0.0,
                    "dollar": 0.0,
                },
            )
            row["dollar"] += cl * vol
            if minute > row["last_minute"]:
                row["last_minute"] = minute
                row["last_close"] = cl
            if minute < 15:
                bit = 1 << minute
                if row["mask"] & bit:
                    raise ValueError(f"duplicate opening minute: {ticker} {minute}")
                row["mask"] |= bit
                row["ov"] += vol
                row["hi"] = max(row["hi"], hi)
                row["lo"] = min(row["lo"], lo)
                if minute == 0:
                    row["open"] = op
                if minute == 14:
                    row["close15"] = cl
    summary: dict[str, DailySummary] = {}
    total_minutes = (stop_ns - start_ns) // NS_MINUTE
    for ticker, row in raw.items():
        # Do not filter today's candidates on whether the stock survives to
        # the close: that is FUTURE information at 09:46. Later only complete
        # near-close summaries are admitted into the historical baseline.
        if row["mask"] != (1 << 15) - 1:
            continue
        summary[ticker] = DailySummary(
            opening_volume=row["ov"],
            opening_high=row["hi"],
            opening_low=row["lo"],
            opening_open=row["open"],
            opening_close=row["close15"],
            last_close=row["last_close"],
            dollar_volume=row["dollar"],
            close_eligible=row["last_minute"] >= total_minutes - 5,
        )
    return summary


def load_or_summarize_market(
    source: Path,
    *,
    cache_root: Path,
    day: date,
    source_sha256: str,
    opening: datetime,
    closing: datetime,
) -> tuple[dict[str, DailySummary], bool]:
    """Cache by verified source revision, session, and parser version."""
    digest = hashlib.sha256(
        json.dumps([VERSION, day.isoformat(), source_sha256]).encode()
    ).hexdigest()[:24]
    path = cache_root / f"{day:%Y}" / f"{day}-{digest}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("identity") != [VERSION, day.isoformat(), source_sha256]:
            raise ValueError("market summary cache identity mismatch")
        return (
            {symbol: DailySummary(**values) for symbol, values in payload["symbols"].items()},
            True,
        )
    summaries = summarize_market_minutes(source, opening=opening, closing=closing)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            temp = f.name
        with gzip.open(temp, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "identity": [VERSION, day.isoformat(), source_sha256],
                    "symbols": {key: vars_for_json(val) for key, val in summaries.items()},
                },
                handle,
                separators=(",", ":"),
            )
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)
    return summaries, False


def vars_for_json(value: DailySummary) -> dict[str, int | float | bool]:
    """Explicit dataclass serialization; works with frozen/slotted records."""
    return {name: getattr(value, name) for name in DailySummary.__dataclass_fields__}


def summarize_market_job(job: SummaryJob) -> SummaryResult:
    """Source-keyed daily summary task for bounded multiprocessing."""
    day, source, sha, cache_root, opening, closing = job
    summaries, cache_hit = load_or_summarize_market(
        source,
        cache_root=cache_root,
        day=day,
        source_sha256=sha,
        opening=opening,
        closing=closing,
    )
    return day, summaries, cache_hit
